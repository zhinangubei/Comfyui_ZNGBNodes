"""ZNGB custom nodes.

A small set of null-tolerant nodes. The goal is that when a video URL is empty
(None), downstream nodes (get video components / image batch multi /
audio concat multi) can still run without raising, outputting null values so the
rest of a video pipeline can continue and reach a final video combine.
"""

from __future__ import annotations

import os
import sys
import math
import json
import shutil
import hashlib
import datetime
import subprocess
import urllib.parse
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio

import folder_paths
import comfy.model_management as model_management
from comfy.utils import common_upscale
from comfy_api.latest import InputImpl, Types


MAX_RESOLUTION = 16384

_LAMA_INPAINT_MODEL = None
_LAMA_INPAINT_MODEL_DEVICE = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMPTY_URL_VALUES = {"", "none", "null", "nan"}


def _is_empty_url(url) -> bool:
    """Return True when the url should be treated as "no input"."""
    if url is None:
        return True
    if not isinstance(url, str):
        return False
    return url.strip().lower() in _EMPTY_URL_VALUES


def _download_video(url: str) -> str:
    """Download a remote video into the ComfyUI input directory and return its path.

    The file is cached by a hash of the URL so repeated runs don't re-download.
    Only http/https URLs are allowed.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)")

    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)

    ext = os.path.splitext(parsed.path)[1]
    if not ext or len(ext) > 5:
        ext = ".mp4"

    name = "zngb_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ext
    dest = os.path.join(input_dir, name)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-ZNGBNodes/1.0"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp, dest)
    return dest


def _download_audio(url: str) -> str:
    """Download a remote audio file into the input directory (cached by URL hash)."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r} (only http/https allowed)")

    input_dir = folder_paths.get_input_directory()
    os.makedirs(input_dir, exist_ok=True)

    ext = os.path.splitext(parsed.path)[1]
    if not ext or len(ext) > 5:
        ext = ".mp3"

    name = "zngb_" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:16] + ext
    dest = os.path.join(input_dir, name)

    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        return dest

    req = urllib.request.Request(url, headers={"User-Agent": "ComfyUI-ZNGBNodes/1.0"})
    tmp = dest + ".part"
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as f:
        shutil.copyfileobj(resp, f)
    os.replace(tmp, dest)
    return dest


def _parse_pad_color(pad_color, channels):
    """Parse a "R, G, B" string into a list of 0-1 floats, broadcast to `channels`."""
    parts = [p.strip() for p in str(pad_color).replace(";", ",").split(",") if p.strip() != ""]
    try:
        vals = [float(p) / 255.0 for p in parts]
    except ValueError:
        vals = [0.0]
    if not vals:
        vals = [0.0]
    if len(vals) == 1:
        vals = vals * channels
    return vals


def _resize_image_tensor(image, width, height, upscale_method, keep_proportion,
                         pad_color, crop_position, divisible_by, device="cpu"):
    """Resize an IMAGE tensor [B,H,W,C]; shared by Resize Image and Video Clip."""
    B, H, W, C = image.shape
    dev = model_management.get_torch_device() if device == "gpu" else torch.device("cpu")
    img = image.to(dev)

    pad_left = pad_right = pad_top = pad_bottom = 0

    if keep_proportion in ("resize", "pad"):
        if width == 0 and height == 0:
            new_w, new_h = W, H
        elif width == 0:
            ratio = height / H
            new_w, new_h = round(W * ratio), height
        elif height == 0:
            ratio = width / W
            new_w, new_h = width, round(H * ratio)
        else:
            ratio = min(width / W, height / H)
            new_w, new_h = round(W * ratio), round(H * ratio)

        if keep_proportion == "pad":
            target_w = width if width else new_w
            target_h = height if height else new_h
            if crop_position == "center":
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
            elif crop_position == "top":
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                pad_top = 0
                pad_bottom = target_h - new_h
            elif crop_position == "bottom":
                pad_left = (target_w - new_w) // 2
                pad_right = target_w - new_w - pad_left
                pad_top = target_h - new_h
                pad_bottom = 0
            elif crop_position == "left":
                pad_left = 0
                pad_right = target_w - new_w
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
            elif crop_position == "right":
                pad_left = target_w - new_w
                pad_right = 0
                pad_top = (target_h - new_h) // 2
                pad_bottom = target_h - new_h - pad_top
            pad_left, pad_right = max(0, pad_left), max(0, pad_right)
            pad_top, pad_bottom = max(0, pad_top), max(0, pad_bottom)

        width, height = new_w, new_h
    else:
        if width == 0:
            width = W
        if height == 0:
            height = H

    if divisible_by > 1:
        width = width - (width % divisible_by)
        height = height - (height % divisible_by)

    if keep_proportion == "crop":
        old_h, old_w = img.shape[-3], img.shape[-2]
        old_aspect = old_w / old_h
        new_aspect = width / height
        if old_aspect > new_aspect:
            crop_w, crop_h = round(old_h * new_aspect), old_h
        else:
            crop_w, crop_h = old_w, round(old_w / new_aspect)
        if crop_position == "center":
            x, y = (old_w - crop_w) // 2, (old_h - crop_h) // 2
        elif crop_position == "top":
            x, y = (old_w - crop_w) // 2, 0
        elif crop_position == "bottom":
            x, y = (old_w - crop_w) // 2, old_h - crop_h
        elif crop_position == "left":
            x, y = 0, (old_h - crop_h) // 2
        elif crop_position == "right":
            x, y = old_w - crop_w, (old_h - crop_h) // 2
        img = img.narrow(-2, x, crop_w).narrow(-3, y, crop_h)

    img = common_upscale(img.movedim(-1, 1), width, height, upscale_method, "disabled").movedim(1, -1)

    if keep_proportion == "pad" and (pad_left or pad_right or pad_top or pad_bottom):
        color = _parse_pad_color(pad_color, C)
        padded_h = height + pad_top + pad_bottom
        padded_w = width + pad_left + pad_right
        canvas = torch.empty((img.shape[0], padded_h, padded_w, C), dtype=img.dtype, device=img.device)
        for ch in range(C):
            canvas[..., ch] = color[ch] if ch < len(color) else 0.0
        canvas[:, pad_top:pad_top + height, pad_left:pad_left + width, :] = img
        img = canvas

    return img.cpu()


# ---------------------------------------------------------------------------
# 1. Load Video From Url
# ---------------------------------------------------------------------------

class ZNGBLoadVideoFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": True}),
                "keep_audio": ("BOOLEAN", {"default": True}),
                "trim_time": ("FLOAT", {"default": 0, "min": 0, "max": 100000, "step": 0.1,
                                        "tooltip": "Keep only the first N seconds (0 = full video)."}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("VIDEO",)
    FUNCTION = "load"
    CATEGORY = "ZNGBNodes/video"
    DESCRIPTION = "Download a video from an http(s) url. If url is empty/None, output is null."

    def load(self, url, keep_audio, trim_time):
        if _is_empty_url(url):
            return (None,)

        path = _download_video(url.strip())
        duration = float(trim_time) if trim_time and float(trim_time) > 0 else 0
        video = InputImpl.VideoFromFile(path, start_time=0, duration=duration)

        if not keep_audio:
            comps = video.get_components()
            video = InputImpl.VideoFromComponents(
                Types.VideoComponents(
                    images=comps.images,
                    frame_rate=comps.frame_rate,
                    audio=None,
                )
            )

        return (video,)

    @classmethod
    def IS_CHANGED(cls, url, keep_audio, trim_time):
        return f"{url}|{keep_audio}|{trim_time}"


# ---------------------------------------------------------------------------
# 1b. Load Audio From Url
# ---------------------------------------------------------------------------

class ZNGBLoadAudioFromUrl:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = "ZNGBNodes/audio"
    DESCRIPTION = "Download an audio file from an http(s) url. If url is empty/None, output is null."

    def load(self, url):
        if _is_empty_url(url):
            return (None,)

        path = _download_audio(url.strip())
        from comfy_extras.nodes_audio import load as _load_audio
        waveform, sr = _load_audio(path)       # [C, T], float32
        waveform = waveform.unsqueeze(0)       # [1, C, T]
        return ({"waveform": waveform, "sample_rate": int(sr)},)

    @classmethod
    def IS_CHANGED(cls, url):
        return f"{url}"


# ---------------------------------------------------------------------------
# 2. Get Video Components (null tolerant)
# ---------------------------------------------------------------------------

class ZNGBGetVideoComponents:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "video": ("VIDEO",),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "FLOAT", "INT")
    RETURN_NAMES = ("images", "audio", "fps", "bit_depth")
    FUNCTION = "get_components"
    CATEGORY = "ZNGBNodes/video"
    DESCRIPTION = ("Like the official Get Video Components, but accepts a null video input. "
                   "When the video is null, images/audio/bit_depth are null but fps falls back "
                   "to 30.0 so simple downstream math doesn't crash.")

    def get_components(self, video=None):
        if video is None:
            # fps defaults to 30.0 so later frame-count / duration math keeps working
            # even when the video url was empty; the rest stay null.
            return (None, None, 30.0, None)

        comps = video.get_components()
        fps = float(comps.frame_rate)
        return (comps.images, comps.audio, fps, 8)


# ---------------------------------------------------------------------------
# 3. Image Batch Multi (null tolerant)
# ---------------------------------------------------------------------------

class ZNGBImageBatchMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 1000, "step": 1}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "combine"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Creates an image batch from multiple images. Any input may be null and is "
                   "skipped. If every input is null, the output is null.")

    def combine(self, inputcount, **kwargs):
        images = []
        for c in range(inputcount):
            img = kwargs.get(f"image_{c + 1}")
            if img is not None:
                images.append(img)

        if not images:
            return (None,)

        first = images[0]
        h, w = first.shape[1], first.shape[2]
        max_ch = max(img.shape[-1] for img in images)

        out_list = []
        for img in images:
            if img.shape[1:3] != (h, w):
                img = common_upscale(img.movedim(-1, 1), w, h, "bilinear", "center").movedim(1, -1)
            if img.shape[-1] < max_ch:
                img = torch.nn.functional.pad(img, (0, max_ch - img.shape[-1]), mode="constant", value=1.0)
            out_list.append(img)

        out = torch.cat(out_list, dim=0)
        return (out.cpu(),)


# ---------------------------------------------------------------------------
# 4. Audio Concat Multi (null tolerant)
# ---------------------------------------------------------------------------

class ZNGBAudioConcatMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 1000, "step": 1}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 384000, "step": 1,
                                        "tooltip": "Target sample rate. Every input is resampled to "
                                                   "this rate before concatenation (this keeps each "
                                                   "clip's duration unchanged)."}),
            },
            "optional": {
                "audio_1": ("AUDIO",),
                "audio_2": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concat"
    CATEGORY = "ZNGBNodes/audio"
    DESCRIPTION = ("Concatenates multiple audios one after another (in input order). Inputs with "
                   "different sample rates are resampled to the target sample_rate (durations are "
                   "preserved). Any input may be null and is skipped. If every input is null, the "
                   "output is null.")

    def concat(self, inputcount, sample_rate, **kwargs):
        target_sr = int(sample_rate)
        waveforms = []

        for c in range(inputcount):
            audio = kwargs.get(f"audio_{c + 1}")
            if audio is None:
                continue
            wf = audio["waveform"]
            sr = int(audio["sample_rate"])
            if sr != target_sr:
                # Resample to the target rate; this changes the sample count but keeps duration.
                wf = torchaudio.functional.resample(wf, sr, target_sr)
            waveforms.append(wf)

        if not waveforms:
            return (None,)

        # Pad channel counts so all waveforms can be concatenated along time (dim=2).
        max_ch = max(wf.shape[1] for wf in waveforms)
        padded = []
        for wf in waveforms:
            if wf.shape[1] < max_ch:
                pad = torch.zeros(wf.shape[0], max_ch - wf.shape[1], wf.shape[2],
                                  dtype=wf.dtype, device=wf.device)
                wf = torch.cat((wf, pad), dim=1)
            padded.append(wf)

        combined = torch.cat(padded, dim=2)
        return ({"waveform": combined, "sample_rate": target_sr},)


# ---------------------------------------------------------------------------
# 5. Get Image Range From Batch (null tolerant)
# ---------------------------------------------------------------------------

class ZNGBGetImageRangeFromBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_index": ("INT", {"default": 0, "min": -1, "max": 4096, "step": 1}),
                "num_frames": ("INT", {"default": 1, "min": 1, "max": 4096, "step": 1}),
            },
            "optional": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "get_range"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Returns a range of frames from an image batch (like KJNodes' Get Image Range, "
                   "without mask). If images is null, the output is null. start_index of -1 means "
                   "the last num_frames frames.")

    def get_range(self, start_index, num_frames, images=None):
        if images is None:
            return (None,)

        if start_index == -1:
            start_index = max(0, len(images) - num_frames)
        if start_index < 0 or start_index >= len(images):
            raise ValueError("Start index is out of range")
        end_index = min(start_index + num_frames, len(images))
        return (images[start_index:end_index],)


# ---------------------------------------------------------------------------
# 6. Resize Image (null tolerant, no mask)
# ---------------------------------------------------------------------------

class ZNGBResizeImage:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "crop"], {"default": "stretch"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color used for padding (R, G, B, 0-255)."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 1, "min": 0, "max": 512, "step": 1}),
            },
            "optional": {
                "image": ("IMAGE",),
                "device": (["cpu", "gpu"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    FUNCTION = "resize"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Resizes an image, similar to KJNodes' Resize Image v2 (without mask). "
                   "If image is null, the image/width/height outputs are all null.")

    def resize(self, width, height, upscale_method, keep_proportion, pad_color,
               crop_position, divisible_by, image=None, device="cpu"):
        if image is None:
            return (None, None, None)

        out = _resize_image_tensor(image, width, height, upscale_method, keep_proportion,
                                   pad_color, crop_position, divisible_by, device)
        return (out, out.shape[2], out.shape[1])


# ---------------------------------------------------------------------------
# 6b. Image Padding
# ---------------------------------------------------------------------------

class ZNGBImagePadding:
    @classmethod
    def INPUT_TYPES(cls):
        padding = {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1}
        color = {"default": 255, "min": 0, "max": 255, "step": 1}
        return {
            "required": {
                "image": ("IMAGE",),
                "padding_left": ("INT", padding),
                "padding_right": ("INT", padding),
                "padding_top": ("INT", padding),
                "padding_bottom": ("INT", padding),
                "red": ("INT", color),
                "green": ("INT", color),
                "blue": ("INT", color),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "pad"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = "Pads an image and returns a mask where the added area is 1."

    def pad(self, image, padding_left, padding_right, padding_top,
            padding_bottom, red, green, blue):
        batch, height, width, channels = image.shape
        output_height = height + padding_top + padding_bottom
        output_width = width + padding_left + padding_right
        output = torch.empty(
            (batch, output_height, output_width, channels),
            dtype=image.dtype,
            device=image.device,
        )
        color = (red / 255.0, green / 255.0, blue / 255.0)
        for channel in range(channels):
            output[..., channel] = color[channel] if channel < 3 else 1.0
        output[
            :, padding_top:padding_top + height,
            padding_left:padding_left + width, :,
        ] = image

        mask = torch.ones(
            (batch, output_height, output_width),
            dtype=torch.float32,
            device=image.device,
        )
        mask[
            :, padding_top:padding_top + height,
            padding_left:padding_left + width,
        ] = 0.0
        return (output, mask)


# ---------------------------------------------------------------------------
# 7. Audio Crop (null tolerant)
# ---------------------------------------------------------------------------

class ZNGBAudioCrop:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                                         "tooltip": "Start position in seconds."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01,
                                       "tooltip": "Length to keep in seconds (0 = until the end)."}),
            },
            "optional": {
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "crop"
    CATEGORY = "ZNGBNodes/audio"
    DESCRIPTION = ("Trims an audio clip by start time and duration (in seconds). "
                   "If audio is null, the output is null.")

    def crop(self, start_time, duration, audio=None):
        if audio is None:
            return (None,)

        waveform = audio["waveform"]
        sample_rate = audio["sample_rate"]
        total = waveform.shape[-1]

        start = max(0, min(int(start_time * sample_rate), total))
        if duration and duration > 0:
            end = min(total, start + int(duration * sample_rate))
        else:
            end = total

        cropped = waveform[..., start:end]
        return ({"waveform": cropped, "sample_rate": sample_rate},)


# ---------------------------------------------------------------------------
# 7b. Audio Overlay Multi (overlay multiple audios onto a source video timeline)
# ---------------------------------------------------------------------------

class ZNGBAudioOverlayMulti:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "source_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001,
                                           "tooltip": "Crop start of the source video, in seconds."}),
                "source_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001,
                                         "tooltip": "Crop end of the source video, in seconds "
                                                    "(<= start means until the end). Defines the "
                                                    "output length when the video has no audio."}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 384000, "step": 1,
                                        "tooltip": "Target sample rate for the mixed output."}),
            },
            "optional": {
                "SourceVideo": ("VIDEO",),
                "audio_1": ("AUDIO",),
                "audio_1_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                "audio_1_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "audio_2": ("AUDIO",),
                "audio_2_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                "audio_2_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "overlay"
    CATEGORY = "ZNGBNodes/audio"
    DESCRIPTION = ("Overlays multiple audios onto the source video's timeline. The output length is "
                   "the cropped source video duration (source_start..source_end). Base track = the "
                   "video's own audio (cropped) if present, else silence of the same length. Each "
                   "audio_i is mixed in at audio_i_start with audio_i_volume. If SourceVideo is null "
                   "the overlays are mixed alone and the length = max(start + audio length); if all "
                   "audios are also null the output is null. Peaks above 1.0 are normalized.")

    def overlay(self, inputcount, source_start, source_end, sample_rate,
                SourceVideo=None, **kwargs):
        target_sr = int(sample_rate)

        # No source video: mix the overlays alone; length = max(start + audio length).
        if SourceVideo is None:
            clips = []
            max_samples = 0
            for c in range(inputcount):
                audio = kwargs.get(f"audio_{c + 1}")
                if audio is None:
                    continue
                vol = float(kwargs.get(f"audio_{c + 1}_volume", 1.0))
                off = float(kwargs.get(f"audio_{c + 1}_start", 0.0))
                wf = audio["waveform"]
                in_sr = int(audio["sample_rate"])
                if in_sr != target_sr:
                    wf = torchaudio.functional.resample(wf, in_sr, target_sr)
                offset = max(0, int(round(off * target_sr)))
                clips.append((wf, offset, vol))
                max_samples = max(max_samples, offset + wf.shape[-1])
            if not clips:
                return (None,)
            base = torch.zeros((1, 2, max_samples), dtype=torch.float32)
            for wf, offset, vol in clips:
                base = self._mix_into(base, wf, offset, vol)
            peak = base.abs().max()
            if peak > 1.0:
                base = base / peak
            return ({"waveform": base, "sample_rate": target_sr},)

        comps = SourceVideo.get_components()
        fps = float(comps.frame_rate)
        total_frames = comps.images.shape[0]
        video_seconds = total_frames / fps if fps > 0 else 0.0

        start = max(0.0, source_start)
        end = source_end if source_end and source_end > start else video_seconds
        clip_seconds = max(0.0, end - start)
        total = max(1, int(round(clip_seconds * target_sr)))

        # Base track: source audio (cropped) if present, else silence.
        base = torch.zeros((1, 2, total), dtype=torch.float32)
        src_audio = comps.audio
        if src_audio is not None:
            wf = src_audio["waveform"]
            in_sr = int(src_audio["sample_rate"])
            a0 = max(0, int(round(start * in_sr)))
            a1 = min(wf.shape[-1], int(round(end * in_sr)))
            seg = wf[..., a0:a1]
            if in_sr != target_sr:
                seg = torchaudio.functional.resample(seg, in_sr, target_sr)
            base = self._mix_into(base, seg, 0, 1.0)

        for c in range(inputcount):
            audio = kwargs.get(f"audio_{c + 1}")
            if audio is None:
                continue
            vol = float(kwargs.get(f"audio_{c + 1}_volume", 1.0))
            off = float(kwargs.get(f"audio_{c + 1}_start", 0.0))
            wf = audio["waveform"]
            in_sr = int(audio["sample_rate"])
            if in_sr != target_sr:
                wf = torchaudio.functional.resample(wf, in_sr, target_sr)
            base = self._mix_into(base, wf, int(round(off * target_sr)), vol)

        peak = base.abs().max()
        if peak > 1.0:
            base = base / peak
        return ({"waveform": base, "sample_rate": target_sr},)

    @staticmethod
    def _mix_into(base, seg, offset, volume):
        # base: [1, 2, T]; seg: [B, C, t] -> mix to stereo at offset.
        s = seg[0] if seg.dim() == 3 else seg
        if s.shape[0] == 1:
            s = s.expand(2, -1)
        elif s.shape[0] > 2:
            s = s[:2]
        elif s.shape[0] == 0:
            return base
        t = s.shape[-1]
        total = base.shape[-1]
        if offset >= total:
            return base
        end = min(total, offset + t)
        base[0, :, offset:end] += s[:, :end - offset] * volume
        return base


# ---------------------------------------------------------------------------
# 8. Video Clip (cut + resize images and align audio, null tolerant)
# ---------------------------------------------------------------------------

class ZNGBVideoClip:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_fps": ("FLOAT", {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                         "tooltip": "Frame rate of the INPUT images (the original "
                                                    "video fps, e.g. from get video components). Used "
                                                    "to map start/end seconds to source frame indices "
                                                    "and to measure the real duration of the cut."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                  "tooltip": "OUTPUT/composite frame rate. The selected frames are "
                                             "retimed (duplicated/dropped) to this fps so the frame "
                                             "count matches the final video, and the audio is aligned "
                                             "to that output duration to keep lip-sync."}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001,
                                    "tooltip": "Clip start in seconds (millisecond precision)."}),
                "end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001,
                                  "tooltip": "Clip end in seconds (0 or <= start means until the end)."}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "crop"], {"default": "pad"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color used for padding (R, G, B, 0-255)."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 1, "min": 0, "max": 512, "step": 1}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 384000, "step": 1,
                                        "tooltip": "Target audio sample rate for the output. Use the "
                                                   "same value as audio concat multi for later merging."}),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "device": (["cpu", "gpu"],),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "clip"
    CATEGORY = "ZNGBNodes/video"
    DESCRIPTION = (
        "Cuts a [start, end] segment from an image batch (located by source_fps), resizes it (like "
        "resize image), then retimes the frames from source_fps to the output fps so the frame count "
        "matches the composite video. The audio is aligned to that output duration so lip-sync stays "
        "correct even when the original video fps differs from the composite fps.\n"
        "- images and audio both null => both outputs null.\n"
        "- images present, audio null => images are output and audio is silence of the same length.\n"
        "- images null, audio present => black frames matching the audio duration + the audio."
    )

    @staticmethod
    def _fit_length(waveform, target_len):
        cur = waveform.shape[-1]
        if cur == target_len:
            return waveform
        if cur > target_len:
            return waveform[..., :target_len]
        pad = torch.zeros(waveform.shape[0], waveform.shape[1], target_len - cur,
                          dtype=waveform.dtype, device=waveform.device)
        return torch.cat((waveform, pad), dim=-1)

    def clip(self, source_fps, fps, start, end, width, height, upscale_method, keep_proportion,
             pad_color, crop_position, divisible_by, sample_rate,
             images=None, audio=None, device="cpu"):
        target_sr = int(sample_rate)
        out_fps = float(fps) if fps and fps > 0 else 30.0

        # Both empty: nothing to output.
        if images is None and audio is None:
            return (None, None)

        # No images but audio present: output black frames matching the audio's
        # full duration at the output fps, and pass the audio through.
        if images is None:
            wf = audio["waveform"]
            in_sr = int(audio["sample_rate"])
            total = wf.shape[-1]
            seg = wf
            clip_seconds = max(0.0, total / in_sr)
            out_frame_count = max(1, int(round(clip_seconds * out_fps)))
            out_seconds = out_frame_count / out_fps
            target_samples = int(round(out_seconds * target_sr))
            if in_sr != target_sr:
                seg = torchaudio.functional.resample(seg, in_sr, target_sr)
            seg = self._fit_length(seg, target_samples)
            black = torch.zeros((out_frame_count, height, width, 3), dtype=torch.float32)
            return (black, {"waveform": seg, "sample_rate": target_sr})

        src_fps = float(source_fps) if source_fps and source_fps > 0 else 30.0
        total_frames = images.shape[0]

        # 1. Locate the segment in the SOURCE frames using the original video fps.
        start_f = max(0, int(round(start * src_fps)))
        start_f = min(start_f, total_frames)
        if end and end > start:
            end_f = int(round(end * src_fps))
        else:
            end_f = total_frames
        end_f = min(max(end_f, start_f), total_frames)

        sel = images[start_f:end_f]
        actual_frames = sel.shape[0]
        if actual_frames == 0:
            return (None, None)

        # Real duration of the selected source frames (measured at the source fps).
        clip_seconds = actual_frames / src_fps

        out_images = _resize_image_tensor(sel, width, height, upscale_method, keep_proportion,
                                          pad_color, crop_position, divisible_by, device)

        # 2. Retime the frames from source_fps to the output fps so the frame count
        #    matches the composite timeline. This is what keeps lip-sync aligned when
        #    the original video fps (e.g. 24) differs from the composite fps (e.g. 30).
        out_frame_count = max(1, int(round(clip_seconds * out_fps)))
        if out_frame_count != actual_frames:
            idx = torch.linspace(0, actual_frames - 1, steps=out_frame_count).round().long()
            out_images = out_images[idx]

        # 3. Tie the audio length to the OUTPUT video duration (frames / out_fps) so the
        #    audio and the retimed frames span exactly the same time.
        out_seconds = out_frame_count / out_fps
        target_samples = int(round(out_seconds * target_sr))

        if audio is not None:
            wf = audio["waveform"]
            in_sr = int(audio["sample_rate"])
            a_start = max(0, int(round(start * in_sr)))
            a_len = int(round(clip_seconds * in_sr))
            a_end = min(wf.shape[-1], a_start + a_len)
            seg = wf[..., a_start:a_end]
            # Pad with silence if the source audio is shorter than the needed segment.
            if seg.shape[-1] < a_len:
                pad = torch.zeros(seg.shape[0], seg.shape[1], a_len - seg.shape[-1],
                                  dtype=seg.dtype, device=seg.device)
                seg = torch.cat((seg, pad), dim=-1)
            if in_sr != target_sr:
                seg = torchaudio.functional.resample(seg, in_sr, target_sr)
            # Force an exact sample count so frames and audio stay perfectly aligned.
            seg = self._fit_length(seg, target_samples)
            out_audio = {"waveform": seg, "sample_rate": target_sr}
        else:
            silent = torch.zeros((1, 1, target_samples), dtype=torch.float32)
            out_audio = {"waveform": silent, "sample_rate": target_sr}

        return (out_images, out_audio)


# ---------------------------------------------------------------------------
# 8b. Video Clip V2 (video clip + multi audio overlay, null tolerant)
# ---------------------------------------------------------------------------

class ZNGBVideoClipV2:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1,
                                       "tooltip": "Number of overlay audio tracks (audio_i + start "
                                                  "+ volume). Press 'Update inputs' to apply."}),
                "source_fps": ("FLOAT", {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                         "tooltip": "Frame rate of source_images (the original video "
                                                    "fps). Maps start/end seconds to source frames."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                  "tooltip": "Output/composite frame rate. Frames are retimed to this "
                                             "fps and audio is aligned to it to keep lip-sync."}),
                "source_video_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001,
                                                 "tooltip": "Clip start in seconds (millisecond precision)."}),
                "source_video_end": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001,
                                               "tooltip": "Clip end in seconds (0 or <= start = until the end)."}),
                "width": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 0, "max": MAX_RESOLUTION, "step": 1}),
                "upscale_method": (cls.upscale_methods,),
                "keep_proportion": (["stretch", "resize", "pad", "crop"], {"default": "pad"}),
                "pad_color": ("STRING", {"default": "0, 0, 0", "tooltip": "Color used for padding (R, G, B, 0-255)."}),
                "crop_position": (["center", "top", "bottom", "left", "right"], {"default": "center"}),
                "divisible_by": ("INT", {"default": 1, "min": 0, "max": 512, "step": 1}),
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 384000, "step": 1,
                                        "tooltip": "Target audio sample rate for the output."}),
            },
            "optional": {
                "source_images": ("IMAGE",),
                "source_audio": ("AUDIO",),
                "device": (["cpu", "gpu"],),
                "audio_1": ("AUDIO",),
                "audio_1_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                "audio_1_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "audio_2": ("AUDIO",),
                "audio_2_start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.001}),
                "audio_2_volume": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO")
    RETURN_NAMES = ("images", "audio")
    FUNCTION = "clip"
    CATEGORY = "ZNGBNodes/video"
    DESCRIPTION = (
        "Like video clip, but mixes extra background audios on top of the result. Cuts source_images "
        "[source_video_start, source_video_end] (located by source_fps), resizes and retimes the "
        "frames to the output fps, then aligns/mixes the audio to that duration so lip-sync stays "
        "correct. Each audio_i is overlaid at audio_i_start (output timeline) with audio_i_volume.\n"
        "- source_images & source_audio null, overlays present => base = longest overlay, others mixed "
        "in at their start; images = black frames matching the mixed audio length.\n"
        "- source_images present, source_audio null, overlays present => audio = overlays on a silent "
        "track matching the output video length.\n"
        "- source_images present, source_audio null, no overlays => the images plus silence of the "
        "same output duration.\n"
        "- everything null => both outputs null.\n"
        "- source_images & source_audio present => standard clip, overlays mixed in if any."
    )

    @staticmethod
    def _fit_length(waveform, target_len):
        cur = waveform.shape[-1]
        if cur == target_len:
            return waveform
        if cur > target_len:
            return waveform[..., :target_len]
        pad = torch.zeros(waveform.shape[0], waveform.shape[1], target_len - cur,
                          dtype=waveform.dtype, device=waveform.device)
        return torch.cat((waveform, pad), dim=-1)

    def clip(self, inputcount, source_fps, fps, source_video_start, source_video_end,
             width, height, upscale_method, keep_proportion, pad_color, crop_position,
             divisible_by, sample_rate, source_images=None, source_audio=None,
             device="cpu", **kwargs):
        target_sr = int(sample_rate)
        out_fps = float(fps) if fps and fps > 0 else 30.0
        src_fps = float(source_fps) if source_fps and source_fps > 0 else 30.0
        mix = ZNGBAudioOverlayMulti._mix_into

        # Collect overlay audios (skip the null ones).
        overlays = []
        for c in range(inputcount):
            a = kwargs.get(f"audio_{c + 1}")
            if a is None:
                continue
            vol = float(kwargs.get(f"audio_{c + 1}_volume", 1.0))
            off = float(kwargs.get(f"audio_{c + 1}_start", 0.0))
            overlays.append((a, off, vol))
        has_overlay = len(overlays) > 0

        # ---- Case A: no source video (source_images is null) ----
        if source_images is None:
            clips = []
            max_samples = 0
            # If a source audio still exists, use it as the base at offset 0.
            if source_audio is not None:
                wf = source_audio["waveform"]
                in_sr = int(source_audio["sample_rate"])
                if in_sr != target_sr:
                    wf = torchaudio.functional.resample(wf, in_sr, target_sr)
                clips.append((wf, 0, 1.0))
                max_samples = max(max_samples, wf.shape[-1])
            for a, off, vol in overlays:
                wf = a["waveform"]
                in_sr = int(a["sample_rate"])
                if in_sr != target_sr:
                    wf = torchaudio.functional.resample(wf, in_sr, target_sr)
                offset = max(0, int(round(off * target_sr)))
                clips.append((wf, offset, vol))
                max_samples = max(max_samples, offset + wf.shape[-1])
            if not clips:
                # Everything null => both outputs null.
                return (None, None)
            base = torch.zeros((1, 2, max_samples), dtype=torch.float32)
            for wf, offset, vol in clips:
                base = mix(base, wf, offset, vol)
            peak = base.abs().max()
            if peak > 1.0:
                base = base / peak
            audio_seconds = max_samples / target_sr if target_sr > 0 else 0.0
            out_frame_count = max(1, int(round(audio_seconds * out_fps)))
            out_seconds = out_frame_count / out_fps
            base = self._fit_length(base, int(round(out_seconds * target_sr)))
            black = torch.zeros((out_frame_count, height, width, 3), dtype=torch.float32)
            return (black, {"waveform": base, "sample_rate": target_sr})

        # ---- Case B: source images present ----
        total_frames = source_images.shape[0]
        start = max(0.0, source_video_start)
        if source_video_end and source_video_end > start:
            end = source_video_end
        else:
            end = total_frames / src_fps

        start_f = min(max(0, int(round(start * src_fps))), total_frames)
        end_f = min(max(int(round(end * src_fps)), start_f), total_frames)
        sel = source_images[start_f:end_f]
        actual_frames = sel.shape[0]
        if actual_frames == 0:
            return (None, None)

        clip_seconds = actual_frames / src_fps
        out_images = _resize_image_tensor(sel, width, height, upscale_method, keep_proportion,
                                          pad_color, crop_position, divisible_by, device)

        out_frame_count = max(1, int(round(clip_seconds * out_fps)))
        if out_frame_count != actual_frames:
            idx = torch.linspace(0, actual_frames - 1, steps=out_frame_count).round().long()
            out_images = out_images[idx]

        out_seconds = out_frame_count / out_fps
        target_samples = int(round(out_seconds * target_sr))

        has_source_audio = source_audio is not None
        # No audio at all: images plus silence of the same output duration.
        if not has_source_audio and not has_overlay:
            silent = torch.zeros((1, 1, target_samples), dtype=torch.float32)
            return (out_images, {"waveform": silent, "sample_rate": target_sr})

        # Base track aligned to the output video duration.
        base = torch.zeros((1, 2, target_samples), dtype=torch.float32)
        if has_source_audio:
            wf = source_audio["waveform"]
            in_sr = int(source_audio["sample_rate"])
            a_start = max(0, int(round(start * in_sr)))
            a_len = int(round(clip_seconds * in_sr))
            a_end = min(wf.shape[-1], a_start + a_len)
            seg = wf[..., a_start:a_end]
            if seg.shape[-1] < a_len:
                pad = torch.zeros(seg.shape[0], seg.shape[1], a_len - seg.shape[-1],
                                  dtype=seg.dtype, device=seg.device)
                seg = torch.cat((seg, pad), dim=-1)
            if in_sr != target_sr:
                seg = torchaudio.functional.resample(seg, in_sr, target_sr)
            seg = self._fit_length(seg, target_samples)
            base = mix(base, seg, 0, 1.0)

        # Overlay audios at their start (relative to the output clip timeline).
        for a, off, vol in overlays:
            wf = a["waveform"]
            in_sr = int(a["sample_rate"])
            if in_sr != target_sr:
                wf = torchaudio.functional.resample(wf, in_sr, target_sr)
            base = mix(base, wf, max(0, int(round(off * target_sr))), vol)

        peak = base.abs().max()
        if peak > 1.0:
            base = base / peak
        return (out_images, {"waveform": base, "sample_rate": target_sr})


# ---------------------------------------------------------------------------
# 9. Float (round to 3 decimals, for video clip start/end)
# ---------------------------------------------------------------------------

class ZNGBFloat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -1.0e9, "max": 1.0e9, "step": 0.0000000001,
                                    "tooltip": "A float value (up to 10 decimal places). The output "
                                               "is rounded to 3 decimals (millisecond precision) for "
                                               "use as video clip start/end."}),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("float",)
    FUNCTION = "get_value"
    CATEGORY = "ZNGBNodes/utils"
    DESCRIPTION = ("A float input node. Accepts up to 10 decimal places and outputs the value "
                   "rounded to 3 decimals (millisecond precision), meant for the video clip "
                   "start/end inputs.")

    def get_value(self, value):
        return (round(float(value), 3),)


# ---------------------------------------------------------------------------
# 9b. Text
# ---------------------------------------------------------------------------

class ZNGBText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("string",)
    FUNCTION = "get_text"
    CATEGORY = "ZNGBNodes/utils"
    DESCRIPTION = "A multiline text input that returns the text unchanged."

    def get_text(self, text):
        return (text,)


# ---------------------------------------------------------------------------
# 9c. Text To Text List
# ---------------------------------------------------------------------------

class ZNGBTextToTextList:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "delimiter": ("STRING", {"default": "\\n"}),
                "strip_whitespace": ("BOOLEAN", {"default": False}),
                "remove_empty": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split_text"
    CATEGORY = "ZNGBNodes/utils"
    DESCRIPTION = "Split text into a STRING list with an escaped or literal delimiter."

    @staticmethod
    def _decode_delimiter(delimiter):
        escapes = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\"}
        decoded = []
        index = 0
        while index < len(delimiter):
            if delimiter[index] == "\\" and index + 1 < len(delimiter):
                escaped = escapes.get(delimiter[index + 1])
                if escaped is not None:
                    decoded.append(escaped)
                    index += 2
                    continue
            decoded.append(delimiter[index])
            index += 1
        return "".join(decoded)

    def split_text(self, text, delimiter, strip_whitespace, remove_empty):
        separator = self._decode_delimiter(delimiter)
        items = text.split(separator) if separator else [text]
        if strip_whitespace:
            items = [item.strip() for item in items]
        if remove_empty:
            items = [item for item in items if item != ""]
        return (items,)


# ---------------------------------------------------------------------------
# 10. Equirect 360 To Views (extract perspective views from a panorama)
# ---------------------------------------------------------------------------

class ZNGBEquirect360ToViews:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "num_views": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1,
                                      "tooltip": "Number of perspective views taken evenly around the "
                                                 "horizon (yaw). 4 = front/right/back/left."}),
                "fov": ("FLOAT", {"default": 90.0, "min": 1.0, "max": 179.0, "step": 0.5,
                                  "tooltip": "Horizontal field of view of each output view, in degrees."}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.5,
                                    "tooltip": "Vertical look angle: +up / -down, in degrees."}),
                "yaw_offset": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.5,
                                         "tooltip": "Rotation offset applied to the first view, in degrees."}),
                "width": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 1024, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            },
            "optional": {
                "device": (["cpu", "gpu"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("views",)
    FUNCTION = "extract"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Extracts perspective (rectilinear) views from an equirectangular 360 panorama. "
                   "Produces num_views images evenly spaced around the horizon as one image batch.")

    def extract(self, image, num_views, fov, pitch, yaw_offset, width, height, device="cpu"):
        if image is None:
            return (None,)

        dev = model_management.get_torch_device() if device == "gpu" else torch.device("cpu")
        pano = image.to(dev)  # [B, H, W, C]
        B = pano.shape[0]

        fov_h = math.radians(fov)
        f = 0.5 * width / math.tan(fov_h / 2.0)

        # Pixel grid centered at the principal point.
        ys, xs = torch.meshgrid(
            torch.arange(height, dtype=torch.float32, device=dev),
            torch.arange(width, dtype=torch.float32, device=dev),
            indexing="ij",
        )
        x = (xs - (width - 1) / 2.0)
        y = (ys - (height - 1) / 2.0)
        z = torch.full_like(x, f)
        dirs = torch.stack((x, y, z), dim=-1)
        dirs = dirs / dirs.norm(dim=-1, keepdim=True)  # [H, W, 3]

        pitch_r = math.radians(pitch)
        cp, sp = math.cos(pitch_r), math.sin(pitch_r)
        rot_pitch = torch.tensor([[1, 0, 0], [0, cp, -sp], [0, sp, cp]], dtype=torch.float32, device=dev)

        views = []
        for i in range(num_views):
            yaw_r = math.radians(yaw_offset + i * 360.0 / num_views)
            cy, sy = math.cos(yaw_r), math.sin(yaw_r)
            rot_yaw = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32, device=dev)
            rot = rot_yaw @ rot_pitch
            d = dirs @ rot.T  # [H, W, 3]

            dx, dy, dz = d[..., 0], d[..., 1], d[..., 2]
            lon = torch.atan2(dx, dz)            # -pi..pi
            lat = torch.asin(torch.clamp(dy, -1.0, 1.0))  # -pi/2..pi/2
            u = lon / math.pi                    # -1..1
            v = lat / (math.pi / 2.0)            # -1..1
            grid = torch.stack((u, v), dim=-1).unsqueeze(0).expand(B, -1, -1, -1)

            src = pano.movedim(-1, 1)            # [B, C, H, W]
            out = F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)
            views.append(out.movedim(1, -1))     # [B, H, W, C]

        result = torch.cat(views, dim=0) if len(views) > 1 else views[0]
        return (result.cpu(),)


# ---------------------------------------------------------------------------
# 11. Lens Distortion Correction (OpenCV Brown-Conrady camera model)
# ---------------------------------------------------------------------------

class ZNGBLensDistortionCorrection:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "source_horizontal_fov": ("FLOAT", {
                    "default": 90.0, "min": 1.0, "max": 179.0, "step": 0.5,
                    "tooltip": "Horizontal FOV used by Equirect360ToViews.",
                }),
                "k1": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001,
                    "tooltip": "Primary radial coefficient. Start here; barrel distortion usually needs a negative value.",
                }),
                "k2": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001,
                    "tooltip": "Secondary radial coefficient. Keep at 0 until k1 alone is insufficient.",
                }),
                "k3": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.001,
                    "tooltip": "Third radial coefficient for strong edge distortion.",
                }),
                "p1": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.0005,
                    "tooltip": "Vertical tangential coefficient. Normally leave at 0.",
                }),
                "p2": ("FLOAT", {
                    "default": 0.0, "min": -0.5, "max": 0.5, "step": 0.0005,
                    "tooltip": "Horizontal tangential coefficient. Normally leave at 0.",
                }),
                "center_x": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Normalized distortion center X. 0.5 is the image center.",
                }),
                "center_y": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 1.0, "step": 0.001,
                    "tooltip": "Normalized distortion center Y. 0.5 is the image center.",
                }),
                "zoom": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.01,
                    "tooltip": "Output zoom. Increase only to crop invalid borders after correction.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "correct"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = (
        "Corrects consistent radial and tangential distortion with OpenCV's Brown-Conrady camera "
        "model. Unlike changing FOV, this bends pixels non-linearly to straighten curved lines. "
        "Use one shared parameter set for every view from the same AI panorama."
    )

    def correct(self, images, source_horizontal_fov, k1, k2, k3, p1, p2,
                center_x, center_y, zoom):
        if images is None:
            return (None,)

        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RuntimeError("lens distortion correction requires opencv-python") from exc

        source = images.detach().cpu().float().numpy()
        _, height, width, _ = source.shape
        focal = 0.5 * width / math.tan(math.radians(source_horizontal_fov) / 2.0)
        camera = np.array([
            [focal, 0.0, center_x * (width - 1)],
            [0.0, focal, center_y * (height - 1)],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        output_camera = camera.copy()
        output_camera[0, 0] *= zoom
        output_camera[1, 1] *= zoom
        distortion = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera, distortion, None, output_camera, (width, height), cv2.CV_32FC1,
        )
        corrected = [
            cv2.remap(frame, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            for frame in source
        ]
        return (torch.from_numpy(np.stack(corrected)).clamp_(0.0, 1.0),)


# ---------------------------------------------------------------------------
# 14. Gaussian Splatting converter / compressor (ply <-> spz <-> ...)
# ---------------------------------------------------------------------------

# Target format -> output file extension.
_GS_FORMAT_EXT = {
    "spz": ".spz",
    "3dgs": ".ply",
    "compressed_ply": ".ply",
    "cc": ".ply",
    "ksplat": ".ksplat",
    "splat": ".splat",
    "sog": ".sog",
    "parquet": ".parquet",
}


def _resolve_gsconverter_cmd():
    """Return the base command list used to launch 3dgsconverter.

    Prefers the console script installed alongside the current Python so we run
    inside the exact same environment as ComfyUI. Falls back to a module call.
    Running it as a subprocess keeps Taichi/CUDA initialization out of the
    ComfyUI process.
    """
    exe_dir = os.path.dirname(sys.executable)
    candidates = []
    for sub in ("Scripts", "bin", ""):
        for name in ("3dgsconverter.exe", "3dgsconverter"):
            candidates.append(os.path.join(exe_dir, sub, name))
    for p in candidates:
        if os.path.isfile(p):
            return [p]

    found = shutil.which("3dgsconverter")
    if found:
        return [found]

    # Last resort: invoke the module's main() in the same interpreter.
    return [sys.executable, "-c", "from gsconverter.main import main; main()"]


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"


def _get_lama_inpaint_model(device):
    global _LAMA_INPAINT_MODEL, _LAMA_INPAINT_MODEL_DEVICE

    model_path = os.path.join(folder_paths.models_dir, "cv_fft_inpainting_lama")
    model_file = os.path.join(model_path, "pytorch_model.pt")
    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"LaMa model not found at {model_path!r}; pytorch_model.pt is required"
        )

    if _LAMA_INPAINT_MODEL is None:
        from modelscope.models.cv.image_inpainting.model import FFTInpainting

        _LAMA_INPAINT_MODEL = FFTInpainting(
            model_path, predict_only=True
        ).eval()
    if _LAMA_INPAINT_MODEL_DEVICE != str(device):
        _LAMA_INPAINT_MODEL.to(device=device, dtype=torch.float32)
        _LAMA_INPAINT_MODEL_DEVICE = str(device)
    return _LAMA_INPAINT_MODEL


class LamaInpainting_zngb:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "inpaint"
    CATEGORY = "ZNGBNodes/image"

    def inpaint(self, img, mask):
        if img.shape[0] != mask.shape[0] and mask.shape[0] != 1:
            raise ValueError(
                "mask batch size must be 1 or match the image batch size"
            )

        device = model_management.get_torch_device()
        model = _get_lama_inpaint_model(device)
        outputs = []

        for image_index, item in enumerate(img):
            source = item[:, :, :3].detach().cpu().float().clamp(0.0, 1.0)
            source_height, source_width = source.shape[:2]
            current_mask = mask[0 if mask.shape[0] == 1 else image_index]
            current_mask = current_mask.detach().cpu().float().unsqueeze(0).unsqueeze(0)
            if current_mask.shape[-2:] != (source_height, source_width):
                current_mask = F.interpolate(
                    current_mask,
                    size=(source_height, source_width),
                    mode="nearest",
                )
            current_mask = current_mask[0, 0] > 0

            pad_height = (-source_height) % 8
            pad_width = (-source_width) % 8
            image_array = source.permute(2, 0, 1).numpy()
            mask_array = current_mask.numpy().astype(np.float32)[None, ...]
            image_array = np.pad(
                image_array,
                ((0, 0), (0, pad_height), (0, pad_width)),
                mode="symmetric",
            )
            mask_array = np.pad(
                mask_array,
                ((0, 0), (0, pad_height), (0, pad_width)),
                mode="symmetric",
            )
            batch = {
                "image": torch.from_numpy(image_array).unsqueeze(0).to(device),
                "mask": torch.from_numpy(mask_array).unsqueeze(0).to(device),
            }

            with torch.inference_mode():
                result = model(batch)["inpainted"]
            result = result[0, :, :source_height, :source_width]
            outputs.append(
                result.permute(1, 2, 0).detach().float().cpu().clamp(0.0, 1.0).unsqueeze(0)
            )

        return (torch.cat(outputs, dim=0),)


class CropImageByBBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("imgs_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "crop"
    CATEGORY = "ZNGBNodes/image"

    def crop(self, bboxes, image):
        if not isinstance(bboxes, (list, tuple)):
            raise TypeError("bboxes must be a list of [x1, y1, x2, y2] boxes")

        crops = []
        for item in image:
            height, width = item.shape[:2]
            for box in bboxes:
                if isinstance(box, dict):
                    box = box.get("bbox_2d") or box.get("bbox")
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue

                x1, y1, x2, y2 = (float(value) for value in box[:4])
                left = max(0, min(width, int(math.floor(min(x1, x2)))))
                top = max(0, min(height, int(math.floor(min(y1, y2)))))
                right = max(0, min(width, int(math.ceil(max(x1, x2)))))
                bottom = max(0, min(height, int(math.ceil(max(y1, y2)))))
                if right <= left or bottom <= top:
                    continue

                crops.append(item[top:bottom, left:right, :].unsqueeze(0))

        if not crops:
            raise ValueError("No valid bounding boxes overlap the input image")
        return (crops,)


class CropImgByBBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bboxes": ("BOUNDING_BOX",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("imgs_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "crop"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Crops each official ComfyUI BBOX as a separate image without resizing. "
                   "BBoxes use per-frame lists of x/y/width/height dictionaries.")

    def crop(self, image, bboxes):
        if isinstance(bboxes, dict):
            bboxes = [[bboxes]]
        elif isinstance(bboxes, list) and bboxes and isinstance(bboxes[0], dict):
            bboxes = [bboxes]
        elif not isinstance(bboxes, list):
            raise TypeError("bboxes must be a dict, list of dicts, or per-frame list of dicts")

        crops = []
        for frame_index, item in enumerate(image):
            if not bboxes:
                break
            frame_bboxes = bboxes[min(frame_index, len(bboxes) - 1)]
            if not isinstance(frame_bboxes, list):
                raise TypeError("each frame's bboxes must be a list of dictionaries")

            height, width = item.shape[:2]
            for box in frame_bboxes:
                if not isinstance(box, dict):
                    raise TypeError("each bbox must be a dictionary with x, y, width, and height")
                try:
                    x = float(box["x"])
                    y = float(box["y"])
                    box_width = float(box["width"])
                    box_height = float(box["height"])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        "each bbox must contain numeric x, y, width, and height values"
                    ) from exc

                left = max(0, min(width, int(math.floor(x))))
                top = max(0, min(height, int(math.floor(y))))
                right = max(0, min(width, int(math.ceil(x + box_width))))
                bottom = max(0, min(height, int(math.ceil(y + box_height))))
                if right <= left or bottom <= top:
                    continue
                crops.append(item[top:bottom, left:right, :].unsqueeze(0))

        if not crops:
            raise ValueError("No valid bounding boxes overlap the input image")
        return (crops,)


class ZNGBMasksToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"masks": ("MASK",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "combine"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = "Combine every mask from all SAM3 batches into one mask using a pixel-wise union."

    def combine(self, masks):
        if masks is None:
            raise ValueError("masks cannot be null")
        if isinstance(masks, torch.Tensor):
            masks = [masks]

        batches = []
        for mask_batch in masks:
            if mask_batch is None:
                continue
            if not isinstance(mask_batch, torch.Tensor):
                raise TypeError("every masks input must be a torch tensor")
            if mask_batch.dim() == 2:
                mask_batch = mask_batch.unsqueeze(0)
            if mask_batch.dim() != 3:
                raise ValueError("masks must have shape [N, H, W] or [H, W]")
            if mask_batch.shape[0] > 0:
                batches.append(mask_batch)

        if not batches:
            raise ValueError("masks must contain at least one mask")
        height, width = batches[0].shape[-2:]
        if any(batch.shape[-2:] != (height, width) for batch in batches):
            raise ValueError("all masks must have the same height and width")

        combined = torch.cat(batches, dim=0).amax(dim=0, keepdim=True)
        return (combined.clamp(0.0, 1.0),)


class ZNGBMaskFillHole:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "invert_mask": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "fill"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = "Fill enclosed holes in each binary mask, with optional output inversion."

    def fill(self, mask, invert_mask):
        from scipy import ndimage

        if mask is None:
            raise ValueError("mask cannot be null")
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        if mask.dim() != 3:
            raise ValueError("mask must have shape [B, H, W] or [H, W]")
        if mask.shape[0] == 0:
            return (mask.to(dtype=torch.float32),)

        structure = ndimage.generate_binary_structure(2, 2)
        outputs = []
        for current_mask in mask:
            binary = current_mask.detach().cpu().numpy() > (127.0 / 255.0)
            filled = ndimage.binary_fill_holes(binary, structure=structure)
            if invert_mask:
                filled = ~filled
            outputs.append(torch.from_numpy(filled.astype(np.float32)))

        result = torch.stack(outputs).to(device=mask.device, dtype=torch.float32)
        return (result,)


class ImageAddMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "masks": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "add_masks"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Adds each SAM3 individual mask to the same source image as an alpha channel "
                   "and returns one RGBA image per mask.")

    def add_masks(self, image, masks):
        if image is None or len(image) == 0:
            raise ValueError("image must contain at least one image")
        if masks is None:
            raise ValueError("masks cannot be null")

        if masks.dim() == 2:
            masks = masks.unsqueeze(0)
        if masks.dim() != 3:
            raise ValueError("masks must have shape [N, H, W] or [H, W]")
        if masks.shape[0] == 0:
            raise ValueError("masks must contain at least one mask")

        source = image[0, :, :, :3]
        height, width = source.shape[:2]
        resized_masks = masks.to(device=source.device, dtype=source.dtype)
        if resized_masks.shape[-2:] != (height, width):
            resized_masks = F.interpolate(
                resized_masks.unsqueeze(1),
                size=(height, width),
                mode="nearest",
            ).squeeze(1)

        outputs = []
        for mask in resized_masks:
            rgba = torch.cat((source, mask.clamp(0.0, 1.0).unsqueeze(-1)), dim=-1)
            outputs.append(rgba.unsqueeze(0))
        return (outputs,)


class ZNGBCheckerboardToMasks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("masks",)
    FUNCTION = "segment"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Removes a baked-in light checkerboard background and returns one mask "
                   "for each disconnected foreground element. Uses the alpha channel when present.")

    @staticmethod
    def _checkerboard_colors(rgb):
        pixels = rgb.reshape(-1, 3)
        quantized = (pixels // 4) * 4
        colors, counts = np.unique(quantized, axis=0, return_counts=True)
        brightness = colors.mean(axis=1)
        neutrality = colors.max(axis=1) - colors.min(axis=1)
        candidates = np.where((brightness >= 180) & (neutrality <= 24))[0]
        if candidates.size < 2:
            raise ValueError("Could not detect two light checkerboard background colors")

        ordered = candidates[np.argsort(counts[candidates])[::-1]]
        first = colors[ordered[0]].astype(np.float32) + 1.5
        second = None
        for index in ordered[1:]:
            candidate = colors[index].astype(np.float32) + 1.5
            distance = np.linalg.norm(candidate - first)
            if 6.0 <= distance <= 80.0:
                second = candidate
                break
        if second is None:
            raise ValueError("Could not distinguish the two checkerboard background colors")
        return first, second

    @staticmethod
    def _foreground_from_checkerboard(rgb):
        color_a, color_b = ZNGBCheckerboardToMasks._checkerboard_colors(rgb)
        pixels = rgb.astype(np.float32)
        distance_a = np.linalg.norm(pixels - color_a, axis=2)
        distance_b = np.linalg.norm(pixels - color_b, axis=2)
        background_distance = np.minimum(distance_a, distance_b)

        # JPEG compression and resized checkerboards produce many colors around the
        # two nominal tile colors. Keep only candidates connected to the canvas edge
        # so similarly colored pixels enclosed by an object are not removed.
        background_like = (background_distance <= 30.0).astype(np.uint8)

        flood_source = np.pad(background_like, 1, mode="constant", constant_values=1)
        flood_mask = np.zeros((flood_source.shape[0] + 2, flood_source.shape[1] + 2), np.uint8)
        cv2.floodFill(flood_source, flood_mask, (0, 0), 2)
        exterior_background = flood_source[1:-1, 1:-1] == 2

        # Generated cutouts often bake a neutral drop shadow into the RGB image.
        # Grow the exterior only through low-saturation pixels, stopping at colored
        # artwork and dark outlines. This also consumes isolated checkerboard halos.
        saturation = pixels.max(axis=2) - pixels.min(axis=2)
        brightness = pixels.mean(axis=2)
        neutral_background = (
            (saturation <= 18.0)
            & (brightness >= 105.0)
            & (background_distance <= 34.0)
        ).astype(np.uint8)
        background_candidates = ((neutral_background > 0) | exterior_background).astype(np.uint8)
        _, candidate_labels = cv2.connectedComponents(background_candidates, connectivity=8)
        exterior_labels = np.unique(candidate_labels[exterior_background])
        expanded_background = np.isin(candidate_labels, exterior_labels) & (candidate_labels > 0)

        foreground = (~expanded_background).astype(np.uint8)
        return foreground

    @staticmethod
    def _foreground_for_frame(frame):
        if frame.shape[-1] >= 4 and np.any(frame[..., 3] < 0.999):
            alpha = np.clip(frame[..., 3], 0.0, 1.0).astype(np.float32)
            return (alpha > 0.01).astype(np.uint8), alpha

        rgb = np.clip(frame[..., :3] * 255.0, 0, 255).astype(np.uint8)
        return ZNGBCheckerboardToMasks._foreground_from_checkerboard(rgb), None

    @staticmethod
    def _component_masks(foreground, alpha=None):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
        min_area = max(16, int(foreground.size * 0.0002))
        components = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            component = (labels == label).astype(np.float32)
            if alpha is not None:
                component *= alpha
            else:
                component = cv2.GaussianBlur(component, (0, 0), sigmaX=0.65)
            components.append((area, np.clip(component, 0.0, 1.0)))
        components.sort(key=lambda item: item[0], reverse=True)
        return [component for _, component in components]

    def segment(self, image):
        if image is None or len(image) == 0:
            raise ValueError("image must contain at least one image")

        output_masks = []
        for frame in image.detach().cpu().numpy():
            foreground, alpha = self._foreground_for_frame(frame)
            output_masks.extend(self._component_masks(foreground, alpha))

        if not output_masks:
            raise ValueError("No disconnected foreground elements were found")
        return (torch.from_numpy(np.stack(output_masks)).to(dtype=torch.float32),)


class ZNGBCheckerboardToBBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "padding_ratio": ("FLOAT", {
                    "default": 0.08, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Expand each robust box by this fraction for SAM3 prompting.",
                }),
                "outlier_percent": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Ignore this percentage of component pixels at each outer edge.",
                }),
                "min_area_ratio": ("FLOAT", {
                    "default": 0.0002, "min": 0.0, "max": 0.1, "step": 0.0001,
                    "tooltip": "Discard components smaller than this fraction of the image area.",
                }),
            }
        }

    RETURN_TYPES = ("BOUNDING_BOX",)
    RETURN_NAMES = ("bboxes",)
    FUNCTION = "detect"
    CATEGORY = "ZNGBNodes/image"
    DESCRIPTION = ("Finds disconnected elements on a checkerboard background and returns "
                   "per-frame bounding boxes compatible with the official SAM3 Detect node.")

    @staticmethod
    def _robust_box(labels, label, width, height, outlier_percent, padding_ratio):
        ys, xs = np.where(labels == label)
        low = float(outlier_percent)
        high = 100.0 - low
        left = int(math.floor(np.percentile(xs, low)))
        top = int(math.floor(np.percentile(ys, low)))
        right = int(math.ceil(np.percentile(xs, high))) + 1
        bottom = int(math.ceil(np.percentile(ys, high))) + 1

        box_width = max(1, right - left)
        box_height = max(1, bottom - top)
        if padding_ratio > 0:
            pad_x = max(4, int(math.ceil(box_width * padding_ratio)))
            pad_y = max(4, int(math.ceil(box_height * padding_ratio)))
            left = max(0, left - pad_x)
            top = max(0, top - pad_y)
            right = min(width, right + pad_x)
            bottom = min(height, bottom + pad_y)

        return {
            "x": left,
            "y": top,
            "width": right - left,
            "height": bottom - top,
        }

    def detect(self, image, padding_ratio, outlier_percent, min_area_ratio):
        if image is None or len(image) == 0:
            raise ValueError("image must contain at least one image")

        per_frame_boxes = []
        for frame in image.detach().cpu().numpy():
            foreground, _ = ZNGBCheckerboardToMasks._foreground_for_frame(frame)
            height, width = foreground.shape
            count, labels, stats, _ = cv2.connectedComponentsWithStats(foreground, connectivity=8)
            min_area = max(16, int(foreground.size * min_area_ratio))
            components = []
            for label in range(1, count):
                area = int(stats[label, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                core_box = self._robust_box(
                    labels, label, width, height, outlier_percent, 0.0
                )
                box = self._robust_box(
                    labels, label, width, height, outlier_percent, padding_ratio
                )
                components.append((area, box, core_box))
            components.sort(key=lambda item: item[0], reverse=True)
            boxes = []
            core_boxes = []
            for _, box, core_box in components:
                right = core_box["x"] + core_box["width"]
                bottom = core_box["y"] + core_box["height"]
                contained = any(
                    core_box["x"] >= parent["x"]
                    and core_box["y"] >= parent["y"]
                    and right <= parent["x"] + parent["width"]
                    and bottom <= parent["y"] + parent["height"]
                    for parent in core_boxes
                )
                if not contained:
                    boxes.append(box)
                    core_boxes.append(core_box)
            per_frame_boxes.append(boxes)

        if not any(per_frame_boxes):
            raise ValueError("No disconnected foreground elements were found")
        return (per_frame_boxes,)


class ZNGBGaussianSplattingConverter:
    """Convert / losslessly compress a Gaussian Splatting model via 3dgsconverter.

    Loads a model file by path (.ply / .spz / .ksplat / .splat / .sog / .parquet)
    and writes it out in the chosen target format, returning the saved path. Great
    for shrinking a large 3DGS .ply (e.g. 381 MB) into a compact .spz.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "tooltip": "Path to the source model "
                                          "(.ply/.spz/.ksplat/.splat/.sog/.parquet)."}),
                "target_format": (list(_GS_FORMAT_EXT.keys()), {"default": "spz",
                                  "tooltip": "Output format. 'spz' gives the best lossless size "
                                             "reduction; 'compressed_ply' is a quantized PLY."}),
                "compression_level": ("INT", {"default": 9, "min": 0, "max": 9, "step": 1,
                                      "tooltip": "0-9. For SPZ this is the Gzip (v3) / ZSTD (v4) "
                                                 "effort and is lossless. 9 = smallest file."}),
            },
            "optional": {
                "spz_version": (["3", "4"], {"default": "3", "tooltip": "SPZ version. 3 = Gzip "
                                             "(widest support), 4 = ZSTD + native SH4."}),
                "force": ("BOOLEAN", {"default": True, "tooltip": "Overwrite the output if it exists."}),
                "rgb": ("BOOLEAN", {"default": False, "tooltip": "Add RGB values derived from SH "
                                    "(useful for CC/SOG/SPZ viewers)."}),
                "crop_sh": ("BOOLEAN", {"default": False, "tooltip": "Only write the SH coefficients "
                            "present in the source (disable canonical SH padding). Prevents a small "
                            "SH-0 model from ballooning when exporting to '3dgs'/'cc'."}),
                "extra_elements": ("BOOLEAN", {"default": False, "tooltip": "Preserve extra PLY "
                                   "elements (camera extrinsic/intrinsic) for 3dgs/cc formats."}),
                "sh_level": ("INT", {"default": -1, "min": -1, "max": 4, "step": 1,
                             "tooltip": "Target SH degree 0-4 (-1 = keep source). Lower = smaller."}),
                "min_opacity": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1,
                                "tooltip": "Drop splats with opacity below this (0 = keep all). "
                                           "Note: >0 makes the result lossy."}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output_path", "info")
    FUNCTION = "convert"
    OUTPUT_NODE = True
    CATEGORY = "ZNGBNodes/3d"
    DESCRIPTION = ("Convert or losslessly compress a 3D Gaussian Splatting model using "
                   "3dgsconverter. Input a model path, get the saved path back. Use target "
                   "'spz' with compression_level 9 to shrink a large .ply with no data loss.")

    @classmethod
    def IS_CHANGED(cls, input_path, target_format, compression_level,
                   spz_version="3", force=True, rgb=False, crop_sh=False,
                   extra_elements=False, sh_level=-1, min_opacity=0):
        # Always re-run: the output name is timestamped, so every run is unique.
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    def convert(self, input_path, target_format, compression_level,
                spz_version="3", force=True, rgb=False, crop_sh=False,
                extra_elements=False, sh_level=-1, min_opacity=0):
        # Null tolerant: an empty path just passes through so a pipeline can continue.
        if not input_path or not str(input_path).strip():
            return ("", "no input path provided")

        input_path = str(input_path).strip().strip('"')
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input model not found: {input_path!r}")

        # Always save under <ComfyUI output>/3dgsconver/ with a timestamped name.
        ext = _GS_FORMAT_EXT[target_format]
        out_dir = os.path.join(folder_paths.get_output_directory(), "3dgsconver")
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(out_dir, f"3dgsconver_{stamp}{ext}")

        cmd = _resolve_gsconverter_cmd() + [
            "-i", input_path,
            "-f", target_format,
            "-o", output_path,
        ]
        if force:
            cmd.append("--force")
        if compression_level is not None and compression_level >= 0:
            cmd += ["--compression_level", str(compression_level)]
        if target_format == "spz":
            cmd += ["--spz_version", str(spz_version)]
        if rgb:
            cmd.append("--rgb")
        if crop_sh:
            cmd.append("--crop_sh")
        if extra_elements:
            cmd.append("--extra_elements")
        if sh_level is not None and sh_level >= 0:
            cmd += ["--sh_level", str(sh_level)]
        if min_opacity and min_opacity > 0:
            cmd += ["--min_opacity", str(min_opacity)]

        print(f"[ZNGB] 3dgsconverter: {' '.join(cmd)}")
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        if proc.returncode != 0:
            raise RuntimeError(
                f"3dgsconverter failed (exit {proc.returncode}).\n"
                f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
            )
        if not os.path.isfile(output_path):
            raise RuntimeError(
                f"3dgsconverter reported success but the output was not found: {output_path!r}"
            )

        in_size = os.path.getsize(input_path)
        out_size = os.path.getsize(output_path)
        ratio = (in_size / out_size) if out_size else float("inf")
        saved_pct = (1.0 - out_size / in_size) * 100.0 if in_size else 0.0
        info = (
            f"{os.path.basename(input_path)} ({_human_size(in_size)}) -> "
            f"{os.path.basename(output_path)} ({_human_size(out_size)})  "
            f"ratio {ratio:.2f}x, saved {saved_pct:.1f}%\n{output_path}"
        )
        print(f"[ZNGB] {info}")
        return (output_path, info)


NODE_CLASS_MAPPINGS = {
    "LamaInpainting_zngb": LamaInpainting_zngb,
    "CropImageByBBoxes_zngb": CropImageByBBoxes,
    "CropImgByBBoxes_zngb": CropImgByBBoxes,
    "ZNGB_MasksToMask": ZNGBMasksToMask,
    "ZNGB_MaskFillHole": ZNGBMaskFillHole,
    "ImageAddMasks_zngb": ImageAddMasks,
    "ZNGB_CheckerboardToMasks": ZNGBCheckerboardToMasks,
    "ZNGB_CheckerboardToBBoxes": ZNGBCheckerboardToBBoxes,
    "ZNGB_LoadVideoFromUrl": ZNGBLoadVideoFromUrl,
    "ZNGB_LoadAudioFromUrl": ZNGBLoadAudioFromUrl,
    "ZNGB_GetVideoComponents": ZNGBGetVideoComponents,
    "ZNGB_ImageBatchMulti": ZNGBImageBatchMulti,
    "ZNGB_AudioConcatMulti": ZNGBAudioConcatMulti,
    "ZNGB_AudioOverlayMulti": ZNGBAudioOverlayMulti,
    "ZNGB_GetImageRangeFromBatch": ZNGBGetImageRangeFromBatch,
    "ZNGB_ResizeImage": ZNGBResizeImage,
    "ZNGB_ImagePadding": ZNGBImagePadding,
    "ZNGB_AudioCrop": ZNGBAudioCrop,
    "ZNGB_VideoClip": ZNGBVideoClip,
    "ZNGB_VideoClipV2": ZNGBVideoClipV2,
    "ZNGB_Float": ZNGBFloat,
    "ZNGB_Text": ZNGBText,
    "ZNGB_TextToTextList": ZNGBTextToTextList,
    "ZNGB_Equirect360ToViews": ZNGBEquirect360ToViews,
    "ZNGB_LensDistortionCorrection": ZNGBLensDistortionCorrection,
    "ZNGB_GaussianSplattingConverter": ZNGBGaussianSplattingConverter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LamaInpainting_zngb": "LamaInpainting_zngb",
    "CropImageByBBoxes_zngb": "crop image by bboxes",
    "CropImgByBBoxes_zngb": "crop img by bboxes",
    "ZNGB_MasksToMask": "masks to mask",
    "ZNGB_MaskFillHole": "Mask Fill Hole",
    "ImageAddMasks_zngb": "Image add Masks",
    "ZNGB_CheckerboardToMasks": "checkerboard to element masks",
    "ZNGB_CheckerboardToBBoxes": "checkerboard to element bboxes (SAM3)",
    "ZNGB_LoadVideoFromUrl": "load video from url",
    "ZNGB_LoadAudioFromUrl": "load audio from url",
    "ZNGB_GetVideoComponents": "get video components",
    "ZNGB_ImageBatchMulti": "image batch multi",
    "ZNGB_AudioConcatMulti": "audio concat multi",
    "ZNGB_AudioOverlayMulti": "audio overlay multi",
    "ZNGB_GetImageRangeFromBatch": "get image range from batch",
    "ZNGB_ResizeImage": "resize image",
    "ZNGB_ImagePadding": "image padding",
    "ZNGB_AudioCrop": "audio crop",
    "ZNGB_VideoClip": "video clip",
    "ZNGB_VideoClipV2": "video clip V2",
    "ZNGB_Float": "float",
    "ZNGB_Text": "text",
    "ZNGB_TextToTextList": "text2textlist ZNGB",
    "ZNGB_Equirect360ToViews": "Equirect360ToViews",
    "ZNGB_LensDistortionCorrection": "lens distortion correction (OpenCV)",
    "ZNGB_GaussianSplattingConverter": "gaussian splatting converter (ply/spz)",
}
