"""ZNGB custom nodes.

A small set of null-tolerant nodes. The goal is that when a video URL is empty
(None), downstream nodes (get video components / image batch multi /
audio concat multi) can still run without raising, outputting null values so the
rest of a video pipeline can continue and reach a final video combine.
"""

from __future__ import annotations

import os
import math
import shutil
import hashlib
import urllib.parse
import urllib.request

import torch
import torch.nn.functional as F
import torchaudio

import folder_paths
import comfy.model_management as model_management
from comfy.utils import common_upscale
from comfy_api.latest import InputImpl, Types


MAX_RESOLUTION = 16384


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


NODE_CLASS_MAPPINGS = {
    "ZNGB_LoadVideoFromUrl": ZNGBLoadVideoFromUrl,
    "ZNGB_LoadAudioFromUrl": ZNGBLoadAudioFromUrl,
    "ZNGB_GetVideoComponents": ZNGBGetVideoComponents,
    "ZNGB_ImageBatchMulti": ZNGBImageBatchMulti,
    "ZNGB_AudioConcatMulti": ZNGBAudioConcatMulti,
    "ZNGB_AudioOverlayMulti": ZNGBAudioOverlayMulti,
    "ZNGB_GetImageRangeFromBatch": ZNGBGetImageRangeFromBatch,
    "ZNGB_ResizeImage": ZNGBResizeImage,
    "ZNGB_AudioCrop": ZNGBAudioCrop,
    "ZNGB_VideoClip": ZNGBVideoClip,
    "ZNGB_Float": ZNGBFloat,
    "ZNGB_Equirect360ToViews": ZNGBEquirect360ToViews,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZNGB_LoadVideoFromUrl": "load video from url",
    "ZNGB_LoadAudioFromUrl": "load audio from url",
    "ZNGB_GetVideoComponents": "get video components",
    "ZNGB_ImageBatchMulti": "image batch multi",
    "ZNGB_AudioConcatMulti": "audio concat multi",
    "ZNGB_AudioOverlayMulti": "audio overlay multi",
    "ZNGB_GetImageRangeFromBatch": "get image range from batch",
    "ZNGB_ResizeImage": "resize image",
    "ZNGB_AudioCrop": "audio crop",
    "ZNGB_VideoClip": "video clip",
    "ZNGB_Float": "float",
    "ZNGB_Equirect360ToViews": "Equirect360ToViews",
}
