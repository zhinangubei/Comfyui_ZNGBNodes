"""ZNGB custom nodes.

A small set of null-tolerant nodes. The goal is that when a video URL is empty
(None), downstream nodes (get video components / image batch multi /
audio concat multi) can still run without raising, outputting null values so the
rest of a video pipeline can continue and reach a final video combine.
"""

from __future__ import annotations

import os
import shutil
import hashlib
import urllib.parse
import urllib.request

import torch
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
                "sample_rate": ("INT", {"default": 44100, "min": 1, "max": 384000, "step": 1,
                                        "tooltip": "Sample rate used only for the silent audio "
                                                   "generated when audio is null and duration > 0."}),
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
                   "If audio is null: when duration > 0 it outputs silent audio of that length, "
                   "otherwise the output stays null.")

    def crop(self, start_time, duration, sample_rate, audio=None):
        if audio is None:
            # No input audio: only synthesize silence when a positive duration is requested.
            if duration and duration > 0:
                num_samples = int(duration * sample_rate)
                silent = torch.zeros((1, 1, num_samples), dtype=torch.float32)
                return ({"waveform": silent, "sample_rate": int(sample_rate)},)
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
# 8. Video Clip (cut + resize images and align audio, null tolerant)
# ---------------------------------------------------------------------------

class ZNGBVideoClip:
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "fps": ("FLOAT", {"default": 30.0, "min": 0.01, "max": 1000.0, "step": 0.01,
                                  "tooltip": "Frame rate of the input images. Used to map start/end "
                                             "seconds to frame indices and to align the audio length "
                                             "to the number of output frames."}),
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
        "Cuts a [start, end] segment from an image batch and resizes it (like resize image), and "
        "produces matching audio. The audio length is tied to the number of output frames "
        "(frames / fps) so lip-sync stays aligned for later concatenation.\n"
        "- images and audio both null => both outputs null.\n"
        "- images present, audio null => images are output and audio is silence of the same length.\n"
        "- images null => both outputs null."
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

    def clip(self, fps, start, end, width, height, upscale_method, keep_proportion,
             pad_color, crop_position, divisible_by, sample_rate,
             images=None, audio=None, device="cpu"):
        target_sr = int(sample_rate)

        # The clip is driven by images; without frames there is no segment to output.
        if images is None:
            return (None, None)

        fps = float(fps) if fps and fps > 0 else 30.0
        total_frames = images.shape[0]

        start_f = max(0, int(round(start * fps)))
        start_f = min(start_f, total_frames)
        if end and end > start:
            end_f = int(round(end * fps))
        else:
            end_f = total_frames
        end_f = min(max(end_f, start_f), total_frames)

        sel = images[start_f:end_f]
        actual_frames = sel.shape[0]
        if actual_frames == 0:
            return (None, None)

        out_images = _resize_image_tensor(sel, width, height, upscale_method, keep_proportion,
                                          pad_color, crop_position, divisible_by, device)

        # Tie audio length to the output frame count to keep lip-sync aligned.
        clip_seconds = actual_frames / fps
        target_samples = int(round(clip_seconds * target_sr))

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


NODE_CLASS_MAPPINGS = {
    "ZNGB_LoadVideoFromUrl": ZNGBLoadVideoFromUrl,
    "ZNGB_GetVideoComponents": ZNGBGetVideoComponents,
    "ZNGB_ImageBatchMulti": ZNGBImageBatchMulti,
    "ZNGB_AudioConcatMulti": ZNGBAudioConcatMulti,
    "ZNGB_GetImageRangeFromBatch": ZNGBGetImageRangeFromBatch,
    "ZNGB_ResizeImage": ZNGBResizeImage,
    "ZNGB_AudioCrop": ZNGBAudioCrop,
    "ZNGB_VideoClip": ZNGBVideoClip,
    "ZNGB_Float": ZNGBFloat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZNGB_LoadVideoFromUrl": "load video from url",
    "ZNGB_GetVideoComponents": "get video components",
    "ZNGB_ImageBatchMulti": "image batch multi",
    "ZNGB_AudioConcatMulti": "audio concat multi",
    "ZNGB_GetImageRangeFromBatch": "get image range from batch",
    "ZNGB_ResizeImage": "resize image",
    "ZNGB_AudioCrop": "audio crop",
    "ZNGB_VideoClip": "video clip",
    "ZNGB_Float": "float",
}
