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

import folder_paths
from comfy.utils import common_upscale
from comfy_api.latest import InputImpl, Types


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
                   "When the video is null, all four outputs are null.")

    def get_components(self, video=None):
        if video is None:
            return (None, None, None, None)

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
    DESCRIPTION = ("Concatenates multiple audios one after another (in input order). Any input "
                   "may be null and is skipped. If every input is null, the output is null.")

    def concat(self, inputcount, **kwargs):
        waveforms = []
        sample_rate = None

        for c in range(inputcount):
            audio = kwargs.get(f"audio_{c + 1}")
            if audio is None:
                continue
            wf = audio["waveform"]
            sr = audio["sample_rate"]
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                raise ValueError(
                    f"Sample rate mismatch: audio_{c + 1} is {sr} Hz but expected {sample_rate} Hz."
                )
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
        return ({"waveform": combined, "sample_rate": sample_rate},)


NODE_CLASS_MAPPINGS = {
    "ZNGB_LoadVideoFromUrl": ZNGBLoadVideoFromUrl,
    "ZNGB_GetVideoComponents": ZNGBGetVideoComponents,
    "ZNGB_ImageBatchMulti": ZNGBImageBatchMulti,
    "ZNGB_AudioConcatMulti": ZNGBAudioConcatMulti,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZNGB_LoadVideoFromUrl": "load video from url",
    "ZNGB_GetVideoComponents": "get video components",
    "ZNGB_ImageBatchMulti": "image batch multi",
    "ZNGB_AudioConcatMulti": "audio concat multi",
}
