"""Alibaba Cloud Model Studio Qwen API nodes."""

from __future__ import annotations

import base64
import io
import os

import numpy as np
from PIL import Image


DEFAULT_BASE_URL = (
    "https://llm-s4he8vsvh2yhvz1m.cn-beijing.maas.aliyuncs.com/compatible-mode/v1"
)
QWEN_CONFIG_TYPE = "QWEN_LLM_CONFIG_ZNGB"
DEFAULT_THINKING_BUDGET = 1024
QWEN_MODELS = [
    "qwen3.8-max",
    "qwen3.7-plus",
    "qwen3.7-flash",
    "custom",
]


def _image_to_data_url(image) -> str:
    array = image.detach().cpu().float().clamp(0.0, 1.0).numpy()
    array = np.rint(array * 255.0).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    elif array.shape[-1] not in (3, 4):
        array = array[..., :3]

    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _build_user_content(prompt: str, images) -> list[dict]:
    content = [{"type": "text", "text": prompt}]
    for image_batch in images:
        if image_batch is None:
            continue
        for image in image_batch:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_to_data_url(image)},
                }
            )
    return content


class ZNGBQwenLLMConfig:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": DEFAULT_BASE_URL}),
                "api_key": ("STRING", {"default": "", "password": True}),
            }
        }

    RETURN_TYPES = (QWEN_CONFIG_TYPE,)
    RETURN_NAMES = ("api_config",)
    FUNCTION = "build_config"
    CATEGORY = "ZNGBNodes/LLM"
    DESCRIPTION = (
        "Alibaba Cloud Model Studio OpenAI-compatible endpoint configuration. "
        "When api_key is empty, DASHSCOPE_API_KEY is used."
    )

    def build_config(self, base_url, api_key):
        return ({"base_url": base_url.strip().rstrip("/"), "api_key": api_key.strip()},)


class ZNGBQwenLLMAPI:
    @classmethod
    def INPUT_TYPES(cls):
        optional_images = {f"image{index}": ("IMAGE",) for index in range(1, 9)}
        return {
            "required": {
                "api_config": (QWEN_CONFIG_TYPE,),
                "model": (QWEN_MODELS, {"default": "qwen3.7-plus"}),
                "custom_model_id": ("STRING", {
                    "default": "",
                    "tooltip": "Used only when model is set to custom.",
                }),
                "role": ("STRING", {"default": "", "multiline": True}),
                "prompt": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                "temperature": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.99, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "control_after_generate": True}),
                "max_completion_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072, "step": 1}),
                "top_p": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "enable_thinking": ("BOOLEAN", {"default": True}),
                "thinking_budget": ("INT", {
                    "default": DEFAULT_THINKING_BUDGET,
                    "min": 1,
                    "max": 131072,
                    "step": 1,
                    "tooltip": "Maximum reasoning tokens. Existing workflow value 0 falls back to 1024.",
                }),
                "vl_high_resolution_images": ("BOOLEAN", {"default": False}),
                "skip_error": ("BOOLEAN", {"default": False}),
            },
            "optional": optional_images,
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "ZNGBNodes/LLM"
    DESCRIPTION = (
        "Call a Qwen model through Alibaba Cloud Model Studio's OpenAI-compatible Chat API. "
        "Up to eight IMAGE inputs are sent as PNG data URLs. The final answer is returned "
        "in one non-streaming response; reasoning content is not exposed."
    )

    def generate(
        self,
        api_config,
        model,
        custom_model_id,
        role,
        prompt,
        temperature,
        seed,
        max_completion_tokens,
        top_p,
        presence_penalty,
        frequency_penalty,
        enable_thinking,
        thinking_budget,
        vl_high_resolution_images,
        skip_error,
        **kwargs,
    ):
        try:
            from openai import OpenAI

            api_key = api_config.get("api_key") or os.getenv("DASHSCOPE_API_KEY", "")
            base_url = api_config.get("base_url", "").strip().rstrip("/")
            if not api_key:
                raise ValueError("API Key is empty; set it in LLM config or DASHSCOPE_API_KEY")
            if not base_url:
                raise ValueError("base_url is empty")
            selected_model = custom_model_id.strip() if model == "custom" else model.strip()
            if not selected_model:
                raise ValueError("custom_model_id is empty while model is set to custom")

            messages = []
            if role.strip():
                messages.append({"role": "system", "content": role.strip()})
            images = [kwargs.get(f"image{index}") for index in range(1, 9)]
            messages.append(
                {"role": "user", "content": _build_user_content(prompt, images)}
            )

            extra_body = {
                "enable_thinking": enable_thinking,
                "vl_high_resolution_images": vl_high_resolution_images,
            }
            if enable_thinking:
                effective_thinking_budget = (
                    thinking_budget if thinking_budget > 0 else DEFAULT_THINKING_BUDGET
                )
                if effective_thinking_budget >= max_completion_tokens:
                    raise ValueError(
                        "thinking_budget must be smaller than max_completion_tokens "
                        "to leave room for the final response"
                    )
                extra_body["thinking_budget"] = effective_thinking_budget

            client = OpenAI(api_key=api_key, base_url=base_url)
            completion = client.chat.completions.create(
                model=selected_model,
                messages=messages,
                temperature=temperature,
                seed=seed,
                max_completion_tokens=max_completion_tokens,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                extra_body=extra_body,
                stream=False,
            )

            if not completion.choices:
                raise RuntimeError("Qwen API returned no choices")
            return (completion.choices[0].message.content or "",)
        except Exception as exc:
            if skip_error:
                return (f"Qwen API error: {exc}",)
            raise RuntimeError(f"Qwen API request failed: {exc}") from exc


NODE_CLASS_MAPPINGS = {
    "ZNGB_QwenLLMConfig": ZNGBQwenLLMConfig,
    "ZNGB_QwenLLMAPI": ZNGBQwenLLMAPI,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ZNGB_QwenLLMConfig": "LLM config",
    "ZNGB_QwenLLMAPI": "qwen LLM API ZNGB",
}