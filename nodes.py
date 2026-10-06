from pathlib import Path

import numpy as np
import torch

import comfy.model_management as mm
from comfy.utils import ProgressBar
import folder_paths

from .runtime import MODELS, resolve_runtime, run_worker

folder_paths.add_model_folder_path("woosh", str(Path(folder_paths.models_dir) / "woosh"))


class WooshGenerate:
    CATEGORY = "audio/Woosh"
    FUNCTION = "generate"
    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    DESCRIPTION = "本地 Woosh 文本音效生成；超过约 5 秒时分段生成并交叉淡化拼接。独立环境需先运行 setup.ps1。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": "Steady rain falling on leaves in a quiet forest, distant thunder, natural ambience, no speech, no music."}),
            "model": (list(MODELS),),
            "duration_seconds": ("FLOAT", {"default": 5.0, "min": 0.1, "max": 300.0, "step": 0.1, "tooltip": "超过约 5 秒会分段生成；适合环境声，无法保证跨段事件连续。"}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "control_after_generate": True}),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 16}),
            "cfg": ("FLOAT", {"default": 4.5, "min": 0.0, "max": 15.0, "step": 0.1}),
            "dflow_steps": ("INT", {"default": 4, "min": 1, "max": 64, "tooltip": "仅 DFlow 使用；Flow 使用官方自适应 ODE 采样。"}),
            "crossfade_seconds": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0, "step": 0.05}),
            "device": (["auto", "cuda", "cpu", "mps"],),
            "normalize": ("BOOLEAN", {"default": True, "tooltip": "只缩小超过 ±1 的音频，避免保存时削波；不会放大小音量。"}),
        }, "optional": {
            "release_comfy_models": ("BOOLEAN", {"default": True, "tooltip": "生成前卸载 ComfyUI 已加载模型，为独立 Woosh 进程腾出显存。"}),
            "timeout_seconds": ("INT", {"default": 1800, "min": 30, "max": 86400}),
            "crop_mode": (["start", "peak"], {"default": "start", "tooltip": "start 从开头截取；peak 用于短音效，围绕每条音频的最大振幅截取，并保留最多 30ms 前导。多段环境声保持从开头截取。"}),
        }}

    def generate(self, prompt, model, duration_seconds, seed, batch_size, cfg, dflow_steps,
                 crossfade_seconds, device, normalize, release_comfy_models=True, timeout_seconds=1800, crop_mode="start"):
        if not np.isfinite(duration_seconds) or duration_seconds <= 0:
            raise ValueError("duration_seconds 必须是正数。")
        if not np.isfinite(crossfade_seconds) or crossfade_seconds < 0:
            raise ValueError("crossfade_seconds 必须非负。")
        if device not in ("auto", "cuda", "cpu", "mps"):
            raise ValueError(f"Unsupported device: {device}")
        if crop_mode not in ("start", "peak"):
            raise ValueError(f"Unsupported crop mode: {crop_mode}")
        python, source, root = resolve_runtime(model, folder_paths.get_folder_paths("woosh"))
        mm.throw_exception_if_processing_interrupted()
        if release_comfy_models:
            mm.unload_all_models()
            mm.soft_empty_cache()
        bar = ProgressBar(100)
        bar.update_absolute(0)
        request = dict(prompt=prompt, model=model, model_root=str(root), duration_seconds=duration_seconds,
                       seed=seed, batch_size=batch_size, cfg=cfg, dflow_steps=dflow_steps,
                       crossfade_seconds=crossfade_seconds, device=device, normalize=normalize, crop_mode=crop_mode)
        waveform, sample_rate = run_worker(
            python, source, request, mm.throw_exception_if_processing_interrupted,
            lambda current, total: bar.update_absolute(round(100 * current / total)), timeout_seconds,
        )
        return ({"waveform": torch.from_numpy(waveform), "sample_rate": sample_rate},)
