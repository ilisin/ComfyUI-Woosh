"""Standalone backend; this file is executed by Woosh's Python, not imported by ComfyUI."""
import json
from pathlib import Path
import sys

import numpy as np
import torch

from audio_ops import SAMPLE_RATE, append_audio, finish_audio


def generate(request, progress_path):
    # Optional backend imports belong only to the separate runtime.
    from woosh.components.base import LoadConfig
    from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
    from woosh.model.ldm import LatentDiffusionModel
    from woosh.inference.flowmap_sampler import sample_euler
    from woosh.inference.flowmatching_sampler import flowmatching_integrate

    root = Path(request["model_root"])
    component_overrides = {
        "conditioners": {"text": {"path": str(root / "TextConditionerA")}},
        "autoencoder": {"path": str(root / "Woosh-AE")},
    }
    model_name = request["model"]
    if model_name == "Woosh-DFlow":
        model = FlowMapFromPretrained(LoadConfig(path=str(root / model_name), ldm=component_overrides))
    elif model_name == "Woosh-Flow":
        model = LatentDiffusionModel(LoadConfig(path=str(root / model_name), **component_overrides))
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    device = request["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.eval().to(device)
    batch = request["batch_size"]
    torch.manual_seed(request["seed"])
    cond = model.get_cond({"audio": None, "description": [request["prompt"]] * batch}, no_dropout=True, device=device)
    total = max(1, round(request["duration_seconds"] * SAMPLE_RATE))
    overlap = round(request["crossfade_seconds"] * SAMPLE_RATE)
    audio = None
    while audio is None or audio.shape[-1] < total:
        noise = torch.randn(batch, 128, 501, device=device)
        if model_name == "Woosh-DFlow":
            steps = request["dflow_steps"]
            renoise = [0, 0.5, 0.5, 0.3] if steps == 4 else 0.0
            latent = sample_euler(model=model, noise=noise, cond=cond.copy(), num_steps=steps, renoise=renoise, cfg=request["cfg"])
        else:
            latent = flowmatching_integrate(model, noise=noise, cond=cond, cfg=request["cfg"],
                                            atol=0.001, rtol=0.001, device=device)
        segment = model.autoencoder.inverse(latent).detach().cpu().float().numpy()
        if segment.ndim != 3 or segment.shape[:2] != (batch, 1) or segment.shape[-1] < 2:
            raise ValueError(f"Unexpected Woosh audio shape: {segment.shape}")
        audio = append_audio(audio, segment, overlap)
        progress_path.write_text(json.dumps({"samples": min(total, audio.shape[-1]), "total": total}), encoding="utf-8")
    crop_mode = request.get("crop_mode", "start") if total <= segment.shape[-1] else "start"
    return finish_audio(audio, total, request["normalize"], crop_mode)


def main():
    request_path, result_path, progress_path, source = map(Path, sys.argv[1:])
    sys.path.insert(0, str(source))
    # This standalone process never trains; Flow's ODE solver needs autograd disabled.
    torch.set_grad_enabled(False)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    audio = generate(request, progress_path)
    np.savez(result_path, waveform=audio, sample_rate=np.int64(SAMPLE_RATE))


if __name__ == "__main__":
    main()
