import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import numpy as np

PLUGIN_DIR = Path(__file__).resolve().parent
MODELS = ("Woosh-DFlow", "Woosh-Flow")


def resolve_runtime(model, model_roots):
    if model not in MODELS:
        raise ValueError(f"Unsupported Woosh model: {model}")
    config_path = PLUGIN_DIR / "runtime.json"
    config = json.loads(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
    source = Path(os.environ.get("WOOSH_SOURCE") or config.get("source") or PLUGIN_DIR / "runtime" / "Woosh").expanduser().resolve()
    python_default = source / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = Path(os.environ.get("WOOSH_PYTHON") or config.get("python") or python_default).expanduser().resolve()
    if not python.is_file() or not (source / "woosh" / "__init__.py").is_file():
        raise FileNotFoundError(
            "Woosh 独立环境尚未配置。请运行插件目录的 setup.ps1，"
            "或在 runtime.json 中设置 python 和 source（参见 README.md）。"
        )
    override = os.environ.get("WOOSH_MODEL_DIR") or config.get("model_root")
    roots = [Path(override)] if override else [Path(p) for p in model_roots] + [source / "checkpoints"]
    components = (model, "Woosh-AE", "TextConditionerA")
    missing_by_root = []
    for root in roots:
        root = root.expanduser().resolve()
        missing = [f"{name}/{filename}" for name in components for filename in ("config.yaml", "weights.safetensors")
                   if not (root / name / filename).is_file()]
        if not missing:
            return python, source, root
        missing_by_root.append(f"{root}: {', '.join(missing)}")
    raise FileNotFoundError(
        "缺少 Woosh 模型文件：\n" + "\n".join(missing_by_root)
        + f"\n请按 README 下载官方权重，或运行 setup.ps1 -DownloadModels -Model {model}。"
    )


def run_worker(python, source, request, check_interrupt, progress, timeout_seconds):
    with tempfile.TemporaryDirectory(prefix="comfy-woosh-") as temporary:
        temporary = Path(temporary)
        request_path = temporary / "request.json"
        result_path = temporary / "audio.npz"
        progress_path = temporary / "progress.json"
        request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        environment = os.environ.copy()
        environment.update({"PYTHONUTF8": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1"})
        # Keep the backend isolated from ComfyUI's Python module search path.
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        started = time.monotonic()
        with (temporary / "worker.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(python), "-u", str(PLUGIN_DIR / "worker.py"), str(request_path), str(result_path), str(progress_path), str(source)],
                cwd=source, env=environment, stdout=log, stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                while process.poll() is None:
                    check_interrupt()
                    if time.monotonic() - started > timeout_seconds:
                        raise TimeoutError(f"Woosh 超过 {timeout_seconds} 秒。可提高 timeout_seconds，或降低批量 / 时长。")
                    if progress_path.exists():
                        try:
                            state = json.loads(progress_path.read_text(encoding="utf-8"))
                        except json.JSONDecodeError:
                            pass  # The worker may be writing the next segment's progress.
                        else:
                            progress(state["samples"], state["total"])
                    time.sleep(0.25)
                check_interrupt()
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()
        if process.returncode:
            detail = (temporary / "worker.log").read_text(encoding="utf-8")[-6000:]
            raise RuntimeError(f"Woosh 后端运行失败（退出码 {process.returncode}）：\n{detail}")
        if not result_path.is_file():
            raise RuntimeError("Woosh 后端未返回音频文件。")
        with np.load(result_path, allow_pickle=False) as result:
            waveform = result["waveform"].copy()
            sample_rate = int(result["sample_rate"])
        if waveform.ndim != 3 or waveform.shape[:2] != (request["batch_size"], 1) or sample_rate != 48000:
            raise ValueError("Woosh 返回的音频格式不正确；应为 [batch, 1, samples] / 48000 Hz。")
        progress(1, 1)
        return waveform, sample_rate
