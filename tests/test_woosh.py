import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch
import urllib.request
import zipfile

import numpy as np
import torch

PLUGIN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PLUGIN))
import audio_ops
import download_models
import runtime
import worker


class AudioTests(unittest.TestCase):
    def test_crossfade_preserves_batch_channels(self):
        first = np.ones((2, 1, 10), dtype=np.float32)
        second = np.full((2, 1, 10), 3, dtype=np.float32)
        joined = audio_ops.append_audio(first, second, 4)
        self.assertEqual(joined.shape, (2, 1, 16))
        np.testing.assert_allclose(joined[0, 0, 6:10], [1, 5 / 3, 7 / 3, 3])
        np.testing.assert_equal(joined[:, :, :6], first[:, :, :6])

    def test_zero_and_oversized_overlap(self):
        segment = np.zeros((1, 1, 10), dtype=np.float32)
        self.assertEqual(audio_ops.append_audio(segment, segment, 0).shape[-1], 20)
        self.assertEqual(audio_ops.append_audio(segment, segment, 100).shape[-1], 15)

    def test_normalization_is_per_item_and_only_attenuates(self):
        values = np.array([[[2, -1, 0]], [[0.2, -0.1, 0]]], dtype=np.float64)
        result = audio_ops.finish_audio(values, 2, True)
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[[1, -0.5]], [[0.2, -0.1]]])
        np.testing.assert_equal(values[0], [[2, -1, 0]])

    def test_no_normalization_and_silence(self):
        values = np.full((1, 1, 10), 2, dtype=np.float32)
        np.testing.assert_equal(audio_ops.finish_audio(values, 3, False), values[..., :3])
        self.assertTrue(np.isfinite(audio_ops.finish_audio(values * 0, 10, True)).all())

    def test_reject_nonfinite_and_short_output(self):
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            audio_ops.finish_audio(np.full((1, 1, 2), np.nan), 2, True)
        with self.assertRaisesRegex(ValueError, "shorter"):
            audio_ops.finish_audio(np.zeros((1, 1, 2)), 3, True)

    def test_peak_crop_keeps_late_clicks_in_each_batch_item(self):
        audio = np.zeros((2, 1, 240000), dtype=np.float32)
        audio[0, 0, 120000] = 0.8
        audio[1, 0, 200000] = 0.6
        result = audio_ops.finish_audio(audio, 12000, True, "peak")
        self.assertEqual(result.shape, (2, 1, 12000))
        np.testing.assert_equal(result[:, 0, 1440], [np.float32(0.8), np.float32(0.6)])
        self.assertTrue((audio_ops.finish_audio(audio, 12000, True, "start") == 0).all())

    def test_peak_crop_near_edges_and_silence(self):
        audio = np.zeros((2, 1, 1000), dtype=np.float32)
        audio[0, 0, 5] = 0.8
        audio[1, 0, 980] = 0.6
        result = audio_ops.finish_audio(audio, 100, True, "peak")
        self.assertEqual(result.shape, (2, 1, 100))
        self.assertGreater(result[0].max(), 0)
        self.assertGreater(result[1].max(), 0)
        self.assertTrue((audio_ops.finish_audio(audio * 0, 100, True, "peak") == 0).all())


class RuntimeTests(unittest.TestCase):
    def test_model_selection_cannot_escape_root(self):
        with self.assertRaises(ValueError):
            runtime.resolve_runtime("../../outside", [])

    def test_missing_runtime_has_install_guidance(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(runtime, "PLUGIN_DIR", Path(directory)), patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(FileNotFoundError, "setup.ps1"):
                runtime.resolve_runtime("Woosh-DFlow", [])

    def test_custom_root_and_safetensors_requirement(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            source = root / "source"
            (source / "woosh").mkdir(parents=True)
            (source / "woosh" / "__init__.py").touch()
            (root / "runtime.json").write_text(json.dumps({"python": sys.executable, "source": str(source), "model_root": str(root / "models")}), encoding="utf-8")
            for name in ("Woosh-DFlow", "Woosh-AE", "TextConditionerA"):
                target = root / "models" / name
                target.mkdir(parents=True)
                (target / "config.yaml").touch()
                (target / "weights.safetensors").touch()
            with patch.object(runtime, "PLUGIN_DIR", root):
                python, _, models = runtime.resolve_runtime("Woosh-DFlow", [])
                self.assertEqual(python, Path(sys.executable).resolve())
                self.assertEqual(models, root / "models")
                (models / "Woosh-AE" / "weights.safetensors").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "Woosh-AE"):
                    runtime.resolve_runtime("Woosh-DFlow", [])

    def run_fake(self, code, timeout=10, interrupt=lambda: None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "worker.py").write_text(code, encoding="utf-8")
            progress = []
            processes = []
            original_popen = subprocess.Popen

            def record_process(*args, **kwargs):
                process = original_popen(*args, **kwargs)
                processes.append(process)
                return process

            with patch.object(runtime, "PLUGIN_DIR", root), patch.object(runtime.subprocess, "Popen", record_process):
                try:
                    result = runtime.run_worker(sys.executable, root, {"batch_size": 2}, interrupt,
                                                lambda current, total: progress.append((current, total)), timeout)
                    return result, progress
                finally:
                    self.assertTrue(processes)
                    self.assertIsNotNone(processes[0].poll(), "Backend was left running")

    def test_real_subprocess_result_and_offline_environment(self):
        code = '''import os, sys, numpy as np
assert os.environ['HF_HUB_OFFLINE'] == '1'
assert 'PYTHONPATH' not in os.environ
np.savez(sys.argv[2], waveform=np.zeros((2, 1, 480), np.float32), sample_rate=48000)
'''
        (waveform, rate), progress = self.run_fake(code)
        self.assertEqual(waveform.shape, (2, 1, 480))
        self.assertEqual(rate, 48000)
        self.assertEqual(progress[-1], (1, 1))

    def test_worker_error_contains_diagnostic(self):
        with self.assertRaisesRegex(RuntimeError, "example model error"):
            self.run_fake("raise RuntimeError('example model error')")

    def test_bad_audio_layout_rejected(self):
        with self.assertRaisesRegex(ValueError, "48000"):
            self.run_fake("import sys,numpy as np; np.savez(sys.argv[2], waveform=np.zeros((2,480)), sample_rate=44100)")

    def test_timeout_kills_process(self):
        with self.assertRaises(TimeoutError):
            self.run_fake("import time; time.sleep(30)", timeout=0.01)

    def test_interrupt_kills_process(self):
        def interrupt():
            raise InterruptedError("cancelled")
        with self.assertRaises(InterruptedError):
            self.run_fake("import time; time.sleep(30)", interrupt=interrupt)

    def test_partial_progress_write_is_tolerated(self):
        code = '''import sys, time, pathlib, numpy as np
pathlib.Path(sys.argv[3]).write_text('{', encoding='utf-8')
time.sleep(0.6)
np.savez(sys.argv[2], waveform=np.zeros((2,1,480),np.float32), sample_rate=48000)
'''
        self.run_fake(code)


class WorkerTests(unittest.TestCase):
    def fake_backend(self):
        calls = {"config": [], "sampler": []}

        class Model:
            def __init__(self, config):
                self.autoencoder = self
            def eval(self):
                return self
            def to(self, device):
                return self
            def get_cond(self, batch, **kwargs):
                calls["prompt"] = batch["description"]
                return {"prompt": batch["description"]}
            def inverse(self, latent):
                return latent[:, :1, :1].expand(-1, 1, 2400)

        def load_config(**kwargs):
            calls["config"].append(kwargs)
            return kwargs

        def sampler(**kwargs):
            calls["sampler"].append(kwargs)
            return kwargs["noise"]

        def flow(model, **kwargs):
            calls["sampler"].append(kwargs)
            return kwargs["noise"]

        definitions = {
            "woosh.components.base": {"LoadConfig": load_config},
            "woosh.model.flowmap_from_pretrained": {"FlowMapFromPretrained": Model},
            "woosh.model.ldm": {"LatentDiffusionModel": Model},
            "woosh.inference.flowmap_sampler": {"sample_euler": sampler},
            "woosh.inference.flowmatching_sampler": {"flowmatching_integrate": flow},
        }
        modules = {}
        for name, fields in definitions.items():
            modules[name] = types.ModuleType(name)
            modules[name].__dict__.update(fields)
        return modules, calls

    def generate(self, model="Woosh-DFlow", duration=0.12, steps=4):
        request = dict(model=model, model_root="E:/models/woosh", device="cpu", batch_size=2,
                       seed=123, prompt="rain", duration_seconds=duration, crossfade_seconds=0.01,
                       normalize=True, dflow_steps=steps, cfg=4.5)
        modules, calls = self.fake_backend()
        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules):
            audio = worker.generate(request, Path(directory) / "progress.json")
        return audio, calls

    def test_dflow_contract_and_long_assembly(self):
        audio, calls = self.generate()
        self.assertEqual(audio.shape, (2, 1, 5760))
        self.assertEqual(len(calls["sampler"]), 3)
        self.assertEqual(calls["prompt"], ["rain", "rain"])
        self.assertEqual(calls["sampler"][0]["renoise"], [0, 0.5, 0.5, 0.3])
        self.assertEqual(calls["sampler"][0]["noise"].shape, (2, 128, 501))
        self.assertEqual(calls["config"][0]["ldm"]["autoencoder"]["path"], str(Path("E:/models/woosh/Woosh-AE")))
        self.assertLessEqual(np.abs(audio).max(), 1)

    def test_seed_reproducibility(self):
        first, _ = self.generate()
        second, _ = self.generate()
        np.testing.assert_equal(first, second)

    def test_nondefault_dflow_steps(self):
        _, calls = self.generate(steps=8)
        self.assertEqual(calls["sampler"][0]["num_steps"], 8)
        self.assertEqual(calls["sampler"][0]["renoise"], 0.0)

    def test_flow_contract_and_short_crop(self):
        audio, calls = self.generate(model="Woosh-Flow", duration=0.025)
        self.assertEqual(audio.shape[-1], 1200)
        self.assertNotIn("ldm", calls["config"][0])
        self.assertIn("conditioners", calls["config"][0])
        self.assertEqual(calls["sampler"][0]["atol"], 0.001)


class DownloadTests(unittest.TestCase):
    def test_extracts_only_expected_files_from_official_assets(self):
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as archive:
            archive.writestr("checkpoints/model/config.yaml", "sample_rate: 48000")
            archive.writestr("checkpoints/model/weights.safetensors", b"test")
            archive.writestr("../../outside.txt", b"unexpected")
        payload = data.getvalue()
        release = {"assets": [{"name": f"{name}.zip", "size": len(payload), "browser_download_url": f"https://example.invalid/{name}"}
                              for name in ("Woosh-DFlow", "Woosh-AE", "TextConditionerA")]}

        def response(request, **kwargs):
            return io.BytesIO(json.dumps(release).encode() if isinstance(request, urllib.request.Request) else payload)

        with tempfile.TemporaryDirectory() as directory, patch.object(download_models.urllib.request, "urlopen", response):
            root = Path(directory)
            download_models.fetch_models("Woosh-DFlow", root / "models")
            self.assertFalse((root / "outside.txt").exists())
            for name in ("Woosh-DFlow", "Woosh-AE", "TextConditionerA"):
                self.assertTrue((root / "models" / name / "weights.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
