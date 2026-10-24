import dataclasses
import inspect
import json
import os
import subprocess
import sys

import pytest
import torch

from atas_geometry import device as accel
from tests.mac.conftest import GEOMETRY, darwin_only


def test_cuda_branch_identity(monkeypatch):
    """Under CUDA every helper builds exactly what the upstream inline code built."""
    monkeypatch.setenv("GEOMETRY_DEVICE", "cuda")
    import atas_geometry.service.device_mapping as mapping
    monkeypatch.setattr(mapping, "device_index", lambda physical: {5: 2}[physical])
    monkeypatch.setattr(mapping, "gpu_selector", lambda physical: f"GPU-{physical}")
    assert accel.torch_device(5) == torch.device("cuda:2")
    assert accel.backend() == "cuda" and accel.compute_dtype() is torch.bfloat16
    autocast = accel.autocast()
    assert (autocast.device, autocast.fast_dtype) == ("cuda", torch.bfloat16)
    assert accel.worker_env(5) == {"CUDA_VISIBLE_DEVICES": "GPU-5", "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    assert accel.float_dtype(torch.device("cuda:0")) is torch.float64
    assert accel.peak_memory_source() == "cuda_max_memory_reserved"
    assert accel.sam3_backend() == "official"
    calls = []
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: calls.append(device) or (123, 456))
    assert accel.free_memory_bytes(torch.device("cuda:2")) == 123 and calls == [torch.device("cuda:2")]
    class Event:
        def __init__(self, enable_timing): calls.append(("event", enable_timing))
        def record(self): calls.append("record")
        def elapsed_time(self, other): return 7.5
    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: calls.append("sync"))
    timer = accel.Timer(torch.device("cuda:2")).start()
    assert timer.stop_ms() == 7.5
    assert calls[1:] == [("event", True), ("event", True), "record", "record", "sync"]
    set_calls = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: set_calls.append(device))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: set_calls.append("empty"))
    accel.set_device(torch.device("cuda:2")); accel.empty_cache()
    assert set_calls == [torch.device("cuda:2"), "empty"]


@darwin_only
def test_mps_branch(monkeypatch):
    monkeypatch.delenv("GEOMETRY_DEVICE", raising=False)
    monkeypatch.delenv("GEOMETRY_MPS_DTYPE", raising=False)
    assert accel.backend() == "mps" and accel.torch_device(0) == torch.device("mps")
    assert accel.compute_dtype() is torch.float32 and accel.float_dtype(torch.device("mps")) is torch.float32
    assert not accel.autocast()._enabled
    monkeypatch.setenv("GEOMETRY_MPS_DTYPE", "bfloat16")
    assert accel.compute_dtype() is torch.bfloat16 and accel.autocast().device == "mps"
    assert accel.free_memory_bytes(torch.device("mps")) > 0
    x = torch.ones(1024, device="mps")
    timer = accel.Timer(torch.device("mps")).start()
    assert float((x * 2).sum().item()) == 2048.0 and timer.stop_ms() >= 0
    assert accel.peak_memory_bytes(torch.device("mps")) > 0
    assert accel.sam3_backend() == "cpp"
    with pytest.raises(ValueError):
        monkeypatch.setenv("GEOMETRY_MPS_DTYPE", "float64"); accel.compute_dtype()


def test_cpu_branch(monkeypatch):
    monkeypatch.setenv("GEOMETRY_DEVICE", "cpu")
    assert accel.torch_device(3) == torch.device("cpu")
    assert accel.compute_dtype() is torch.float32 and accel.float_dtype(torch.device("cpu")) is torch.float64
    assert accel.worker_env(3) == {"GEOMETRY_DEVICE": "cpu"}
    assert accel.free_memory_bytes(torch.device("cpu")) > 0
    monkeypatch.setenv("GEOMETRY_DEVICE", "rocm")
    with pytest.raises(ValueError):
        accel.backend()


def test_backend_never_imports_torch():
    code = "import sys; from atas_geometry import device; device.backend(); device.sam3_backend(); print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=GEOMETRY, check=True)
    assert out.stdout.strip() == "False"


def test_upstream_deadline_defaults_unchanged(monkeypatch):
    from atas_geometry.service.config import Settings
    from atas_geometry.service import isolated_worker
    defaults = {f.name: f.default for f in dataclasses.fields(Settings) if f.default is not dataclasses.MISSING}
    assert {k: defaults[k] for k in ("recon_deadline_seconds", "render_deadline_seconds",
            "rpc_deadline_seconds", "worker_heartbeat_timeout")} == {
            "recon_deadline_seconds": 900.0, "render_deadline_seconds": 300.0,
            "rpc_deadline_seconds": 120.0, "worker_heartbeat_timeout": 90.0}
    for key in list(os.environ):
        if key.startswith("GEOMETRY_"):
            monkeypatch.delenv(key)
    settings = Settings.from_env()
    assert (settings.recon_deadline_seconds, settings.render_deadline_seconds,
            settings.rpc_deadline_seconds, settings.worker_heartbeat_timeout) == (2400, 900, 300, 90)
    assert (settings.recon_min_free_mib, settings.gpu_min_free_mib) == (24576, 15360)
    assert "self.stop.wait(5)" in inspect.getsource(isolated_worker.WorkerStore._heartbeat)


def test_child_env_cuda_identical(monkeypatch):
    """The upstream call sites built exactly this pair of variables."""
    monkeypatch.setenv("GEOMETRY_DEVICE", "cuda")
    from atas_geometry.service.device_mapping import gpu_selector
    upstream = dict(CUDA_VISIBLE_DEVICES=gpu_selector(4), CUDA_DEVICE_ORDER="PCI_BUS_ID")
    assert accel.worker_env(4) == upstream


@darwin_only
def test_child_env_mps(monkeypatch):
    monkeypatch.delenv("GEOMETRY_DEVICE", raising=False)
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    env = accel.worker_env(0)
    assert env == {"GEOMETRY_DEVICE": "mps", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                   "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0", "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.8235"}
    assert "CUDA_VISIBLE_DEVICES" not in env
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    assert accel.worker_env(0)["PYTORCH_ENABLE_MPS_FALLBACK"] == "0"


@darwin_only
def test_child_env_effective_in_child(monkeypatch):
    """The child sees the variables before it imports torch (they are read once at import)."""
    from atas_geometry.service.unified_pool import probe_accelerator
    for key in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO", "GEOMETRY_DEVICE"):
        monkeypatch.delenv(key, raising=False)
    result = probe_accelerator(sys.executable, "mps", env={**os.environ, **accel.worker_env(0)})
    assert result["ok"] is True, json.dumps(result)
    assert result["env_before_import"] == {"PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "1.0",
                                           "PYTORCH_MPS_LOW_WATERMARK_RATIO": "0.8235"}
    assert result["recommended_max_memory"] > 0 and result["matmul_max_rel_err"] < 1e-4
