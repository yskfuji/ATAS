"""The single accelerator decision for the geometry service.

On CUDA every helper returns exactly what upstream code constructed inline, so
NVIDIA deployments behave byte-for-byte as before. On Apple Silicon the same
call sites run on MPS (or CPU). ``backend()`` never imports torch: the HTTP
process decides admission without touching an accelerator.
"""
from __future__ import annotations

import contextlib
import os
import sys
import time

BACKENDS = ("cuda", "mps", "cpu")
# MPS has no float64 and Apple GPUs no bf16 flash kernels; fp32 is the safe default
# until tools/pi3_dtype_probe.py selects a reduced precision with evidence.
DEFAULT_MPS_DTYPE = "float32"


def backend() -> str:
    explicit = os.getenv("GEOMETRY_DEVICE", "").strip().lower()
    if explicit:
        if explicit not in BACKENDS:
            raise ValueError(f"GEOMETRY_DEVICE must be one of {BACKENDS}, got {explicit!r}")
        return explicit
    return "mps" if sys.platform == "darwin" else "cuda"


def torch_device(gpu_index=None):
    import torch
    kind = backend()
    if kind == "cuda":
        from atas_geometry.service.device_mapping import device_index
        return torch.device(f"cuda:{device_index(gpu_index)}") if gpu_index is not None else torch.device("cuda")
    return torch.device(kind)


def device_type(device=None) -> str:
    if device is None:
        return backend()
    return getattr(device, "type", None) or str(device).split(":", 1)[0]


def set_device(device) -> None:
    import torch
    if device_type(device) == "cuda":
        torch.cuda.set_device(device)


def synchronize(device=None) -> None:
    import torch
    kind = device_type(device)
    if kind == "cuda":
        if device is None:
            torch.cuda.synchronize()
        else:
            torch.cuda.synchronize(device)
    elif kind == "mps":
        torch.mps.synchronize()


def empty_cache(device=None) -> None:
    import torch
    kind = device_type(device)
    if kind == "cuda":
        torch.cuda.empty_cache()
    elif kind == "mps":
        torch.mps.empty_cache()


def compute_dtype(device=None):
    """Reconstruction precision: bf16 on CUDA (upstream), configured on MPS, fp32 on CPU."""
    import torch
    kind = device_type(device)
    if kind == "cuda":
        return torch.bfloat16
    if kind == "mps":
        name = os.getenv("GEOMETRY_MPS_DTYPE", DEFAULT_MPS_DTYPE).strip().lower()
        try:
            return {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[name]
        except KeyError:
            raise ValueError(f"GEOMETRY_MPS_DTYPE must be float32, bfloat16 or float16, got {name!r}") from None
    return torch.float32


def autocast(device=None, dtype=None):
    """``torch.autocast(<backend>, dtype)``; disabled when the compute dtype is fp32."""
    import torch
    kind = device_type(device)
    dtype = dtype or compute_dtype(device)
    if dtype == torch.float32:
        return torch.autocast(kind, enabled=False)
    return torch.autocast(kind, dtype=dtype)


def float_dtype(device):
    """float64 where the device has it; MPS computes in float32."""
    import torch
    return torch.float32 if device_type(device) == "mps" else torch.float64


def as_device_tensor(values, device):
    """torch.as_tensor on ``device``; float64 data becomes float32 only where the device lacks float64."""
    import numpy as np
    import torch
    if device_type(device) == "mps":
        array = np.asarray(values)
        if array.dtype == np.float64:
            values = array.astype(np.float32)
    return torch.as_tensor(values, device=device)


def free_memory_bytes(device) -> int:
    import torch
    kind = device_type(device)
    if kind == "cuda":
        free, _ = torch.cuda.mem_get_info(device)
        return int(free)
    if kind == "mps":
        return max(0, int(torch.mps.recommended_max_memory()) - int(torch.mps.driver_allocated_memory()))
    import psutil
    return int(psutil.virtual_memory().available)


def reset_peak_memory(device=None) -> None:
    import torch
    if device_type(device) == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_bytes(device=None):
    """CUDA: allocator reserved peak. MPS/CPU: the process' lifetime peak physical
    footprint (macOS counts Metal buffers in it); None when unavailable."""
    import torch
    kind = device_type(device)
    if kind == "cuda":
        if not torch.cuda.is_available():
            return None
        index = getattr(device, "index", None) if device is not None else None
        return int(torch.cuda.max_memory_reserved(index) if index is not None else torch.cuda.max_memory_reserved())
    if sys.platform == "darwin":
        from atas_geometry.service.proc_darwin import lifetime_peak_footprint
        return lifetime_peak_footprint(os.getpid())
    return None


def peak_memory_source(device=None) -> str:
    return "cuda_max_memory_reserved" if device_type(device) == "cuda" else "darwin_lifetime_max_phys_footprint"


class Timer:
    """Wall time of device work; CUDA keeps its event timing."""

    def __init__(self, device=None):
        self.device = device
        self.kind = device_type(device)
        self._start = self._end = None

    def start(self):
        if self.kind == "cuda":
            import torch
            self._start = torch.cuda.Event(enable_timing=True)
            self._end = torch.cuda.Event(enable_timing=True)
            self._start.record()
        else:
            self._t0 = time.perf_counter()
        return self

    def stop_ms(self) -> float:
        if self.kind == "cuda":
            import torch
            self._end.record()
            torch.cuda.synchronize()
            return float(self._start.elapsed_time(self._end))
        synchronize(self.device)
        return (time.perf_counter() - self._t0) * 1000.0


def sam3_backend() -> str:
    """official (PyTorch SAM3, CUDA) or cpp (sam3.cpp; the only path without CUDA)."""
    name = os.getenv("GEOMETRY_SAM3_BACKEND", "").strip().lower() or ("official" if backend() == "cuda" else "cpp")
    if name not in {"official", "cpp"}:
        raise ValueError(f"GEOMETRY_SAM3_BACKEND must be official or cpp, got {name!r}")
    return name


def sam3cpp_binary() -> str:
    return os.getenv("GEOMETRY_SAM3CPP_BIN", "atas_sam3_track")


def worker_env(gpu_index) -> dict[str, str]:
    """Accelerator variables for a child process; set before the child imports torch."""
    kind = backend()
    if kind == "cuda":
        from atas_geometry.service.device_mapping import gpu_selector
        return {"CUDA_VISIBLE_DEVICES": gpu_selector(gpu_index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    env = {"GEOMETRY_DEVICE": kind}
    if kind == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        # 1.0 caps MPS at recommended_max_memory: over-budget work fails with an
        # MPS OOM (a classified resource error) instead of swapping.
        high = os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
        env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = high
        # torch rejects a low (soft) watermark above the high one; keep the default 1.4:1.7 proportion.
        env["PYTORCH_MPS_LOW_WATERMARK_RATIO"] = os.environ.get(
            "PYTORCH_MPS_LOW_WATERMARK_RATIO", f"{float(high) * 1.4 / 1.7:.4f}")
    return env


@contextlib.contextmanager
def no_autocast(device=None):
    import torch
    with torch.autocast(device_type(device), enabled=False):
        yield
