"""Admission for one unified-memory accelerator (Apple Silicon MPS, or CPU).

Same interface and resource errors as GpuPool. Capacity is the accelerator's
``recommended_max_memory`` measured once in a fresh probe process, so the HTTP
process never imports torch; free memory is bounded by the host's available memory.
"""
from __future__ import annotations

import json
import subprocess
import sys
import threading
import time

DEVICE_INDEX = 0

PROBE = r"""
import json, os, sys, time
seen = {k: os.environ.get(k) for k in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_HIGH_WATERMARK_RATIO", "PYTORCH_MPS_LOW_WATERMARK_RATIO")}
import torch
kind = sys.argv[1]
out = {"device": kind, "torch": torch.__version__, "env_before_import": seen}
if kind == "mps":
    out["available"] = bool(torch.backends.mps.is_available())
    if out["available"]:
        torch.manual_seed(0)
        a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
        t = time.perf_counter(); am, bm = a.to("mps"), b.to("mps"); product = am @ bm; torch.mps.synchronize()
        out["matmul_ms"] = (time.perf_counter() - t) * 1000
        # Residency proof: the operands and result are held by the MPS allocator, not host memory.
        out["mps_allocated_bytes"] = int(torch.mps.current_allocated_memory())
        out["result_device"] = str(product.device)
        got = product.cpu()
        ref = a @ b
        out["matmul_max_rel_err"] = float(((got - ref).abs().max() / ref.abs().max()).item())
        out["recommended_max_memory"] = int(torch.mps.recommended_max_memory())
else:
    import psutil
    out["available"] = True
    out["recommended_max_memory"] = int(psutil.virtual_memory().total)
print("ATAS_PROBE " + json.dumps(out))
"""


def probe_accelerator(python: str, kind: str, env=None, timeout: float = 180.0) -> dict:
    """Run PROBE in a fresh interpreter; the result reflects the child's own environment."""
    try:
        completed = subprocess.run([python, "-c", PROBE, kind], capture_output=True, text=True,
                                   timeout=timeout, env=env)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"device": kind, "available": False, "error": f"{type(exc).__name__}: {exc}"}
    for line in completed.stdout.splitlines():
        if line.startswith("ATAS_PROBE "):
            result = json.loads(line[len("ATAS_PROBE "):])
            ok = result.get("available") and (kind != "mps" or (result.get("matmul_max_rel_err", 1.0) < 1e-4
                 and result.get("mps_allocated_bytes", 0) >= 3 * 1024 * 1024 * 4 and result.get("result_device") == "mps:0"))
            result["ok"] = bool(ok)
            return result
    return {"device": kind, "available": False, "ok": False,
            "error": (completed.stderr or completed.stdout)[-2000:]}


def host_available_bytes() -> int:
    import psutil  # host_statistics64 on macOS: free + inactive pages
    return int(psutil.virtual_memory().available)


class UnifiedMemoryPool:
    """One accelerator, one job at a time: unified memory is shared by every job."""

    def __init__(self, *, kind: str, python: str, probe=None, available_bytes=host_available_bytes,
                 poll_seconds: float = 5.0, **_ignored) -> None:
        from atas_geometry.device import worker_env
        self.kind = kind
        self.poll_seconds = float(poll_seconds)
        self.max_gpus = 1
        self.indices = (DEVICE_INDEX,)
        if probe is None:
            import os
            probe = probe_accelerator(python, kind, env={**os.environ, **worker_env(DEVICE_INDEX)})
        self.probe = probe
        self.capacity_mib = int(probe.get("recommended_max_memory", 0)) // 2**20 if probe.get("ok") else 0
        self._available_bytes = available_bytes
        self._busy: set[int] = set()
        self._quarantined = False
        self._cv = threading.Condition()

    def free_mib(self) -> int:
        return max(0, min(self.capacity_mib, self._available_bytes() // 2**20))

    def snapshot(self) -> dict[str, object]:
        return {
            "pool": list(self.indices),
            "max_concurrent_gpus": self.max_gpus,
            "busy": sorted(self._busy),
            "accelerator": self.kind,
            "cards": [{"index": DEVICE_INDEX, "free_mib": self.free_mib(), "capacity_mib": self.capacity_mib,
                       "busy": DEVICE_INDEX in self._busy, "telemetry_valid": bool(self.probe.get("ok"))}],
        }

    def acquire(self, *, min_free_mib: int | None = None, stop: threading.Event,
                timeout_seconds: float | None = None) -> int:
        need = int(min_free_mib or 0)
        started = time.monotonic()
        while not stop.is_set():
            if timeout_seconds is not None and time.monotonic() - started > timeout_seconds:
                from atas_geometry.service.resource_errors import ResourceUnavailableError
                raise ResourceUnavailableError("gpu_capacity_unavailable", "GPU resource queue deadline exceeded",
                    min_free_mib=need, candidate_gpu_indices=list(self.indices),
                    waited_seconds=round(time.monotonic() - started, 3),
                    accelerator=self.kind, capacity_mib=self.capacity_mib)
            with self._cv:
                if (not self._busy and not self._quarantined and self.probe.get("ok")
                        and self.free_mib() >= need):
                    self._busy.add(DEVICE_INDEX)
                    return DEVICE_INDEX
                self._cv.wait(timeout=min(self.poll_seconds, .1))
            if self._busy:
                continue
            stop.wait(min(self.poll_seconds, 1.0))
        raise RuntimeError("stopped while waiting for a GPU")

    def release(self, index: int) -> None:
        with self._cv:
            self._busy.discard(index)
            self._cv.notify_all()

    def quarantine(self, index: int) -> None:
        with self._cv:
            if index in self.indices:
                self._quarantined = True
                self._busy.add(index)
