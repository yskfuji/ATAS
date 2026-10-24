import subprocess
import sys
import threading

import pytest

from atas_geometry.service.resource_errors import ResourceUnavailableError
from atas_geometry.service.unified_pool import UnifiedMemoryPool
from tests.mac.conftest import GEOMETRY, OK_PROBE

GIB = 2**30


def pool(available=40 * GIB, probe=OK_PROBE):
    return UnifiedMemoryPool(kind="mps", python=sys.executable, probe=dict(probe),
                             available_bytes=lambda: available, poll_seconds=.05)


def test_admits_within_budget():
    p = pool()
    assert p.free_mib() == 36 * 1024
    assert p.acquire(min_free_mib=24576, stop=threading.Event(), timeout_seconds=1) == 0
    snap = p.snapshot()
    assert snap["busy"] == [0] and snap["cards"][0]["telemetry_valid"] is True and snap["accelerator"] == "mps"


def test_refuses_over_budget_with_upstream_error():
    p = pool(available=20 * GIB)
    with pytest.raises(ResourceUnavailableError) as error:
        p.acquire(min_free_mib=24576, stop=threading.Event(), timeout_seconds=.3)
    fields = error.value.fields()
    assert fields["error_code"] == "gpu_capacity_unavailable" and fields["retryable"] is True
    assert fields["resource_details"]["min_free_mib"] == 24576
    failed = pool(probe={"device": "mps", "ok": False, "error": "no mps"})
    with pytest.raises(ResourceUnavailableError):
        failed.acquire(min_free_mib=2048, stop=threading.Event(), timeout_seconds=.2)


def test_single_lease():
    p = pool()
    stop = threading.Event()
    assert p.acquire(min_free_mib=2048, stop=stop, timeout_seconds=1) == 0
    with pytest.raises(ResourceUnavailableError):
        p.acquire(min_free_mib=2048, stop=stop, timeout_seconds=.3)
    p.release(0)
    assert p.acquire(min_free_mib=2048, stop=stop, timeout_seconds=1) == 0
    p.quarantine(0); p.release(0)
    with pytest.raises(ResourceUnavailableError):
        p.acquire(min_free_mib=2048, stop=stop, timeout_seconds=.3)


def test_requirements_match_upstream(app_client):
    _, client = app_client
    body = client.get("/readyz").json()
    assert body["resource_admission"] == {"recon_min_free_mib": 24576, "render_min_free_mib": 2048}
    import inspect
    from atas_geometry.service.jobs import GeometryRuntime
    source = inspect.getsource(GeometryRuntime._run_isolated)
    assert 'required_memory = 2048 if job["kind"] == "render" else max(' in source


def test_parent_admission_never_queries_torch_mps(tmp_path):
    """Upstream's HTTP process imports torch (via memory.fuse); admission must still never read the
    parent's own MPS allocator, which is about 0 there. Every torch.mps query raises in the parent."""
    code = f"""
import os, sys, threading
os.environ.update(GEOMETRY_HOT_ROOT={str(tmp_path / 'hot')!r}, GEOMETRY_PYTHON=sys.executable)
os.environ.pop('GEOMETRY_DEVICE', None)
import torch
def forbidden(*a, **k): raise AssertionError('parent queried the MPS allocator')
for name in ('recommended_max_memory', 'driver_allocated_memory', 'current_allocated_memory', 'synchronize', 'empty_cache'):
    setattr(torch.mps, name, forbidden)
for name in dir(torch._C):
    if name.startswith('_mps_'):
        setattr(torch._C, name, forbidden)
from atas_geometry.service.app import create_app
from fastapi.testclient import TestClient
app = create_app()
pool = app.state.runtime.pool
with TestClient(app) as client:
    ready = client.get('/readyz').json()
gpu = pool.acquire(min_free_mib=2048, stop=threading.Event(), timeout_seconds=5); pool.release(gpu)
app.state.runtime._stop.set()
print(type(pool).__name__, pool.probe.get('ok'), ready['dependencies']['accelerator_probe']['ok'])
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=GEOMETRY, timeout=300)
    assert out.returncode == 0, out.stderr[-3000:]
    assert out.stdout.split()[-3:] == ["UnifiedMemoryPool", "True", "True"]


def test_admission_uses_probe_value():
    """The admitted budget is the probe's figure, bounded by host memory; nothing else."""
    stop = threading.Event()
    small = UnifiedMemoryPool(kind="mps", python=sys.executable, available_bytes=lambda: 64 * GIB,
                              probe={**OK_PROBE, "recommended_max_memory": 20 * GIB}, poll_seconds=.05)
    with pytest.raises(ResourceUnavailableError):
        small.acquire(min_free_mib=24576, stop=stop, timeout_seconds=.3)
    large = UnifiedMemoryPool(kind="mps", python=sys.executable, available_bytes=lambda: 64 * GIB,
                              probe={**OK_PROBE, "recommended_max_memory": 30 * GIB}, poll_seconds=.05)
    assert large.acquire(min_free_mib=24576, stop=stop, timeout_seconds=1) == 0
    assert large.snapshot()["cards"][0]["capacity_mib"] == 30 * 1024
