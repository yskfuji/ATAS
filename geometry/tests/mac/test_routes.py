import time

import pytest

from tests.mac.conftest import darwin_only

RENDER = {"type": "render", "run_id": "run", "story_id": "story", "shot_id": "shot_01", "scene_id": "scene"}


@darwin_only
def test_startup_without_nvidia_smi(service_env, monkeypatch):
    marker = service_env["root"] / "nvidia-smi-called"
    stub = service_env["root"] / "stubbin" / "nvidia-smi"
    stub.parent.mkdir()
    stub.write_text(f"#!/bin/sh\ntouch {marker}\nexit 1\n")
    stub.chmod(0o755)
    monkeypatch.setenv("PATH", f"{stub.parent}:/usr/bin:/bin")
    from fastapi.testclient import TestClient
    from atas_geometry.service.app import create_app
    from atas_geometry.service.config import Settings
    app = create_app(Settings.from_env())
    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"status": "ok"}
        assert client.get("/readyz").status_code in (200, 503)
    app.state.runtime._stop.set()
    assert not marker.exists()


def test_cuda_device_still_selects_nvidia_pool(service_env, monkeypatch):
    import atas_geometry.service.app as module
    from atas_geometry.service.config import Settings
    built = []
    class Sentinel:
        def __init__(self, **kwargs): built.append(kwargs)
        def snapshot(self): return {}
    monkeypatch.setenv("GEOMETRY_DEVICE", "cuda")
    monkeypatch.setattr(module, "GpuPool", Sentinel)
    app = module.create_app(Settings.from_env())
    app.state.runtime._stop.set()
    assert len(built) == 1 and set(built[0]) == {"max_gpus", "configured", "min_free_mib", "poll_seconds", "exclude_gpu0"}


def test_readyz_all_ok(app_client):
    _, client = app_client
    response = client.get("/readyz")
    body = response.json()
    assert response.status_code == 200 and body["ready"] is True, body
    assert body["dependencies"]["accelerator_probe"]["ok"] is True
    assert body["dependencies"]["sam3cpp_binary"]["ok"] is True
    assert "nvidia_smi" not in body["dependencies"]


@pytest.mark.parametrize("missing", ["python", "pi3_model", "sam3_checkpoint", "sam3cpp_binary", "accelerator_probe"])
def test_readyz_missing_dependency(service_env, monkeypatch, missing):
    import atas_geometry.service.unified_pool as unified_pool
    root = service_env["root"]
    if missing == "python":
        monkeypatch.setenv("GEOMETRY_PYTHON", str(root / "no-python"))
    elif missing == "pi3_model":
        monkeypatch.setenv("GEOMETRY_PI3_MODEL", str(root / "no-pi3x"))
    elif missing == "sam3_checkpoint":
        monkeypatch.setenv("GEOMETRY_SAM3_CHECKPOINT", str(root / "no.ggml"))
    elif missing == "sam3cpp_binary":
        monkeypatch.setenv("GEOMETRY_SAM3CPP_BIN", str(root / "no-binary"))
    else:
        monkeypatch.setattr(unified_pool, "probe_accelerator", lambda *a, **k: {"device": "mps", "ok": False, "error": "x"})
    from fastapi.testclient import TestClient
    from atas_geometry.service.app import create_app
    from atas_geometry.service.config import Settings
    app = create_app(Settings.from_env())
    with TestClient(app) as client:
        response = client.get("/readyz")
    app.state.runtime._stop.set()
    body = response.json()
    assert response.status_code == 503 and body["ready"] is False
    failing = [name for name, check in body["dependencies"].items() if not check["ok"]]
    assert failing == [missing]


def test_healthz_no_auth(service_env, monkeypatch):
    monkeypatch.setenv("GEOMETRY_API_TOKEN", "secret")
    from fastapi.testclient import TestClient
    from atas_geometry.service.app import create_app
    from atas_geometry.service.config import Settings
    app = create_app(Settings.from_env())
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/readyz").status_code in (200, 503)
    app.state.runtime._stop.set()


def test_auth_required(service_env, monkeypatch):
    monkeypatch.setenv("GEOMETRY_API_TOKEN", "secret")
    from fastapi.testclient import TestClient
    from atas_geometry.service.app import create_app
    from atas_geometry.service.config import Settings
    app = create_app(Settings.from_env())
    with TestClient(app) as client:
        routes = [("post", "/v1/jobs"), ("get", "/v1/jobs/x"), ("post", "/v1/camera-context"),
                  ("post", "/v1/world-publication"), ("post", "/v1/world-rebase"),
                  ("get", "/v1/jobs/x/artifacts/memory_input.mp4")]
        for method, path in routes:
            assert getattr(client, method)(path).status_code == 401, path
            assert getattr(client, method)(path, headers={"Authorization": "Bearer wrong"}).status_code == 401, path
        assert client.get("/v1/jobs/x", headers={"Authorization": "Bearer secret"}).status_code == 404
    app.state.runtime._stop.set()


def test_unknown_job_404(app_client):
    _, client = app_client
    assert client.get("/v1/jobs/does-not-exist").status_code == 404
    assert client.get("/v1/jobs/does-not-exist/artifacts/memory_input.mp4").status_code == 404


def test_job_fingerprint_dedupe(app_client):
    _, client = app_client
    first = client.post("/v1/jobs", json={**RENDER, "camera_contract": "semantic-camera-v1"}).json()["job_id"]
    second = client.post("/v1/jobs", json={**RENDER, "camera_contract": "semantic-camera-v1"}).json()["job_id"]
    assert first == second
    other = client.post("/v1/jobs", json={**RENDER, "shot_id": "shot_02", "camera_contract": "semantic-camera-v1"})
    assert other.json()["job_id"] != first
    assert client.post("/v1/jobs", json={**RENDER, "type": "recon"}).status_code == 400


def test_render_requires_camera_contract(app_client):
    """A render job without the semantic camera contract runs in an owned worker and fails there."""
    _, client = app_client
    job_id = client.post("/v1/jobs", json=RENDER).json()["job_id"]
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        job = client.get(f"/v1/jobs/{job_id}").json()
        if job["status"] in {"failed", "succeeded"}:
            break
        time.sleep(.25)
    assert job["status"] == "failed", job
    assert "camera_contract=semantic-camera-v1" in job["error"]
