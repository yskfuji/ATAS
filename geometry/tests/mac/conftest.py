import sys
from pathlib import Path

import pytest

GEOMETRY = Path(__file__).resolve().parents[2]
CACHE = GEOMETRY.parents[1] / "cache"
SAM3CPP_BIN = CACHE / "sam3.cpp" / "atas-build" / "atas_sam3_track"
SAM3CPP_MODEL = CACHE / "weights" / "sam3cpp" / "sam3-f16.ggml"
PI3X_MODEL = CACHE / "weights" / "pi3x"

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="macOS-specific")

OK_PROBE = {"device": "mps", "available": True, "ok": True, "torch": "test",
            "recommended_max_memory": 36 * 2**30, "matmul_max_rel_err": 0.0}


@pytest.fixture
def service_env(tmp_path, monkeypatch):
    """A service configuration whose every dependency exists; the accelerator probe is stubbed."""
    import atas_geometry.service.unified_pool as unified_pool
    model = tmp_path / "models"
    (model / "pi3x").mkdir(parents=True)
    (model / "sam3").mkdir()
    (model / "sam3" / "sam3-f16.ggml").write_bytes(b"x")
    binary = tmp_path / "bin" / "atas_sam3_track"
    binary.parent.mkdir()
    binary.write_text("#!/bin/sh\nexit 0\n")
    binary.chmod(0o755)
    for key in ("GEOMETRY_DEVICE", "GEOMETRY_API_TOKEN", "GEOMETRY_ARCHIVE_ROOT", "GEOMETRY_SAM3_BACKEND"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GEOMETRY_HOT_ROOT", str(tmp_path / "hot"))
    monkeypatch.setenv("GEOMETRY_PI3_MODEL", str(model / "pi3x"))
    monkeypatch.setenv("GEOMETRY_SAM3_CHECKPOINT", str(model / "sam3" / "sam3-f16.ggml"))
    monkeypatch.setenv("GEOMETRY_SAM3CPP_BIN", str(binary))
    monkeypatch.setenv("GEOMETRY_PYTHON", sys.executable)
    monkeypatch.setattr(unified_pool, "probe_accelerator", lambda *a, **k: dict(OK_PROBE))
    return {"root": tmp_path, "model": model, "binary": binary}


@pytest.fixture
def app_client(service_env):
    from fastapi.testclient import TestClient
    from atas_geometry.service.app import create_app
    from atas_geometry.service.config import Settings
    app = create_app(Settings.from_env())
    with TestClient(app) as client:
        yield app, client
    app.state.runtime._stop.set()
