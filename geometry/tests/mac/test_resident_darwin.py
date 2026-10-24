import os
import subprocess
import sys

from tests.mac.conftest import GEOMETRY, darwin_only

pytestmark = darwin_only
DARWIN_SUN_PATH = 104  # sizeof(sockaddr_un.sun_path) on macOS, including the terminating NUL


def test_socket_path_fits_sun_path(monkeypatch):
    from atas_geometry.service.resident_models import location
    monkeypatch.delenv("GEOMETRY_DEVICE", raising=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    root, endpoint = location()
    assert endpoint.name == "gpumps.sock" and len(os.fsencode(str(endpoint))) < DARWIN_SUN_PATH
    assert (root.stat().st_mode & 0o077) == 0 and root.stat().st_uid == os.getuid()


def test_af_unix_roundtrip_real_server(tmp_path, monkeypatch):
    """A real resident server is spawned by the client and answers one task over AF_UNIX.

    The task is an allowed module invoked with --help: its SystemExit travels back through
    the protocol as a structured error, after the server has redirected the task's output."""
    from atas_geometry.service.resident_models import location, resident_command
    monkeypatch.setenv("GEOMETRY_RESIDENT_MODELS", "1")
    monkeypatch.delenv("GEOMETRY_DEVICE", raising=False)
    log = tmp_path / "task.log"
    log.write_text("")
    backbones = GEOMETRY / "backbones"
    command = resident_command([sys.executable, "-m", "atas_geometry.register.anchored_fusion", "--help"],
                               repo=GEOMETRY, vista_repo=backbones, log_path=log)
    assert command[1].endswith("resident_models.py") and command[2] == "client"
    env = {**os.environ, "PYTHONPATH": f"{GEOMETRY}{os.pathsep}{backbones}"}
    completed = subprocess.run(command, cwd=GEOMETRY, env=env, capture_output=True, text=True, timeout=300)
    assert completed.returncode != 0 and "SystemExit: 0" in completed.stderr, completed.stderr[-3000:]
    text = log.read_text()
    assert "ATAS_RESIDENT_BEGIN" in text and "--reference-l0" in text
    from atas_geometry.service.resident_models import connect
    _, endpoint = location()
    # A failed task makes the server exit (state is never reused); its socket no longer accepts.
    assert connect(endpoint) is None
