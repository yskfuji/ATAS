import fcntl
import json
import os

from atas_geometry.service import durable
from tests.mac.conftest import darwin_only

pytestmark = darwin_only


def _trace(monkeypatch, tmp_path, fullfsync_ok=True):
    events = []
    real_fcntl, real_replace, real_fsync = fcntl.fcntl, os.replace, os.fsync
    def _path(fd):
        return real_fcntl(fd, fcntl.F_GETPATH, bytes(1024)).rstrip(b"\0").decode()
    def fake_fcntl(fd, cmd, *args):
        if cmd == fcntl.F_FULLFSYNC:
            events.append(("fullfsync", "dir" if os.path.isdir(_path(fd)) else "file"))
            if not fullfsync_ok:
                raise OSError(45, "ENOTSUP")
            return 0
        return real_fcntl(fd, cmd, *args)
    monkeypatch.setattr(fcntl, "fcntl", fake_fcntl)
    monkeypatch.setattr(os, "replace", lambda a, b: (events.append(("replace",)), real_replace(a, b))[1])
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append(("fsync", "dir" if os.path.isdir(_path(fd)) else "file")), real_fsync(fd))[1])
    return events


def test_atomic_json_uses_fullfsync_order(monkeypatch, tmp_path):
    events = _trace(monkeypatch, tmp_path)
    durable.atomic_json(tmp_path / "state.json", {"a": 1})
    assert events == [("fullfsync", "file"), ("replace",), ("fullfsync", "dir")]
    assert json.loads((tmp_path / "state.json").read_text()) == {"a": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["state.json"]


def test_directory_synced_after_rename(monkeypatch, tmp_path):
    events = _trace(monkeypatch, tmp_path, fullfsync_ok=False)
    durable.atomic_json(tmp_path / "state.json", {"a": 2})
    # F_FULLFSYNC refused: each step falls back to fsync, and the directory is still synced last.
    assert events == [("fullfsync", "file"), ("fsync", "file"), ("replace",), ("fullfsync", "dir"), ("fsync", "dir")]


def test_observation_json_is_atomic(monkeypatch, tmp_path):
    from atas_geometry.world import recapture
    writes = []
    real = durable.atomic_text
    monkeypatch.setattr(durable, "atomic_text", lambda path, text: (writes.append(path), real(path, text))[1])
    l0 = tmp_path / "l0"
    l0.mkdir()
    recapture.record_observation(l0, {"run_id": "r", "story_id": "s", "shot_id": "a", "scene_id": "x",
                                      "event_id": "e", "duration": 2.0}, {"transform": None})
    assert writes == [l0 / "observation.json"]
    body = (l0 / "observation.json").read_text()
    assert body.endswith("}\n") and json.loads(body)["scene_id"] == "x"
    assert not list(l0.glob("*.tmp")) and not list(l0.glob(".observation.json*"))
