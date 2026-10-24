import json
import os
import signal
import subprocess
import sys
import time

import pytest

from tests.mac.conftest import darwin_only

pytestmark = darwin_only


def _lease(process):
    from atas_geometry.service import worker_protocol as wp
    return {"pid": process.pid, "process_start": wp.process_start(process.pid), "active": True,
            "owner_contract": "isolated_geometry_session_v1", "pgid": process.pid,
            "session_id": process.pid, "uid": os.getuid(), "boot_id": wp.boot_id()}


def _wait_gone(pid, seconds=5):
    from atas_geometry.service import proc_darwin
    limit = time.monotonic() + seconds
    while time.monotonic() < limit:
        state = proc_darwin._kinfo(pid)
        if state is None or state[0] == proc_darwin.SZOMB:
            return True
        time.sleep(.05)
    return False


def test_live_worker_is_running():
    from atas_geometry.service import worker_protocol as wp
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        time.sleep(.2)
        lease = _lease(process)
        assert wp.owned_group_members(lease) == [process.pid]
        assert wp.lease_running(lease) is True
    finally:
        process.kill(); process.wait()


def test_killed_worker_lease_is_released(tmp_path):
    from atas_geometry.service import worker_protocol as wp
    command = [sys.executable, "-c", "while True: pass", "atas_geometry.service.isolated_worker"]
    with pytest.raises(TimeoutError, match="hard deadline"):
        wp.run_isolated(command, folder=tmp_path, env=os.environ.copy(), deadline_seconds=.3)
    lease = json.loads((tmp_path / "worker_lease.json").read_text())
    assert lease["active"] is False and not lease.get("quarantined")
    assert wp.process_start(lease["pid"]) is None
    assert wp.lease_running(lease) is False
    wp.reconcile_lease(tmp_path)  # a released lease reconciles without error


def test_reused_pid_lease_rejected(monkeypatch):
    from atas_geometry.service import worker_protocol as wp
    pid = os.getpid()
    lease = dict(pid=pid, pgid=pid, session_id=pid, uid=os.getuid(), owner_contract="isolated_geometry_session_v1",
                 boot_id=wp.boot_id(), process_start="1")
    monkeypatch.setattr(os, "killpg", lambda *a: pytest.fail("a reused PID must never be signalled"))
    assert wp.owned_group_members(lease) == [] and wp.stop_owned(lease) is True
    legacy = {"pid": pid, "process_start": "1"}
    assert wp.owned_group_members(legacy) is None and wp.lease_running(legacy) is False


def test_boot_mismatch_returns_empty():
    from atas_geometry.service import worker_protocol as wp
    pid = os.getpid()
    lease = dict(pid=pid, pgid=pid, session_id=pid, uid=os.getuid(), owner_contract="isolated_geometry_session_v1",
                 boot_id="00000000-0000-0000-0000-000000000000", process_start=wp.process_start(pid))
    assert wp.owned_group_members(lease) == []


def test_nested_member_found_and_reaped(tmp_path):
    from atas_geometry.service import worker_protocol as wp
    code = ("import subprocess,sys,time,pathlib;"
            "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']);"
            f"pathlib.Path({str(tmp_path / 'child')!r}).write_text(str(c.pid));time.sleep(60)")
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        limit = time.monotonic() + 10
        while not (tmp_path / "child").exists() and time.monotonic() < limit:
            time.sleep(.05)
        child = int((tmp_path / "child").read_text())
        lease = _lease(process)
        assert sorted(wp.owned_group_members(lease)) == sorted([process.pid, child])
        assert wp.stop_owned(lease) is True
        process.wait(timeout=5)
        assert _wait_gone(child) and wp.lease_running(lease) is False
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL); process.wait()


def test_exiting_process_counts_as_live(monkeypatch):
    """While exiting, proc_pidinfo fails but the process is not yet waitable: it is still a member."""
    from atas_geometry.service import proc_darwin, worker_protocol as wp
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        time.sleep(.2)
        lease = _lease(process)
        monkeypatch.setattr(proc_darwin, "_bsdinfo", lambda pid: None)
        assert wp.owned_group_members(lease) == [process.pid] and wp.lease_running(lease) is True
    finally:
        process.kill(); process.wait()
