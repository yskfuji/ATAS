"""The static guards must detect the changes they exist to forbid (falsification of the guard)."""
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tools"))
import guards  # noqa: E402

AF = "geometry/atas_geometry/register/anchored_fusion.py"
IW = "geometry/atas_geometry/reconstruct/initial_world.py"
WB = "geometry/atas_geometry/world/writeback.py"


def test_current_fork_passes():
    for path in (AF, IW, WB):
        assert guards.ast_identical_except_device_calls(path)["identical"], path
    assert guards.ast_module("geometry/atas_geometry/service/worker_protocol.py", allow_darwin_block=True)["identical"]


@pytest.mark.parametrize("path,old,new", [
    (AF, '"accepted": bool(n >= 30 and pairs >= 2', '"accepted": bool(n >= 29 and pairs >= 2'),
    (AF, "anchor_tolerance = .02 * scene_depth", "anchor_tolerance = .03 * scene_depth"),
    (AF, "accel.compute_dtype()", "torch.float16"),
    (AF, "init_pi3(model_id, device=accel.backend())", "init_pi3(model_id, device=accel.backend(), dtype=None)"),
    (IW, "MIN_PAIRS = 3", "MIN_PAIRS = 2"),
    (IW, "tolerance = .02*float(np.median(positive))", "tolerance = .04*float(np.median(positive))"),
    (WB, "**accel.worker_env(gpu_index),", "**accel.worker_env(gpu_index), GEOMETRY_EXTRA='1',"),
])
def test_decision_change_is_detected(path, old, new):
    text = (REPO / path).read_text()
    assert text.count(old) == 1, old
    assert not guards.ast_identical_except_device_calls(path, text.replace(old, new))["identical"]


def test_new_accel_name_and_rebinding_are_detected():
    text = (REPO / IW).read_text().replace("import numpy as np", "import numpy as np\nfrom atas_geometry import device as accel\nX = accel.anything()", 1)
    result = guards.ast_identical_except_device_calls(IW, text)
    assert not result["identical"] and any("anything" in v for v in result["violations"])
    rebound = (REPO / IW).read_text() + "\nassess = lambda *a, **k: {'sufficient': True, 'drift': False}\n"
    assert not guards.ast_identical_except_device_calls(IW, rebound)["identical"]


def test_worker_protocol_second_darwin_block_or_linux_edit_detected():
    path = "geometry/atas_geometry/service/worker_protocol.py"
    text = (REPO / path).read_text()
    assert not guards.ast_module(path, True, text + '\nif sys.platform == "darwin":\n    boot_id = lambda: "x"\n')["identical"]
    assert not guards.ast_module(path, True, text.replace('return Path("/proc/sys/kernel/random/boot_id")', 'return Path("/proc/sys/kernel/random/bootid")'))["identical"]
