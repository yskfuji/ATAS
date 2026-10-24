"""A rejected or interrupted publication never changes the world readers see."""
import json
import subprocess
import sys

import pytest

from tests.mac.conftest import GEOMETRY
from tests.mac.world_fixtures import candidate, committed_digest


def _published_world(root):
    from atas_geometry.world.reviewed_world import publish_candidate
    first, scene, _ = candidate(root, "recon_a", "shot_a", parent=None, home=True)
    publish_candidate(root, first)
    return scene


@pytest.mark.parametrize("defect", ["review_hold", "review_wrong_video", "lineage", "geometry_rejected", "cross_scope"])
def test_rejected_job_leaves_tree_hash(tmp_path, defect):
    from atas_geometry.world.reviewed_world import publish_candidate
    scene = _published_world(tmp_path)
    before = committed_digest(scene)
    payload, _, _ = candidate(tmp_path, "recon_b", "shot_b", parent="recon_a", home=False)
    if defect == "review_hold":
        payload["film_review"]["decision"] = "hold"
    elif defect == "review_wrong_video":
        payload["film_review"]["video_sha256"] = "0" * 64
    elif defect == "lineage":
        payload["film_review"]["lineage"] = {"target_render_job_id": "other"}
    elif defect == "geometry_rejected":
        path = scene / ".world_versions/recon_b/candidate.json"
        data = json.loads(path.read_text()); data["geometry_accepted"] = False
        path.write_text(json.dumps(data))
    else:
        payload["shot_id"] = "shot_other"
    with pytest.raises(ValueError):
        publish_candidate(tmp_path, payload)
    assert committed_digest(scene) == before


def test_stale_parent_conflict(tmp_path):
    from atas_geometry.world.reviewed_world import publish_candidate
    scene = _published_world(tmp_path)
    second, _, _ = candidate(tmp_path, "recon_b", "shot_b", parent="recon_a", home=False)
    publish_candidate(tmp_path, second)
    before = committed_digest(scene)
    stale, _, _ = candidate(tmp_path, "recon_c", "shot_c", parent="recon_a", home=False)
    with pytest.raises(RuntimeError, match="world changed"):
        publish_candidate(tmp_path, stale)
    assert committed_digest(scene) == before


KILL = r"""
import os, signal, sys
from pathlib import Path
point, root = sys.argv[1], Path(sys.argv[2])
sys.path.insert(0, sys.argv[3])
import atas_geometry.world.world_state as ws
from tests.mac.world_fixtures import candidate
from atas_geometry.world.reviewed_world import publish_candidate
payload, _, _ = candidate(root, "recon_b", "shot_b", parent="recon_a", home=False)
real = ws.atomic_json
def traced(path, value):
    name = Path(path).name
    if point == "before_manifest" and name == "commit.json": os.kill(os.getpid(), signal.SIGKILL)
    if point == "between_manifest_and_pointer" and name == "world_state.json": os.kill(os.getpid(), signal.SIGKILL)
    real(path, value)
    if point == "after_pointer" and name == "world_state.json": os.kill(os.getpid(), signal.SIGKILL)
ws.atomic_json = traced
publish_candidate(root, payload)
"""


@pytest.mark.parametrize("point", ["before_manifest", "between_manifest_and_pointer", "after_pointer"])
def test_sigkill_is_all_or_nothing(tmp_path, point):
    from atas_geometry.world.world_state import load_world
    scene = _published_world(tmp_path)
    old_digest, old_state = committed_digest(scene), load_world(scene)
    result = subprocess.run([sys.executable, "-c", KILL, point, str(tmp_path), str(GEOMETRY)], cwd=GEOMETRY)
    assert result.returncode == -9
    state = load_world(scene)
    if point == "after_pointer":
        assert state["version"] == "recon_b" and state["parent_version"] == "recon_a"
        assert set(state["observations"]) == {"shot_a", "shot_b"}
        committed_digest(scene)  # every referenced artefact exists and is readable
    else:
        assert state == old_state and committed_digest(scene) == old_digest
    # Interrupted atomic writes leave only hidden temporaries, never a partial visible file.
    assert all(p.name.startswith(".") for p in scene.rglob("*.json*") if not p.name.endswith(".json"))
