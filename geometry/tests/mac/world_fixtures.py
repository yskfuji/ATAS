"""Reviewed-world candidates in the layout publish_candidate expects (as in upstream tests)."""
import hashlib
from pathlib import Path

import numpy as np

from atas_geometry.service.durable import atomic_json


def candidate(root, job_id, shot_id, *, parent, home, points=3, scene_id="room"):
    payload = dict(run_id="run", story_id="story", scene_id=scene_id, shot_id=shot_id, candidate_job_id=job_id)
    scene = Path(root) / "run/story/scenes" / scene_id
    folder = scene / ".world_versions" / job_id
    l0 = folder / "l0"
    l0.mkdir(parents=True)
    (l0 / "video.mp4").write_bytes(job_id.encode() * 64)
    np.savez(folder / "l1.npz", xyz=np.arange(points * 3, dtype=np.float32).reshape(points, 3))
    atomic_json(l0 / "observation.json", {"registered": True, "scene_id": scene_id, "entity_registry": []})
    video = hashlib.sha256(job_id.encode()).hexdigest()
    data = {**payload, "job_id": job_id, "geometry_accepted": True, "l0": f".world_versions/{job_id}/l0",
            "l1": f".world_versions/{job_id}/l1.npz", "source_sha256": video, "parent_version": parent,
            "home": home, "lineage": {"target_render_job_id": None}, "novel_points": points}
    atomic_json(folder / "candidate.json", data)
    payload["film_review"] = {"schema_version": 1, "stage": "film", "decision": "pass", "cache_key": "review",
                              "video_sha256": video, "lineage": data["lineage"]}
    return payload, scene, l0


def tree_digest(folder):
    """Names and bytes of every file below folder."""
    digest = hashlib.sha256()
    for path in sorted(p for p in Path(folder).rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(folder)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def committed_digest(scene):
    """The world as readers see it: the pointer, every committed version, and what they reference."""
    import json
    scene = Path(scene)
    digest = hashlib.sha256()
    pointer = scene / "world_state.json"
    if not pointer.exists():
        return "empty"
    state = json.loads(pointer.read_text())
    digest.update(pointer.read_bytes())
    # Versions reachable from the pointer; an orphan manifest of an interrupted commit is not visible.
    version = state["version"]
    while version:
        commit = scene / ".world_versions" / version / "commit.json"
        digest.update(str(commit.relative_to(scene)).encode() + commit.read_bytes())
        version = json.loads(commit.read_text()).get("parent_version")
    digest.update((scene / state["l1"]).read_bytes())
    for record in sorted(state["observations"].values(), key=lambda r: r["l0"]):
        digest.update(tree_digest(scene / record["l0"]).encode())
    return digest.hexdigest()
