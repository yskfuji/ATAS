"""design.md world-state invariants, exercised on this Mac's devices."""
import ast
import hashlib
import inspect
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest
import torch

from tests.mac.conftest import GEOMETRY

sys.path.insert(0, str(GEOMETRY / "backbones"))
from accelerator import available  # noqa: E402
from tests.mac.world_fixtures import candidate, tree_digest  # noqa: E402


def test_l0_tree_hash_unchanged_after_second_clip(tmp_path):
    from atas_geometry.world.reviewed_world import publish_candidate
    from atas_geometry.world.world_state import load_world
    first, scene, first_l0 = candidate(tmp_path, "recon_a", "shot_a", parent=None, home=True)
    publish_candidate(tmp_path, first)
    before = tree_digest(first_l0)
    l1_before = (scene / ".world_versions/recon_a/l1.npz").read_bytes()
    second, _, _ = candidate(tmp_path, "recon_b", "shot_b", parent="recon_a", home=False, points=5)
    assert publish_candidate(tmp_path, second)["commit_published"]
    world = load_world(scene)
    assert world["version"] == "recon_b" and world["parent_version"] == "recon_a"
    assert set(world["observations"]) == {"shot_a", "shot_b"}
    assert tree_digest(first_l0) == before
    assert (scene / ".world_versions/recon_a/l1.npz").read_bytes() == l1_before


def _memory(n, seed, shot):
    from atas_geometry.memory.fuse import _memory_from_voxels
    rng = np.random.default_rng(seed)
    return _memory_from_voxels(rng.uniform(-3, 3, (n, 3)).astype(np.float32),
                               rng.integers(0, 255, (n, 3), dtype=np.uint8),
                               voxel_size=.01, origin=np.zeros(3), shot_id=shot)


def test_l1_v1_points_bytewise_subset_of_v2():
    from atas_geometry.register.anchored_fusion import append_static
    old, added = _memory(1000, 1, "old"), _memory(200, 2, "new")
    result = append_static(old, added)
    for name in ("xyz", "rgb", "world_time", "confidence", "foreground", "weight", "point_count",
                 "source_count", "conflict_score", "provenance_input_indices"):
        before = np.ascontiguousarray(getattr(old, name))
        after = np.ascontiguousarray(getattr(result, name))[:len(before)]
        assert after.dtype == before.dtype and after.tobytes() == before.tobytes(), name
    assert np.array_equal(result.provenance_offsets[:len(old.provenance_offsets)], old.provenance_offsets)
    assert result.camera_ids[:len(old.camera_ids)] == old.camera_ids and len(result.xyz) == 1200


def _synthetic_l0(folder, frames=4, h=48, w=64):
    """Grey textured wall at depth 3; a red dynamic box at depth 1 with its SAM mask."""
    from utils.media import save_cameras, save_depths, save_masks
    folder.mkdir(parents=True)
    rng = np.random.default_rng(3)
    wall = rng.integers(60, 200, (h, w, 1), dtype=np.uint8).repeat(3, -1)
    video, depth, mask = [], np.full((frames, h, w), 3.0, np.float32), np.zeros((frames, h, w), bool)
    for f in range(frames):
        image = wall.copy()
        y0, x0 = 10 + f, 20 + 2 * f
        image[y0:y0 + 14, x0:x0 + 16] = (255, 0, 0)
        depth[f, y0:y0 + 14, x0:x0 + 16] = 1.0
        mask[f, y0:y0 + 14, x0:x0 + 16] = True
        video.append(image)
    with imageio.get_writer(str(folder / "video.mp4"), fps=10, codec="libx264", quality=None,
                            macro_block_size=1, output_params=["-crf", "0", "-pix_fmt", "yuv444p"]) as writer:
        for image in video:
            writer.append_data(image)
    c2w = np.repeat(np.eye(4, dtype=np.float32)[None], frames, 0)
    c2w[:, 0, 3] = np.linspace(0, .05, frames)
    intrinsics = np.repeat(np.array([[60, 60, w / 2, h / 2]], np.float32), frames, 0)
    save_cameras(str(folder / "cameras.npz"), c2w, intrinsics)
    save_depths(str(folder / "depths"), depth, dtype=np.float32)
    save_masks(str(folder / "dynamic_mask"), mask)
    save_masks(str(folder / "sky_mask"), np.zeros_like(mask))
    return folder


@pytest.mark.parametrize("device_name", ["cpu", "mps"])
def test_l2_points_absent_from_l1(tmp_path, device_name):
    if not available(device_name):
        pytest.skip(f"{device_name} unavailable")
    from atas_geometry.memory.fuse import _accumulate_home_voxels
    from atas_geometry.memory.vista4d_layers import ReconStore
    l0 = _synthetic_l0(tmp_path / "l0")
    recon = ReconStore(l0, duration=1.0, exclude_sky=True)
    xyz, rgb, _, _ = _accumulate_home_voxels(recon, voxel=.02, frames=list(range(recon.count)),
                                             device=torch.device(device_name))
    assert len(xyz) > 100
    assert float(xyz[:, 2].min()) > 2.5, "a dynamic (depth 1) point entered L1"
    red = (rgb[:, 0] > 200) & (rgb[:, 1] < 60) & (rgb[:, 2] < 60)
    assert not red.any(), "dynamic colour entered L1"
    if device_name == "mps":
        cpu_xyz, cpu_rgb, _, _ = _accumulate_home_voxels(recon, voxel=.02, frames=list(range(recon.count)),
                                                         device=torch.device("cpu"))
        assert np.array_equal(cpu_xyz, xyz) and np.array_equal(cpu_rgb, rgb)


def test_decision_constants():
    from atas_geometry.reconstruct import initial_world
    from atas_geometry.register import anchored_fusion, registration_recovery
    assert (initial_world.GROSS, initial_world.MIN_PAIRS, initial_world.MIN_COVERAGE) == (4, 3, .8)
    assert "tolerance = .02*float(np.median(positive))" in inspect.getsource(initial_world.temporal_evidence)
    source = inspect.getsource(anchored_fusion.validate_cross_shot)
    assert '"accepted": bool(n >= 30 and pairs >= 2' in source
    assert registration_recovery.MAX_ATTEMPTS == 2
    tree = ast.parse(inspect.getsource(anchored_fusion))
    assert any(isinstance(node, ast.Compare) and any(isinstance(c, ast.Constant) and c.value == 30
               for c in node.comparators) for node in ast.walk(tree))
