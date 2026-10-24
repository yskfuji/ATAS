"""Deterministic synthetic RGB-D clips with exact geometry (a textured room with a moving box).

Each scene is ray-cast analytically in numpy: exact depth, pinhole intrinsics and camera-to-world
poses, plus the dynamic-object mask. Written in the L0 layout the geometry service reads.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "geometry/backbones"))


def look_at(eye, target, up=(0, -1, 0)):
    forward = np.asarray(target, float) - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up); right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, :3] = np.stack([right, down, forward], 1)
    c2w[:3, 3] = eye
    return c2w


def texture(rng, size=64):
    base = rng.integers(40, 220, (size, size, 3)).astype(np.float32)
    return np.kron(base, np.ones((4, 4, 1)))


def render_scene(seed: int, frames: int = 48, h: int = 360, w: int = 640, path_offset: float = 0.0,
                 box_offset: float = 0.0):
    """Room [-2,2]x[-1.5,1.5]x[0,6] (inside faces) plus a box moving along x; camera orbits slowly."""
    rng = np.random.default_rng(seed)
    textures = [texture(rng) for _ in range(6)]
    box_color = np.array([230, 40, 40], np.float32)
    fx = fy = 0.9 * w
    K = np.array([fx, fy, w / 2, h / 2])
    y, x = np.mgrid[:h, :w]
    rays_cam = np.stack(((x + .5 - K[2]) / fx, (y + .5 - K[3]) / fy, np.ones_like(x, float)), -1)
    radius, phase = 0.4 + 0.2 * rng.random(), rng.random() * 2 * np.pi + path_offset
    video, depth, c2ws, masks = [], [], [], []
    for f in range(frames):
        a = phase + 0.6 * f / frames
        eye = np.array([radius * np.cos(a), 0.2 * np.sin(2 * a), 1.0 + 0.5 * f / frames])
        c2w = look_at(eye, np.array([0.0, 0.0, 4.5]))
        R, t = c2w[:3, :3], c2w[:3, 3]
        d = rays_cam @ R.T
        best = np.full((h, w), np.inf)
        color = np.zeros((h, w, 3), np.float32)
        mask = np.zeros((h, w), bool)
        for i, (axis, value) in enumerate([(0, -2), (0, 2), (1, -1.5), (1, 1.5), (2, 6), (2, 0)]):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (value - t[axis]) / d[..., axis]
            p = t + s[..., None] * d
            u_axis, v_axis = [k for k in range(3) if k != axis]
            hit = (s > 1e-6) & (s < best)
            best = np.where(hit, s, best)
            tex = textures[i]
            u = ((p[..., u_axis] * 40) % tex.shape[1]).astype(int)
            v = ((p[..., v_axis] * 40) % tex.shape[0]).astype(int)
            color = np.where(hit[..., None], tex[v.clip(0, tex.shape[0] - 1), u.clip(0, tex.shape[1] - 1)], color)
        centre = np.array([-1.0 + box_offset + 2.0 * f / frames, 0.6, 3.5]); half = np.array([.35, .35, .35])
        with np.errstate(divide="ignore", invalid="ignore"):
            t0 = (centre - half - t) / d; t1 = (centre + half - t) / d
        near = np.nanmax(np.minimum(t0, t1), -1); far = np.nanmin(np.maximum(t0, t1), -1)
        hit = (near <= far) & (near > 1e-6) & (near < best)
        best = np.where(hit, near, best)
        color = np.where(hit[..., None], box_color, color)
        mask |= hit
        video.append(color.clip(0, 255).astype(np.uint8))
        depth.append((best * 1.0).astype(np.float32))  # rays have unit z, so s is camera-space depth
        c2ws.append(c2w.astype(np.float32))
        masks.append(mask)
    return {"video": np.stack(video), "depth": np.stack(depth), "c2w": np.stack(c2ws),
            "intrinsics": np.repeat(K[None].astype(np.float32), frames, 0), "dynamic": np.stack(masks), "fps": 24}


def write_l0(scene: dict, folder: Path) -> Path:
    import imageio.v2 as imageio
    from utils.media import save_cameras, save_depths, save_masks
    folder.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(folder / "video.mp4"), fps=scene["fps"], codec="libx264", quality=None,
                            macro_block_size=1, output_params=["-crf", "0", "-pix_fmt", "yuv444p"]) as writer:
        for frame in scene["video"]:
            writer.append_data(frame)
    save_cameras(str(folder / "cameras.npz"), scene["c2w"], scene["intrinsics"])
    save_depths(str(folder / "depths"), scene["depth"], dtype=np.float32)
    save_masks(str(folder / "dynamic_mask"), scene["dynamic"])
    save_masks(str(folder / "sky_mask"), np.zeros_like(scene["dynamic"]))
    return folder


def write_clip(scene: dict, path: Path) -> Path:
    import imageio.v2 as imageio
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(path), fps=scene["fps"], codec="libx264", quality=None, macro_block_size=1,
                            output_params=["-crf", "12", "-pix_fmt", "yuv420p"]) as writer:
        for frame in scene["video"]:
            writer.append_data(frame)
    return path


if __name__ == "__main__":
    out = Path(sys.argv[1])
    for seed in range(int(sys.argv[2]) if len(sys.argv) > 2 else 6):
        scene = render_scene(seed)
        write_l0(scene, out / f"synthetic_{seed:02d}" / "l0")
        np.savez(out / f"synthetic_{seed:02d}" / "gt.npz", depth=scene["depth"], c2w=scene["c2w"],
                 intrinsics=scene["intrinsics"], dynamic=scene["dynamic"])
        print(json.dumps({"seed": seed, "frames": len(scene["video"])}))
