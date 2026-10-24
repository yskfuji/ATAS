"""Deterministic evaluation clips with ground truth, read straight from the official archives.

    python tools/datasets.py <out-dir>

Rules (preregistered, no tuning):
* TUM RGB-D (Sturm et al. 2012): one clip per sequence: 48 RGB frames, stride 2 (96 frames at
  30 Hz), centred in the sequence. Depth (factor 5000) and the motion-capture pose are associated
  to each RGB timestamp, nearest within 20 ms. Intrinsics are the published per-camera values.
* MPI Sintel (Butler et al. 2012), final pass: one clip per training sequence: the central
  min(48, length) frames. Depth (.dpt) and camera (.cam: K and world-to-camera [R|t]).
Each clip: clip.mp4 (CFR, lossless-ish), gt.npz (depth, c2w, intrinsics, valid), meta.json.
"""
from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

TUM_K = {"freiburg1": (517.3, 516.5, 318.6, 255.3), "freiburg2": (520.9, 521.0, 325.1, 249.7),
         "freiburg3": (535.4, 539.2, 320.1, 247.6)}
CLIP_FRAMES = 48


def quaternion_c2w(t, q):
    x, y, z, w = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    c2w = np.eye(4); c2w[:3, :3] = R; c2w[:3, 3] = t
    return c2w


def write_clip(frames, fps, path):
    import imageio.v2 as imageio
    with imageio.get_writer(str(path), fps=fps, codec="libx264", quality=None, macro_block_size=1,
                            output_params=["-crf", "10", "-pix_fmt", "yuv420p"]) as writer:
        for frame in frames:
            writer.append_data(frame)


def _listing(lines):
    rows = []
    for line in lines:
        if line.strip() and not line.startswith("#"):
            rows.append(line.split())
    return rows


def prepare_tum(archive: Path, out: Path) -> dict:
    name = archive.name.removesuffix(".tgz")
    camera = name.split("_")[2]
    members, texts = {}, {}
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            if member.isfile():
                rel = member.name.split("/", 1)[1]
                if rel in {"rgb.txt", "depth.txt", "groundtruth.txt"}:
                    texts[rel] = tar.extractfile(member).read().decode().splitlines()
                elif rel.startswith(("rgb/", "depth/")):
                    members[rel] = member.name
        rgb = _listing(texts["rgb.txt"]); depth = _listing(texts["depth.txt"]); gt = _listing(texts["groundtruth.txt"])
        dt = np.array([float(r[0]) for r in depth]); gtt = np.array([float(r[0]) for r in gt])
        usable = [r for r in rgb if np.min(np.abs(dt - float(r[0]))) <= .02 and np.min(np.abs(gtt - float(r[0]))) <= .02]
        span = 2 * CLIP_FRAMES
        start = max(0, (len(usable) - span) // 2)
        windows = {"": usable[start:start + span:2][:CLIP_FRAMES]}
        if start - span >= 0:  # the history window immediately before (first observation of the same scene)
            windows["_history"] = usable[start - span:start:2][:CLIP_FRAMES]
        wanted = {}
        for r in [r for chosen in windows.values() for r in chosen]:
            stamp = float(r[0])
            d = depth[int(np.argmin(np.abs(dt - stamp)))][1]
            g = gt[int(np.argmin(np.abs(gtt - stamp)))]
            wanted[r[1]] = ("rgb", r); wanted[d] = ("depth", r[1])
        data = {}
        with tarfile.open(archive, "r:gz") as tar2:
            for member in tar2:
                rel = member.name.split("/", 1)[1] if "/" in member.name else ""
                if rel in wanted:
                    data[rel] = tar2.extractfile(member).read()
    fx, fy, cx, cy = TUM_K[camera]
    saved = None
    for suffix, chosen in windows.items():
        frames, depths, c2ws = [], [], []
        for r in chosen:
            stamp = float(r[0])
            frames.append(np.asarray(Image.open(io.BytesIO(data[r[1]])).convert("RGB")))
            d = depth[int(np.argmin(np.abs(dt - stamp)))][1]
            depths.append(np.asarray(Image.open(io.BytesIO(data[d])), np.float32) / 5000.0)
            g = gt[int(np.argmin(np.abs(gtt - stamp)))]
            c2ws.append(quaternion_c2w(np.array(g[1:4], float), np.array(g[4:8], float)))
        row = _save(out / f"tum_{name.removeprefix('rgbd_dataset_')}{suffix}", frames, 15, np.stack(depths), np.stack(c2ws),
                    np.tile([fx, fy, cx, cy], (len(frames), 1)),
                    {"dataset": "TUM RGB-D", "sequence": name, "window": suffix or "centre",
                     "rule": "centre 96 frames, stride 2, 48 frames" if not suffix else "the 96 frames before the centre window, stride 2",
                     "rgb": [r[1] for r in chosen], "timestamps": [float(r[0]) for r in chosen]})
        saved = saved or row
    return saved


def _read_dpt(raw: bytes):
    w, h = np.frombuffer(raw[4:12], np.int32)
    return np.frombuffer(raw[12:12 + 4 * w * h], np.float32).reshape(h, w)


def _read_cam(raw: bytes):
    values = np.frombuffer(raw[4:4 + 8 * 21], np.float64)
    K, w2c = values[:9].reshape(3, 3), np.vstack([values[9:21].reshape(3, 4), [0, 0, 0, 1]])
    return K, np.linalg.inv(w2c)


def prepare_sintel(images: Path, depth_zip: Path, out: Path) -> list[dict]:
    rows = []
    with zipfile.ZipFile(images) as zi, zipfile.ZipFile(depth_zip) as zd:
        names = sorted({n.split("/")[2] for n in zi.namelist() if n.startswith("training/final/") and n.count("/") == 3 and n.endswith(".png")})
        for seq in names:
            frames_all = sorted(n for n in zi.namelist() if n.startswith(f"training/final/{seq}/") and n.endswith(".png"))
            count = min(CLIP_FRAMES, len(frames_all))
            start = (len(frames_all) - count) // 2
            chosen = frames_all[start:start + count]
            frames, depths, c2ws, ks = [], [], [], []
            for n in chosen:
                stem = Path(n).stem
                frames.append(np.asarray(Image.open(io.BytesIO(zi.read(n))).convert("RGB")))
                depths.append(_read_dpt(zd.read(f"training/depth/{seq}/{stem}.dpt")))
                K, c2w = _read_cam(zd.read(f"training/camdata_left/{seq}/{stem}.cam"))
                c2ws.append(c2w); ks.append([K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
            rows.append(_save(out / f"sintel_{seq}", frames, 24, np.stack(depths), np.stack(c2ws), np.array(ks),
                              {"dataset": "MPI Sintel (final)", "sequence": seq, "rule": "central min(48, length) frames",
                               "frames": [Path(n).name for n in chosen]}))
    return rows


def _save(folder: Path, frames, fps, depth, c2w, intrinsics, meta) -> dict:
    folder.mkdir(parents=True, exist_ok=True)
    write_clip(frames, fps, folder / "clip.mp4")
    valid = np.isfinite(depth) & (depth > 0)
    np.savez_compressed(folder / "gt.npz", depth=depth.astype(np.float32), c2w=c2w.astype(np.float64),
                        intrinsics=np.asarray(intrinsics, np.float64), valid=valid)
    meta = {**meta, "clip_frames": len(frames), "fps": fps, "height": frames[0].shape[0], "width": frames[0].shape[1]}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2))
    return {"clip": folder.name, **{k: meta[k] for k in ("dataset", "sequence", "clip_frames")}}


def main(out: Path):
    archives = Path(__file__).resolve().parents[2] / "cache/datasets/archives"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for tgz in sorted(archives.glob("rgbd_dataset_*.tgz")):
        rows.append(prepare_tum(tgz, out)); print(json.dumps(rows[-1]), flush=True)
    images, depth = archives / "MPI-Sintel-training_images.zip", archives / "MPI-Sintel-depth-training-20150305.zip"
    if images.is_file() and depth.is_file():
        rows += prepare_sintel(images, depth, out)
    (out / "clips.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]))
