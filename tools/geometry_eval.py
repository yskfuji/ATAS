"""Production reconstructions on CPU and MPS, and the AC12/AC13 metrics computed from them.

    python tools/geometry_eval.py recon  <clip-dir> <out-dir> --device cpu|mps --dtype float32|bfloat16|float16
    python tools/geometry_eval.py masks  <clip-dir> <out-dir>       # sam3.cpp person/sky masks (every 4th frame)
    python tools/geometry_eval.py compare <clip-dir> <out-dir> --dtype float32

Reconstruction runs the service's own recon script (scripts.preprocess.recon_and_seg_single) with the
service's arguments and no segmentation keywords: the device is the only difference between runs.
Depth is compared as the stored production artefact (float16 EXR), whose rounding is a floor of
about 4.9e-4 relative per pixel on any comparison.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
PYTHON = REPO / ".venv-macos/bin/python"
BACKBONES = REPO / "geometry/backbones"
sys.path[:0] = [str(REPO / "geometry"), str(BACKBONES)]
SETTINGS = {"recon_wh": [1280, 720], "requested": {"pi3_pixel_limit": 255000, "pi3_vo_chunk": 48},
            "effective": {"pi3_pixel_limit": 160000, "pi3_vo_chunk": 24, "pi3_vo_overlap": 6}, "low_vram": True}
MASK_STRIDE = 4
# Video-depth benchmark protocol: ground truth capped at 70 and aligned predictions clipped at 70, as in
# MonST3R's evaluation (depth_metric.ipynb: depth_evaluation(..., max_depth=70) for Sintel and Bonn).
MAX_DEPTH = 70.0
CRN_SEEDS = (20261019, 20261020, 20261021)   # A5: device comparisons at three common-random-number seeds
NATIVE_RUNS = 5                              # upstream noise floor: native-RNG CPU repeats (informative)


def run_name(device, dtype, mode, k, threads=None):
    return f"{device}_{dtype}_{mode}{k}" + (f"_t{threads}" if threads else "")


def recon(clip: Path, out: Path, device: str, dtype: str, mode: str = "crn", k: int = 0, threads: int | None = None) -> dict:
    from atas_geometry.memory.timebase import probe
    frames = probe(clip / "clip.mp4")["frames"]
    folder = out / clip.name / run_name(device, dtype, mode, k, threads)
    done = folder / "run.json"
    if device == "cpu" and done.is_file() and json.loads(done.read_text())["exit_code"] == 0 and (folder / "l0" / "cameras.npz").is_file():
        return {**json.loads(done.read_text()), "reused": True}  # the CPU path is unaffected by MPS-only fixes
    folder.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GEOMETRY_DEVICE": device, "GEOMETRY_MPS_DTYPE": dtype,
           "PYTHONPATH": f"{BACKBONES}{os.pathsep}{REPO / 'geometry'}", "GEOMETRY_EVAL_CRN_LOG": str(folder / "crn.json")}
    if threads:  # A7 summation-order floor: same arithmetic, different legitimate reduction order
        env.update(OMP_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS=str(threads))
    if mode == "crn":
        env["GEOMETRY_EVAL_CRN_SEED"] = str(CRN_SEEDS[k])
        env.pop("GEOMETRY_EVAL_NATIVE_RNG", None)
    else:
        env["GEOMETRY_EVAL_NATIVE_RNG"] = "1"
    from atas_geometry import device as accel
    if device == "mps":
        os.environ["GEOMETRY_DEVICE"] = "mps"
        env.update(accel.worker_env(0))
    # Common random numbers: upstream Pi3X draws a random depth-scale augmentation at inference (see crn_launch).
    command = [str(PYTHON), str(REPO / "tools/crn_launch.py"), "scripts.preprocess.recon_and_seg_single", "--video_path", str(clip / "clip.mp4"),
               "--output_folder", str(folder / "l0"), "--seg_keywords", "--recon_method", "pi3",
               "--pi3_model_id", str(CACHE / "weights/pi3x"), "--height", "720", "--width", "1280",
               "--num_frames", str(frames), "--pi3_pixel_limit", "255000", "--pi3_vo_chunk_size", "48",
               "--pi3_vo_overlap", "6", "--low_vram"]
    tick = time.perf_counter()
    with (folder / "recon.log").open("w") as log:
        completed = subprocess.run(command, cwd=BACKBONES, env=env, stdout=log, stderr=subprocess.STDOUT)
    phases = [json.loads(line.split(" ", 1)[1]) for line in (folder / "recon.log").read_text().splitlines()
              if line.startswith("AVWM_RECON_PHASE ")]
    result = {"clip": clip.name, "device": device, "dtype": dtype, "exit_code": completed.returncode,
              "random_numbers": "common" if mode == "crn" else "native", "seed_index": k,
              "crn": json.loads((folder / "crn.json").read_text()) if (folder / "crn.json").is_file() else None,
              "wall_s": time.perf_counter() - tick, "frames": frames, "phases": phases,
              "peak_footprint_mib": max((p.get("peak_footprint_mib", 0) for p in phases), default=None)}
    (folder / "run.json").write_text(json.dumps(result, indent=2))
    return result


def masks(clip: Path, out: Path) -> dict:
    """Person (dynamic) and sky masks on every 4th frame at the reconstruction resolution."""
    if (out / clip.name / "masks.npz").is_file():
        return {"clip": clip.name, "reused": True}
    from utils.media import crop_and_resize_video, load_video
    from utils.recon_and_seg.seg_sam3_cpp import init_sam3_video, run_sam3_video
    video, _ = load_video(str(clip / "clip.mp4"))
    video = crop_and_resize_video(video, 720, 1280, resample="lanczos")[::MASK_STRIDE]
    predictor = init_sam3_video(str(CACHE / "sam3.cpp/atas-build/atas_sam3_track"),
                                str(CACHE / "weights/sam3cpp/sam3-f16.ggml"))
    report = {}
    dynamic, _ = run_sam3_video(video, predictor, ["person"], collect_instances=False, report=report)
    sky, _ = run_sam3_video(video, predictor, ["sky"], collect_instances=False, report=report)
    folder = out / clip.name
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / "masks.npz", dynamic=dynamic, sky=sky, stride=MASK_STRIDE)
    return {"clip": clip.name, "frames": len(video), "dynamic_frac": float(dynamic.mean()), "sky_frac": float(sky.mean())}


def load_l0(folder: Path):
    from atas_geometry.memory.vista4d_layers import _read_exr_depth
    cameras = np.load(folder / "cameras.npz")
    depth = np.stack([_read_exr_depth(p) for p in sorted((folder / "depths").glob("*.exr"))]).astype(np.float64)
    return depth, cameras["cam_c2w"].astype(np.float64), cameras["intrinsics"].astype(np.float64)


def static_mask(masks_npz, index, shape):
    import cv2
    data = np.load(masks_npz)
    k = index // int(data["stride"])
    dynamic = cv2.dilate(data["dynamic"][k].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)  # DEFAULT_MASK_DILATE=2
    return ~(dynamic | data["sky"][k])


def rotation_deg_pre_A12(a, b):
    """The original formula, kept for side-by-side reporting only: arccos is ill-conditioned near 0, so for float32-rounded
    matrices it measures rounding of the matrix entries (~0.02-0.1 deg), not the rotation difference (A12)."""
    cos = (np.trace(a[:3, :3].T @ b[:3, :3]) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def rotation_deg(a, b):
    """Angle of Ra^T Rb in degrees, numerically stable (A12): atan2(|vee(R - R^T)| / 2, (trace R - 1) / 2) in float64."""
    R = a[:3, :3].astype(np.float64).T @ b[:3, :3].astype(np.float64)
    skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.degrees(np.arctan2(np.linalg.norm(skew) / 2.0, (np.trace(R) - 1) / 2.0)))


def unproject(depth, K, c2w, mask, step=4):
    ys, xs = np.nonzero(mask[::step, ::step]); ys, xs = ys * step, xs * step
    z = depth[ys, xs]
    keep = np.isfinite(z) & (z > 1e-5)
    ys, xs, z = ys[keep], xs[keep], z[keep]
    cam = np.stack(((xs + .5 - K[2]) / K[0] * z, (ys + .5 - K[3]) / K[1] * z, z), 1)
    return cam @ c2w[:3, :3].T + c2w[:3, 3]


def parity(ref: Path, test: Path, masks_npz: Path) -> dict:
    """MPS (test) vs CPU-fp32 (ref) on the same clip; raw units; per-clip maximum over frames."""
    from scipy.spatial import cKDTree
    d_ref, c_ref, k_ref = load_l0(ref)
    d_test, c_test, k_test = load_l0(test)
    absrel, rot, rot_old, trans = [], [], [], []
    static_depths, static_ref, static_test = [], [], []
    for i in range(len(d_ref)):
        valid = np.isfinite(d_ref[i]) & (d_ref[i] > 1e-5) & np.isfinite(d_test[i]) & (d_test[i] > 1e-5)
        absrel.append(float(np.mean(np.abs(d_test[i][valid] - d_ref[i][valid]) / d_ref[i][valid])))
        rot.append(rotation_deg(c_ref[i], c_test[i]))
        rot_old.append(rotation_deg_pre_A12(c_ref[i], c_test[i]))
        trans.append(float(np.linalg.norm(c_ref[i][:3, 3] - c_test[i][:3, 3])))
        if i % MASK_STRIDE == 0:
            mask = static_mask(masks_npz, i, d_ref[i].shape) & valid
            static_depths.append(d_ref[i][mask])
            static_ref.append(unproject(d_ref[i], k_ref[i], c_ref[i], mask))
            static_test.append(unproject(d_test[i], k_test[i], c_test[i], mask))
    median_depth = float(np.median(np.concatenate(static_depths)))
    pr, pt = np.concatenate(static_ref), np.concatenate(static_test)
    extent = float(np.linalg.norm(np.percentile(pr, 98, axis=0) - np.percentile(pr, 2, axis=0)))
    chamfer = float((cKDTree(pt).query(pr)[0].mean() + cKDTree(pr).query(pt)[0].mean()) / 2)
    return {"depth_absrel": max(absrel), "rot_deg": max(rot), "rot_deg_pre_A12": max(rot_old), "trans_frac_depth": max(trans) / median_depth,
            "chamfer_frac_extent": chamfer / extent, "median_static_depth": median_depth, "robust_extent": extent,
            "per_frame_depth_absrel": absrel}


def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, S, Vt = np.linalg.svd(cov)
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    scale = np.trace(np.diag(S) @ D) / max((xs ** 2).sum() / len(src), 1e-12)
    return scale, R, mu_d - scale * R @ mu_s


def accuracy(test: Path, clip: Path) -> dict:
    """Against ground truth: median-scaled depth AbsRel and delta<1.25; ATE/RPE after Sim(3) (Umeyama)."""
    from utils.media import resize_2d
    depth, c2w, _ = load_l0(test)
    gt = np.load(clip / "gt.npz")
    gt_depth = resize_2d(gt["depth"].astype(np.float32), 720, 1280, mode="nearest", crop=True).astype(np.float64)
    valid = np.isfinite(gt_depth) & (gt_depth > 0) & (gt_depth < MAX_DEPTH) & np.isfinite(depth) & (depth > 0)
    scale = np.median(gt_depth[valid]) / np.median(depth[valid])
    ratio = np.minimum(scale * depth[valid], MAX_DEPTH) / gt_depth[valid]
    absrel = float(np.mean(np.abs(ratio - 1)))
    delta = float(np.mean(np.maximum(ratio, 1 / ratio) < 1.25))
    gt_c2w = np.linalg.inv(gt["c2w"][0])[None] @ gt["c2w"]  # same first-frame gauge as the reconstruction
    s, R, t = umeyama(c2w[:, :3, 3], gt_c2w[:, :3, 3])
    aligned = (s * (R @ c2w[:, :3, 3].T)).T + t
    ate = float(np.sqrt(np.mean(np.sum((aligned - gt_c2w[:, :3, 3]) ** 2, 1))))
    rel_pred = np.linalg.norm(np.diff(aligned, axis=0), axis=1)
    rel_gt = np.linalg.norm(np.diff(gt_c2w[:, :3, 3], axis=0), axis=1)
    steps = [np.linalg.norm((s * R @ (np.linalg.inv(c2w[i])[:3, :3] @ (c2w[i + 1][:3, 3] - c2w[i][:3, 3])))
                            - (np.linalg.inv(gt_c2w[i])[:3, :3] @ (gt_c2w[i + 1][:3, 3] - gt_c2w[i][:3, 3])))
             for i in range(len(c2w) - 1)]
    return {"depth_absrel": absrel, "delta_1.25": delta, "ate": ate, "rpe": float(np.mean(steps)), "max_depth": MAX_DEPTH,
            "valid_fraction": float(valid.mean()),
            "path_length_gt": float(rel_gt.sum()), "path_length_pred_aligned": float(rel_pred.sum())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["recon", "masks", "compare"])
    parser.add_argument("clip", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--mode", default="crn", choices=["crn", "native"])
    parser.add_argument("--seed-index", type=int, default=0)
    parser.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    if args.action == "recon":
        print(json.dumps(recon(args.clip, args.out, args.device, args.dtype, args.mode, args.seed_index, args.threads)))
    elif args.action == "masks":
        print(json.dumps(masks(args.clip, args.out)))
    else:
        compare(args.clip, args.out, args.dtype)


def compare(clip: Path, out: Path, dtype: str) -> dict:
    """AC12 per clip = max over CRN seeds; AC13 accuracy per clip = mean over seeds; draw sequences must match."""
    base = out / clip.name
    seeds = [k for k in range(len(CRN_SEEDS)) if (base / run_name("cpu", "float32", "crn", k) / "l0" / "cameras.npz").is_file()]
    result = {"clip": clip.name, "dtype": dtype, "settings": SETTINGS, "random_numbers": "common", "seeds": seeds,
              "accuracy_cpu_by_seed": [accuracy(base / run_name("cpu", "float32", "crn", k) / "l0", clip) for k in seeds]}
    result["accuracy_cpu"] = {m: float(np.mean([a[m] for a in result["accuracy_cpu_by_seed"]])) for m in result["accuracy_cpu_by_seed"][0]}
    tested = [k for k in seeds if (base / run_name("mps", dtype, "crn", k) / "l0" / "cameras.npz").is_file()]
    if tested:
        result["accuracy_mps_by_seed"] = [accuracy(base / run_name("mps", dtype, "crn", k) / "l0", clip) for k in tested]
        result["accuracy_mps"] = {m: float(np.mean([a[m] for a in result["accuracy_mps_by_seed"]])) for m in result["accuracy_mps_by_seed"][0]}
        per_seed = [parity(base / run_name("cpu", "float32", "crn", k) / "l0", base / run_name("mps", dtype, "crn", k) / "l0",
                           base / "masks.npz") for k in tested]
        keys = ("depth_absrel", "rot_deg", "trans_frac_depth", "chamfer_frac_extent")
        result["parity"] = {key: max(p[key] for p in per_seed) for key in keys}
        result["parity_by_seed"] = [{key: p[key] for key in keys} for p in per_seed]
        result["parity_pre_A12"] = {"rot_deg": max(p["rot_deg_pre_A12"] for p in per_seed)}
        logs = lambda d, k: json.loads((base / run_name(d, "float32" if d == "cpu" else dtype, "crn", k) / "crn.json").read_text())
        result["crn_installed_all"] = all(logs(d, k)["crn_installed"] for d in ("cpu", "mps") for k in tested)
        result["crn_draw_sequences_equal"] = all(logs("cpu", k)["draw_sequence_sha256"] == logs("mps", k)["draw_sequence_sha256"] for k in tested)
    quantities = ("depth_absrel", "rot_deg", "trans_frac_depth", "chamfer_frac_extent")
    natives = [k for k in range(NATIVE_RUNS) if (base / run_name("cpu", "float32", "native", k) / "l0" / "cameras.npz").is_file()]
    if len(natives) >= 2:
        floor = [parity(base / run_name("cpu", "float32", "native", 0) / "l0", base / run_name("cpu", "float32", "native", k) / "l0",
                        base / "masks.npz") for k in natives[1:]]
        result["native_noise_floor_cpu_vs_cpu"] = {key: max(f[key] for f in floor) for key in quantities}
        result["native_noise_floor_median"] = {key: float(np.median([f[key] for f in floor])) for key in quantities}
        result["native_noise_floor_cpu_vs_cpu_pre_A12"] = {"rot_deg": max(f["rot_deg_pre_A12"] for f in floor)}
        result["native_noise_floor_median_pre_A12"] = {"rot_deg": float(np.median([f["rot_deg_pre_A12"] for f in floor]))}
        result["native_repeats"] = len(natives)
    orders = [parity(base / run_name("cpu", "float32", "crn", k) / "l0", base / run_name("cpu", "float32", "crn", k, threads=4) / "l0",
                     base / "masks.npz") for k in range(len(CRN_SEEDS))
              if (base / run_name("cpu", "float32", "crn", k, threads=4) / "l0" / "cameras.npz").is_file()]
    if orders:  # A7: median over CRN seeds of CPU 10-vs-4-thread deviation
        result["order_floor_cpu_10_vs_4_threads"] = {key: float(np.median([o[key] for o in orders])) for key in quantities}
        result["order_floor_seeds"] = len(orders)
        result["order_floor_cpu_10_vs_4_threads_pre_A12"] = {"rot_deg": float(np.median([o["rot_deg_pre_A12"] for o in orders]))}
    (base / f"compare_{dtype}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result.get(k) for k in ("clip", "accuracy_cpu", "accuracy_mps", "parity", "crn_draw_sequences_equal", "native_noise_floor_cpu_vs_cpu")}))
    return result

if __name__ == "__main__":
    main()
