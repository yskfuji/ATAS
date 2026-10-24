"""AC14: fusion and rendering on MPS vs the CPU reference, on exact synthetic scenes.

    python tools/render_parity.py <run-id> [--scenes 6]

Each device runs in its own process (environment fixed before torch is imported). The worker
runs the production fusion entry (fuse_l0_to_l1) and the production renderer (bounded-square static
splats plus a source-pixel dynamic recapture of the moving box) and records every operator that
PyTorch reports as falling back to the CPU.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv-macos/bin/python"
GREY = 127


def worker(device: str, l0: Path, out: Path):
    sys.path[:0] = [str(REPO / "geometry"), str(REPO / "geometry/backbones")]
    fallback = []
    warnings.simplefilter("always")
    def capture(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "fall back to run on the CPU" in text or "not currently supported on the MPS backend" in text:
            fallback.append(text.split("'")[1] if "'" in text else text[:120])
    warnings.showwarning = capture
    import torch
    from atas_geometry import device as accel
    from atas_geometry.memory.fuse import fuse_l0_to_l1
    from atas_geometry.memory.fusion import FusedSceneMemory
    from atas_geometry.memory.observation import CameraCalibration
    from atas_geometry.memory.vista4d_layers import ReconStore
    from atas_geometry.render.render import render_l1_gpu
    recon = ReconStore(l0, duration=2.0, exclude_sky=True)
    l1 = out / "l1.npz"
    report = fuse_l0_to_l1(l0, l1, duration=2.0, shot_id="synthetic", gpu_index=0)
    memory = FusedSceneMemory.load(l1)
    cameras = []
    for index in range(0, recon.count, 6):
        c2w = recon.cam_c2w[index].copy(); c2w[:3, 3] += [0.15, -0.05, 0.2]  # a novel nearby viewpoint
        k = recon.intrinsics[index]
        intrinsic = np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.0]])
        cameras.append(CameraCalibration("novel", np.linalg.inv(c2w), intrinsic, (int(2 * k[2]), int(2 * k[3]))))
    width, height = cameras[0].image_size
    frames, stats = render_l1_gpu(memory, cameras, width=width, height=height, device=accel.torch_device(0),
                                  splat_mode="bounded_square", point_footprint=float(memory.voxel_size), max_point_radius=2)
    # Dynamic recapture: the box's pixels of each source frame, unprojected and splatted with source support.
    dynamic = []
    for index in range(0, recon.count, 6):
        batch = recon.unproject(index)
        keep = batch.foreground >= 0.5
        xyz = np.asarray(batch.xyz[keep], np.float32)
        rgb = np.asarray(batch.rgb[keep], np.uint8)
        support = np.full(len(xyz), float(memory.voxel_size))
        dynamic.append((xyz, rgb, support, np.full(len(xyz), 2)))
    dyn_frames, dyn_stats = render_l1_gpu(memory, cameras, width=width, height=height, device=accel.torch_device(0),
                                          splat_mode="bounded_square", point_footprint=float(memory.voxel_size),
                                          max_point_radius=2, dynamic_frames=iter(dynamic),
                                          dynamic_splat_mode="source_pixel", max_dynamic_radius=2)
    np.savez_compressed(out / "render.npz", static=np.stack(frames), dynamic=np.stack(dyn_frames))
    (out / "worker.json").write_text(json.dumps({
        "device": str(accel.torch_device(0)), "voxel_size": float(memory.voxel_size), "points": int(len(memory.xyz)),
        "fallback_ops": sorted(set(fallback)), "torch": torch.__version__,
        "fuse_report_keys": sorted(report) if isinstance(report, dict) else None,
        "gpu_splat_ms": [stats["gpu_splat_ms"], dyn_stats["gpu_splat_ms"]],
        "peak_memory_bytes": accel.peak_memory_bytes()}, indent=2))


def psnr_covered(a, b):
    covered = (a != GREY).any(-1) | (b != GREY).any(-1)
    if not covered.any():
        return float("inf"), 0
    diff = (a[covered].astype(np.float64) - b[covered].astype(np.float64)) / 255.0
    mse = float(np.mean(diff ** 2))
    return (float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)), int(covered.sum())


def coverage_iou(a, b):
    ca, cb = (a != GREY).any(-1), (b != GREY).any(-1)
    union = (ca | cb).sum()
    return 1.0 if union == 0 else float((ca & cb).sum() / union)


def run_device(device, l0, out, fallback_enabled="1"):
    out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "GEOMETRY_DEVICE": device, "PYTORCH_ENABLE_MPS_FALLBACK": fallback_enabled,
           "PYTHONPATH": f"{REPO / 'geometry'}{os.pathsep}{REPO / 'geometry/backbones'}"}
    if device == "mps":
        sys.path.insert(0, str(REPO / "geometry"))
        from atas_geometry import device as accel
        env.update({k: v for k, v in accel.worker_env(0).items() if k != "PYTORCH_ENABLE_MPS_FALLBACK"})
    completed = subprocess.run([str(PYTHON), __file__, "worker", device, str(l0), str(out)],
                               env=env, capture_output=True, text=True)
    (out / "worker.log").write_text(completed.stdout + completed.stderr)
    return completed.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--scenes", type=int, default=6)
    args = parser.parse_args()
    sys.path.insert(0, str(REPO / "tools"))
    from synthetic_scenes import render_scene, write_l0
    root = REPO / "evidence" / args.run_id / "render_parity"
    rows = []
    for seed in range(args.scenes):
        scene_dir = root / f"synthetic_{seed:02d}"
        l0 = write_l0(render_scene(seed), scene_dir / "l0")
        codes = {d: run_device(d, l0, scene_dir / d) for d in ("cpu", "mps")}
        strict = run_device("mps", l0, scene_dir / "mps_no_fallback", fallback_enabled="0")
        if any(codes.values()):
            rows.append({"scene": seed, "error": codes}); continue
        cpu, mps = (json.loads((scene_dir / d / "worker.json").read_text()) for d in ("cpu", "mps"))
        cpu_l1, mps_l1 = (np.load(scene_dir / d / "l1.npz") for d in ("cpu", "mps"))
        cpu_r, mps_r = (np.load(scene_dir / d / "render.npz") for d in ("cpu", "mps"))
        psnrs, ious = [], []
        for key in ("static", "dynamic"):
            for a, b in zip(cpu_r[key], mps_r[key]):
                psnrs.append(psnr_covered(a, b)[0]); ious.append(coverage_iou(a, b))
        rows.append({"scene": seed, "cpu_points": cpu["points"], "mps_points": mps["points"],
                     "voxel_count_rel_diff": abs(mps["points"] - cpu["points"]) / cpu["points"],
                     "l1_xyz_identical": bool(np.array_equal(cpu_l1["xyz"], mps_l1["xyz"])),
                     "l1_xyz_max_abs_diff": float(np.abs(cpu_l1["xyz"] - mps_l1["xyz"]).max())
                     if cpu_l1["xyz"].shape == mps_l1["xyz"].shape else None,
                     "psnr_covered_min_db": min(psnrs), "coverage_iou_min": min(ious),
                     "fallback_ops": mps["fallback_ops"], "strict_no_fallback_exit_code": strict,
                     "splat_ms": {"cpu": cpu["gpu_splat_ms"], "mps": mps["gpu_splat_ms"]}})
    ok = [r for r in rows if "error" not in r]
    summary = {"n_scenes": len(rows), "errors": [r for r in rows if "error" in r],
               "voxel_count_rel_diff": max((r["voxel_count_rel_diff"] for r in ok), default=None),
               "psnr_covered_union_min_db": min((r["psnr_covered_min_db"] for r in ok), default=None),
               "coverage_iou_min": min((r["coverage_iou_min"] for r in ok), default=None),
               "fallback_ops": sorted({op for r in ok for op in r["fallback_ops"]}),
               "strict_no_fallback_all_succeeded": all(r["strict_no_fallback_exit_code"] == 0 for r in ok),
               "cpu_fallback_wall_frac": None, "scenes": rows}
    (REPO / "evidence" / args.run_id / "render_parity.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "scenes"}, indent=2, default=float))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "worker":
        worker(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]))
    else:
        main()
