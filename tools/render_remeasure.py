"""AC22 (A9): render time on the preregistered AC26 world, plus a smooth-camera demo render.

    python tools/render_remeasure.py <run-id> [--port 18341] [--demo-out <file.mp4>]

Run after tools/e2e_world.py --device mps has built the world (evidence/<run>/e2e_mps/*). The service is
started on that world's hot root. Five renders use distinct shot ids (so none is deduplicated), a locked
camera grounded on the SAM3 person track of the first frame that shows the subject, and the 20 degree side
offset. Each must reach the preregistered mean coverage; the report records every run, the first one
separately, the peak physical footprint of the service's process tree and the swap change.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools"), str(REPO / "client"), str(REPO / "geometry")]


SIDES = [20, 12, 28, 16, 24]   # distinct camera poses; the standard 20 degrees comes first
SUBJECT_MIN_AREA = 0.02        # a frame "shows the subject" if its projected mask covers >= 2 % of the frame
GREY = 127


def subject_points(l0: Path, frame: int):
    """Scene-world points of the subject (the SAM3 person track's pixels) on the grounding frame, from the world."""
    import numpy as np
    from atas_geometry.memory.vista4d_layers import ReconStore, _read_exr_depth
    from e2e_world import track_masks
    recon = ReconStore(l0, duration=2.0, exclude_sky=True)
    mask = track_masks(l0)[frame]
    depth = _read_exr_depth(recon.depth_files[frame]).astype(np.float64)
    h, w = depth.shape
    if mask.shape != (h, w):
        import cv2
        mask = cv2.resize(mask.astype("uint8"), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    ys, xs = np.nonzero(mask & (depth > 1e-5) & np.isfinite(depth))
    fx, fy, cx, cy = recon.intrinsics[frame][:4]
    z = depth[ys, xs]
    cam = np.stack(((xs + .5 - cx) / fx * z, (ys + .5 - cy) / fy * z, z), 1)
    world = cam @ recon.cam_c2w[frame][:3, :3].T + recon.cam_c2w[frame][:3, 3]
    meta = json.loads((l0 / "observation.json").read_text())["raw_to_scene"] or {"scale": 1, "rotation": np.eye(3).tolist(), "translation": [0, 0, 0]}
    return (float(meta["scale"]) * world @ np.asarray(meta["rotation"]).T + np.asarray(meta["translation"])).astype(np.float64)


def judge_render(job_dir: Path, points, started: float) -> dict:
    """Harness-side facts about one render, computed from the world and the file, not from the render's own report."""
    import cv2
    import imageio.v2 as imageio
    import numpy as np
    query = json.loads((job_dir / "camera_query.json").read_text())
    visible, inside, distances, centers = [], [], [], []
    for frame in query["frames"]:
        cam = frame["camera"]
        w2c, K = np.asarray(cam["world_to_camera"]), np.asarray(cam["intrinsic"])
        width, height = cam["image_size"]
        p = points @ w2c[:3, :3].T + w2c[:3, 3]
        front = p[:, 2] > 1e-3
        uv = p[front] @ K.T
        u, v = uv[:, 0] / uv[:, 2], uv[:, 1] / uv[:, 2]
        ok = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        canvas = np.zeros((height // 4, width // 4), np.uint8)
        canvas[(v[ok] / 4).astype(int), (u[ok] / 4).astype(int)] = 255
        canvas = cv2.dilate(canvas, np.ones((3, 3), np.uint8))
        visible.append(float((canvas > 0).mean()) >= SUBJECT_MIN_AREA)
        inside.append(bool(front.any() and 0 <= u.mean() < width and 0 <= v.mean() < height))
        c2w = np.linalg.inv(w2c)
        centers.append(np.round(c2w[:3, 3], 3).tolist())
        distances.append(float(np.linalg.norm(c2w[:3, 3] - points.mean(0))))
    frames = imageio.mimread(str(job_dir / "memory_input.mp4"), memtest=False)
    grey = float(np.mean([(np.abs(f.astype(int) - GREY).max(-1) <= 3).mean() for f in frames]))
    artifact = job_dir / "memory_input.mp4"
    return {"subject_visible_fraction": float(np.mean(visible)), "subject_center_in_frame": bool(all(inside)),
            "camera_distance_to_subject": float(np.mean(distances)), "camera_start": centers[0], "grey_fraction": grey,
            "frames": len(frames), "from_cache": bool(artifact.stat().st_mtime < started)}


def cpu_check(args, l0, points) -> dict:
    """Two renders of the CPU E2E world through a CPU service: the coverage rule must be passable on CPU too."""
    from e2e_world import RUN, STORY, SCENE, EVENT, start_service, wait_ready, render, production_sam3
    from atas.world.geometry_client import GeometryClient
    hot = CACHE / "hot" / args.run_id / "cpu"
    cpu_world = json.loads((REPO / "evidence" / args.run_id / "e2e_world_cpu.json").read_text())
    cpu_l0 = Path(json.loads((REPO / "evidence" / args.run_id / "e2e_cpu" / "job_shot_a.json").read_text())["l0_dir"])
    cpu_points = subject_points(cpu_l0, __import__("e2e_world").first_visible(cpu_l0))
    log = (REPO / "evidence" / args.run_id / "render_remeasure.cpu_service.log").open("w")
    service = start_service("cpu", hot, args.port + 1, log, sam3_model=production_sam3(args.run_id))
    client = GeometryClient(base_url=f"http://127.0.0.1:{args.port + 1}", token="e2e-token", enabled=True)
    rows = []
    try:
        wait_ready(client)
        for k, side in enumerate(SIDES[:2]):
            payload = {"run_id": RUN, "story_id": STORY, "scene_id": SCENE, "shot_id": f"render_remeasure_cpu_{k}", "event_id": EVENT,
                       "duration": 2.0, "fps": 8, "camera_relation": "cut_recompose", "geometry_layers": "l1"}
            started = time.time()
            rendered = render(client, payload, cpu_l0, side)
            rows.append({"side_degrees": side, "coverage": rendered.get("report", {}).get("mean_coverage"),
                         **judge_render(hot / "jobs" / rendered["job_id"], cpu_points, started)})
    finally:
        os.killpg(service.pid, signal.SIGTERM)
        log.close()
    return {"world_chain": cpu_world.get("world_chain"), "runs": rows, "coverage_min": min(r["coverage"] for r in rows),
            "subject_visible_fraction_min": min(r["subject_visible_fraction"] for r in rows),
            "grey_fraction_max": max(r["grey_fraction"] for r in rows)}


def main():
    from e2e_world import RUN, STORY, SCENE, EVENT, start_service, wait_ready, render, production_sam3, track_masks, first_visible
    from perf import TreeSampler, swap_used_mb
    from atas.world.geometry_client import GeometryClient
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--port", type=int, default=18341)
    parser.add_argument("--demo-out", type=Path)
    args = parser.parse_args()
    evidence = REPO / "evidence" / args.run_id
    hot = CACHE / "hot" / args.run_id / "mps"
    world = json.loads((evidence / "e2e_world.json").read_text())
    pair = json.loads((REPO / "evidence/prereg/e2e_pairs.json").read_text())["e2e_world_pair"]
    l0 = Path(json.loads((evidence / "e2e_mps" / "job_shot_a.json").read_text())["l0_dir"])
    visible = [bool(m.any()) for m in track_masks(l0)]
    margins = json.loads((REPO / "evidence/prereg/margins.json").read_text())
    log = (evidence / "render_remeasure.service.log").open("w")
    service = start_service("mps", hot, args.port, log, sam3_model=production_sam3(args.run_id))
    client = GeometryClient(base_url=f"http://127.0.0.1:{args.port}", token="e2e-token", enabled=True)
    runs, demo = [], None
    points = subject_points(l0, first_visible(l0))
    try:
        ready = wait_ready(client)
        recommended = ready["details"]["cards"][0]["capacity_mib"] * 2**20
        for k in range(5):
            swap_before = swap_used_mb()
            sampler = TreeSampler(service.pid); sampler.start()
            payload = {"run_id": RUN, "story_id": STORY, "scene_id": SCENE, "shot_id": f"render_remeasure_{k}", "event_id": EVENT,
                       "duration": 2.0, "fps": 8, "camera_relation": "cut_recompose", "geometry_layers": "l1"}
            started, tick = time.time(), time.perf_counter()
            rendered = render(client, payload, l0, SIDES[k])
            seconds = time.perf_counter() - tick
            sampler.stop.set(); sampler.join()
            report = rendered.get("report", {})
            runs.append({"run": k, "side_degrees": SIDES[k], "render_s": seconds, "status": rendered["status"],
                         "coverage": report.get("mean_coverage"), "peak_footprint_bytes": sampler.peak,
                         "swap_delta_mb": swap_used_mb() - swap_before, "job_id": rendered["job_id"],
                         **judge_render(hot / "jobs" / rendered["job_id"], points, started)})
            print(json.dumps(runs[-1]), flush=True)
        if args.demo_out:
            payload = {"run_id": RUN, "story_id": STORY, "scene_id": SCENE, "shot_id": "render_demo_arc", "event_id": EVENT,
                       "duration": 4.0, "fps": 24, "camera_relation": "cut_recompose", "geometry_layers": "l1",
                       "camera_move_name": "arc_left", "camera_movement_degrees": 35.0}
            import e2e_world
            original = e2e_world.grounded_plan

            def arc_plan(*a, **kw):
                plan = original(*a, **kw)
                plan["motion"] = {"name": "arc_left", "degrees": 35, "travel_ratio": .25}
                return plan
            e2e_world.grounded_plan = arc_plan
            try:
                demo = render(client, payload, l0, 20)
            finally:
                e2e_world.grounded_plan = original
            src = Path(demo["memory_input"])
            args.demo_out.parent.mkdir(parents=True, exist_ok=True)
            args.demo_out.write_bytes(src.read_bytes())
            print(json.dumps({"demo": str(args.demo_out), "coverage": demo.get("report", {}).get("mean_coverage")}), flush=True)
    finally:
        os.killpg(service.pid, signal.SIGTERM)
        log.close()
    cpu = cpu_check(args, l0, points)
    steady = runs[1:]
    result = {"n_runs": len(runs), "source_world": pair["new"], "source_world_sha256": pair["new_sha256"],
              "source_history": pair["history"], "world_version": world.get("world_chain", [None])[0],
              "subject_visible_frames": sum(visible), "coverage_floor": 0.5,
              "coverage_by_run": [r["coverage"] for r in runs], "coverage_min": min(r["coverage"] for r in runs),
              "coverage_mean": sum(r["coverage"] for r in runs) / len(runs),
              "subject_visible_fraction_min": min(r["subject_visible_fraction"] for r in runs),
              "subject_center_in_frame_all": all(r["subject_center_in_frame"] for r in runs),
              "grey_fraction_max": max(r["grey_fraction"] for r in runs),
              "distinct_camera_poses": len({tuple(r["camera_start"]) for r in runs}),
              "cache_hits": sum(1 for r in runs if r["from_cache"]), "cpu_render_check": cpu,
              "render_s_by_run": [r["render_s"] for r in runs], "render_max_s": max(r["render_s"] for r in runs),
              "first_run_s": runs[0]["render_s"], "steady_max_s": max(r["render_s"] for r in steady),
              "peak_footprint_bytes_max": max(r["peak_footprint_bytes"] for r in runs), "recommended_max_memory": recommended,
              "swap_delta_mb_max": max(r["swap_delta_mb"] for r in runs), "memory_source": "darwin_lifetime_max_phys_footprint",
              "runs": runs, "demo_render": bool(demo)}
    (evidence / "render_remeasure.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
