"""AC26: the real world-state loop through the HTTP service (Pi3X + sam3.cpp), with no generator.

    python tools/e2e_world.py <run-id> --device mps|cpu [--port 18311]

Two clips of one synthetic room (exact geometry) stand in for generated shots: clip A becomes the
first observation, clip B is registered against it, both are published through the reviewed
publication route, a memory render and an event recapture are rendered, and recapture accuracy is
measured against the source frames from the source cameras. The semantic camera plan that the LLM
visual planner would author is replaced by a deterministic plan grounded on the SAM3 track box.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
PYTHON = REPO / ".venv-macos/bin/python"
sys.path[:0] = [str(REPO / "tools"), str(REPO / "client"), str(REPO / "geometry"), str(REPO / "geometry/backbones")]
RUN, STORY, SCENE, EVENT = "e2e", "synthetic", "room", "box_moves"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def production_sam3(run_id: str) -> Path:
    """The SAM3 model chosen by A8 (sam3_quantisation.json); f16 until a selection exists."""
    report = REPO / "evidence" / run_id / "sam3_quantisation.json"
    selected = json.loads(report.read_text())["selected"] if report.is_file() else "f16"
    return CACHE / "weights/sam3cpp" / f"sam3-{selected}.ggml"


def start_service(device, hot, port, log, sam3_model=None):
    env = {**os.environ, "GEOMETRY_DEVICE": device, "GEOMETRY_HOT_ROOT": str(hot), "GEOMETRY_PORT": str(port),
           "GEOMETRY_PYTHON": str(PYTHON), "GEOMETRY_PI3_MODEL": str(CACHE / "weights/pi3x"),
           "GEOMETRY_SAM3_CHECKPOINT": str(sam3_model or CACHE / "weights/sam3cpp/sam3-f16.ggml"),
           "GEOMETRY_SAM3CPP_BIN": str(CACHE / "sam3.cpp/atas-build/atas_sam3_track"),
           "GEOMETRY_API_TOKEN": "e2e-token"}
    if device == "cpu":
        env["GEOMETRY_SAM3CPP_NO_GPU"] = "1"
        # The CPU service is the comparison reference, not a production target: SAM3 on CPU needs ~35 min per
        # clip, above upstream's 2400 s recon deadline (which still applies to MPS). Reference-only setting.
        env["GEOMETRY_RECON_DEADLINE_SECONDS"] = "21600"
        env["GEOMETRY_GPU_QUEUE_TIMEOUT_SECONDS"] = "21600"
    process = subprocess.Popen([str(REPO / "geometry/run_geometry_service.sh")], env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    return process


def wait_ready(client, timeout=600):
    limit = time.monotonic() + timeout
    while time.monotonic() < limit:
        try:
            return client._request("GET", "/readyz")
        except Exception as exc:  # 503 while dependencies are being probed, refused while starting
            last = exc
            time.sleep(2)
    raise TimeoutError(f"service not ready: {last}")


def film_review(video, lineage):
    return {"schema_version": 1, "stage": "film", "decision": "pass", "cache_key": f"review-{sha256(video)[:12]}",
            "video_sha256": sha256(video), "lineage": lineage,
            "checks": [{"criterion": "action", "status": "pass"}, {"criterion": "event", "status": "pass"}]}


def track_masks(l0: Path):
    tracks = json.loads((l0 / "instance_tracks.json").read_text())
    from PIL import Image
    track = next(iter(tracks["tracks"].values()))
    return [np.asarray(Image.open(p)) > 127 for p in sorted((l0 / track["mask_dir"]).glob("*.png"))]


def first_visible(l0: Path) -> int:
    return next(i for i, m in enumerate(track_masks(l0)) if m.any())


def grounded_plan(l0: Path, frame: int, side: float):
    """Deterministic stand-in for the visual planner: the subject box is the SAM3 track's mask box
    on exactly the frame the camera context grounds on."""
    mask = track_masks(l0)[frame]
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise ValueError(f"subject not visible on grounding frame {frame}")
    bbox = [xs.min() / width, ys.min() / height, (xs.max() + 1) / width, (ys.max() + 1) / height]
    return {"schema_version": 1, "status": "ok",
            "subjects": [{"label": "box", "kind": "dynamic", "part": "body", "visible": True, "confidence": .95,
                          "bbox": [float(v) for v in bbox],
                          "anchor_uv": [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)]}],
            "framing": {"shot_size": "medium", "screen_position": [.5, .55], "vertical_fraction": .45,
                        "horizontal_fraction": .45, "side_degrees": side, "elevation_degrees": 0},
            "motion": {"name": "locked", "degrees": 0, "travel_ratio": 0}}


def render(client, payload, l0: Path, side: float):
    if payload.get("render_mode") == "recapture":
        payload["source_time_start"] = first_visible(l0) / json.loads((l0 / "observation.json").read_text())["source_fps"]
        payload["source_time_end"] = payload["source_time_start"] + payload["duration"]
    else:
        payload["camera_grounding_frame_index"] = first_visible(l0)
    context = client.camera_context(payload)
    plan = grounded_plan(l0, context["frame_index"], side)
    plan["context"] = context
    job = client.render({**payload, "camera_plan": plan, "camera_contract": "semantic-camera-v1"})
    return client.wait(job.job_id, interval=3, timeout=1800)


def recapture_accuracy(scene: Path, source_shot: str, device: str):
    """Re-render the committed event from the source cameras at its source times and compare."""
    import torch
    from atas_geometry import device as accel
    from atas_geometry.memory.fusion import FusedSceneMemory
    from atas_geometry.memory.observation import CameraCalibration
    from atas_geometry.memory.vista4d_layers import ReconStore
    from atas_geometry.render.render import render_l1_gpu
    from atas_geometry.world.world_state import load_world, checked_path
    state = load_world(scene)
    l0 = checked_path(scene, state["observations"][source_shot]["l0"])
    memory = FusedSceneMemory.load(checked_path(scene, state["l1"]))
    recon = ReconStore(l0, duration=2.0, exclude_sky=True)
    meta = json.loads((l0 / "observation.json").read_text())
    transform = meta["raw_to_scene"]
    scale, rotation, translation = float(transform["scale"]), np.asarray(transform["rotation"]), np.asarray(transform["translation"])
    frames = list(range(0, recon.count, 4))
    cameras, dynamic, sources, masks = [], [], [], []
    for index in frames:
        c2w = recon.cam_c2w[index].copy()
        world = np.eye(4); world[:3, :3] = rotation @ c2w[:3, :3]; world[:3, 3] = scale * rotation @ c2w[:3, 3] + translation
        k = recon.intrinsics[index]
        h, w = recon.frames[index].shape[:2]
        cameras.append(CameraCalibration("source", np.linalg.inv(world), np.array([[k[0], 0, k[2]], [0, k[1], k[3]], [0, 0, 1.]]), (w, h)))
        batch = recon.unproject(index)
        keep = batch.foreground >= .5
        xyz = (scale * np.asarray(batch.xyz[keep], np.float64) @ rotation.T + translation).astype(np.float32)
        dynamic.append((xyz, np.asarray(batch.rgb[keep], np.uint8), np.full(len(xyz), float(memory.voxel_size)), np.full(len(xyz), 2)))
        sources.append(recon.frames[index]); masks.append(recon.dynamic_mask_at(index, (h, w)))
    h, w = sources[0].shape[:2]
    dev = torch.device(device)
    static, _ = render_l1_gpu(memory, cameras, width=w, height=h, device=dev, splat_mode="bounded_square",
                              point_footprint=float(memory.voxel_size), max_point_radius=2)
    full, _ = render_l1_gpu(memory, cameras, width=w, height=h, device=dev, splat_mode="bounded_square",
                            point_footprint=float(memory.voxel_size), max_point_radius=2, dynamic_frames=iter(dynamic),
                            dynamic_splat_mode="source_pixel", max_dynamic_radius=2)
    psnrs, ious = [], []
    for source, mask, image, background in zip(sources, masks, full, static):
        covered = (image != 127).any(-1)
        diff = (image[covered].astype(np.float64) - source[covered].astype(np.float64)) / 255
        psnrs.append(10 * np.log10(1 / max(float(np.mean(diff ** 2)), 1e-12)))
        if not mask.any():
            continue  # the event is not visible in this source frame: no dynamic region to compare
        dyn = (image != background).any(-1)  # pixels the event's dynamic points changed
        ious.append(float((dyn & mask).sum() / (dyn | mask).sum()))
    return {"frames": len(frames), "psnr_covered_db": float(np.mean(psnrs)), "psnr_covered_min_db": float(np.min(psnrs)),
            "mask_iou": float(np.mean(ious)), "mask_iou_min": float(np.min(ious))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--device", choices=["mps", "cpu"], required=True)
    parser.add_argument("--port", type=int, default=18311)
    parser.add_argument("--clips", type=Path, help="real clips dir: use the preregistered e2e_world_pair instead of the synthetic room")
    args = parser.parse_args()
    from synthetic_scenes import render_scene, write_clip
    from atas.world.geometry_client import GeometryClient
    out = REPO / "evidence" / args.run_id / f"e2e_{args.device}"
    out.mkdir(parents=True, exist_ok=True)
    hot = CACHE / "hot" / args.run_id / args.device
    clips, keyword, durations = {}, "red box", {"shot_a": 2.0, "shot_b": 2.0}
    if args.clips:
        pair = json.loads((REPO / "evidence/prereg/e2e_pairs.json").read_text())["e2e_world_pair"]
        keyword = "person"
        for name, clip in (("shot_a", pair["history"]), ("shot_b", pair["new"])):
            meta = json.loads((args.clips / clip / "meta.json").read_text())
            clips[name], durations[name] = args.clips / clip / "clip.mp4", meta["clip_frames"] / meta["fps"]
        report_source = {"scene_source": "preregistered TUM pair", **pair}
    else:
        for name, offset, box in (("shot_a", 0.0, 0.0), ("shot_b", 0.35, 0.3)):
            clips[name] = write_clip(render_scene(7, frames=48, h=720, w=1280, path_offset=offset, box_offset=box),
                                     out / f"{name}.mp4")
        report_source = {"scene_source": "synthetic room (functional dry run)"}
    log = (out / "service.log").open("w")
    service = start_service(args.device, hot, args.port, log, sam3_model=production_sam3(args.run_id))
    client = GeometryClient(base_url=f"http://127.0.0.1:{args.port}", token="e2e-token", enabled=True)
    report = {"device": args.device, "via_http": True, "job_ids": {}, "timings_s": {}, **report_source,
              "sam3_model": production_sam3(args.run_id).name}
    try:
        ready = wait_ready(client)
        (out / "readyz_live.json").write_text(json.dumps({"status_code": 200, "body": ready}, indent=2))
        published = {}
        for name in ("shot_a", "shot_b"):
            lineage = {"target_render_job_id": report["job_ids"].get("render")}
            tick = time.perf_counter()
            job = client.recon(run_id=RUN, story_id=STORY, shot_id=name, scene_id=SCENE, event_id=EVENT, duration=durations[name],
                               video=clips[name], dynamic_keywords=(keyword,), publication_mode="reviewed_candidate_v1",
                               review_lineage=lineage)
            report["job_ids"][f"recon_{name}"] = job.job_id
            result = client.wait(job.job_id, interval=5, timeout=4800 if args.device == "mps" else 25200)
            report["timings_s"][f"recon_{name}"] = time.perf_counter() - tick
            (out / f"job_{name}.json").write_text(json.dumps(result, indent=2, default=str))
            fuse = result.get("fuse", {})
            publication = client._request("POST", "/v1/world-publication", {
                "run_id": RUN, "story_id": STORY, "scene_id": SCENE, "shot_id": name,
                "candidate_job_id": fuse.get("candidate_job_id"), "film_review": film_review(clips[name], lineage)}, timeout=600)
            published[name] = {"fuse": {k: fuse.get(k) for k in ("registered", "candidate_job_id", "publication_status", "home")},
                               "publication": {k: publication.get(k) for k in ("publication_status", "world_version", "commit_published")},
                               "initial_world": fuse.get("initial_world_alignment")}
            if name == "shot_a":
                l0 = Path(result["l0_dir"])
                payload = {"run_id": RUN, "story_id": STORY, "scene_id": SCENE, "shot_id": "render_1", "event_id": EVENT,
                           "duration": 2.0, "fps": 8, "camera_relation": "cut_recompose", "geometry_layers": "l1"}
                tick = time.perf_counter()
                rendered = render(client, payload, l0, 20)
                report["timings_s"]["render"] = time.perf_counter() - tick
                report["job_ids"]["render"] = rendered["job_id"]
                report["render"] = {"status": rendered["status"], "coverage": rendered.get("report", {}).get("mean_coverage")}
        scene = hot / RUN / STORY / "scenes" / SCENE
        world = json.loads((scene / "world_state.json").read_text())
        chain, version = [], world["version"]
        while version:
            chain.append(version)
            version = json.loads((scene / ".world_versions" / version / "commit.json").read_text()).get("parent_version")
        payload = {"run_id": RUN, "story_id": STORY, "scene_id": SCENE, "shot_id": "recapture_1", "event_id": EVENT,
                   "duration": 1.0, "fps": 8, "camera_relation": "recapture", "geometry_layers": "l1+l2",
                   "render_mode": "recapture", "source_shot_id": "shot_a", "source_event_id": EVENT,
                   "start_from_camera": "source_time_start"}
        tick = time.perf_counter()
        recaptured = render(client, payload, Path(json.loads((out / "job_shot_a.json").read_text())["l0_dir"]), 0)
        report["timings_s"]["recapture"] = time.perf_counter() - tick
        report["job_ids"]["recapture"] = recaptured["job_id"]
        accuracy = recapture_accuracy(scene, "shot_a", args.device)
        iw = published["shot_a"]["initial_world"] or {}
        report.update({
            "clip_a": {"initial_world_status": iw.get("status"), "reason": iw.get("reason"),
                       "frames": json.loads((out / "job_shot_a.json").read_text()).get("num_frames"),
                       "first_window": (iw.get("input") or {}).get("first_window"), **published["shot_a"]},
            "clip_b": {"registered": int(bool(published["shot_b"]["fuse"].get("registered"))), **published["shot_b"]},
            "world_versions": len(chain), "world_chain": chain,
            "recapture": {"status": recaptured["status"], "dynamic_points_per_frame": recaptured.get("report", {}).get("dynamic_points_per_frame"),
                          **accuracy, "psnr_floor_rule": "max(20 dB, CPU-fp32 recapture PSNR - 1 dB)"}})
    finally:
        os.killpg(service.pid, signal.SIGTERM)
        try:
            service.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(service.pid, signal.SIGKILL)
        log.close()
    name = "e2e_world.json" if args.device == "mps" else "e2e_world_cpu.json"
    (REPO / "evidence" / args.run_id / name).write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({k: report.get(k) for k in ("clip_a", "clip_b", "world_versions", "recapture", "timings_s")}, indent=2, default=str))


if __name__ == "__main__":
    main()
