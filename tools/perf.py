"""AC22: recon and render wall time and memory through the service on this Mac (n >= 5 each).

    python tools/perf.py <run-id> <clips-dir> [--runs 5] [--port 18331]

Workload: the first TUM analysis clip, cropped/resized by the service to 1280x720, with dynamic
keyword "person"; every run is a fresh story, so each recon is a complete first observation (Pi3X,
SAM3 via sam3.cpp, initial-world check, fusion). Memory is the peak, over 0.5 s samples, of the summed
physical footprint of the service's worker process tree (macOS counts Metal buffers in it).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools"), str(REPO / "client"), str(REPO / "geometry")]


def swap_used_mb():
    text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    return float(text.split("used = ")[1].split("M")[0])


class TreeSampler(threading.Thread):
    def __init__(self, root_pid):
        super().__init__(daemon=True)
        self.root, self.peak, self.stop = root_pid, 0, threading.Event()

    def run(self):
        import psutil
        from atas_geometry.service.proc_darwin import phys_footprint
        while not self.stop.wait(.5):
            try:
                tree = [psutil.Process(self.root)] + psutil.Process(self.root).children(recursive=True)
            except psutil.NoSuchProcess:
                continue
            total = sum(phys_footprint(p.pid) or 0 for p in tree)
            self.peak = max(self.peak, total)


def main():
    from e2e_world import start_service, wait_ready, render, film_review, production_sam3
    from atas.world.geometry_client import GeometryClient
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("clips", type=Path)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--port", type=int, default=18331)
    args = parser.parse_args()
    analysis = [c for c in json.loads((REPO / "evidence/prereg/analysis_clips.json").read_text())["clips"] if c["dataset"] == "TUM RGB-D"]
    margins = json.loads((REPO / "evidence/prereg/margins.json").read_text())
    clip = args.clips / analysis[0]["clip"] / "clip.mp4"
    out = REPO / "evidence" / args.run_id / "perf"
    out.mkdir(parents=True, exist_ok=True)
    hot = CACHE / "hot" / args.run_id / "perf"
    log = (out / "service.log").open("w")
    service = start_service("mps", hot, args.port, log, sam3_model=production_sam3(args.run_id))
    client = GeometryClient(base_url=f"http://127.0.0.1:{args.port}", token="e2e-token", enabled=True)
    runs = []
    try:
        ready = wait_ready(client)
        recommended = ready["details"]["cards"][0]["capacity_mib"] * 2**20
        for i in range(args.runs):
            story = f"perf{i}"
            swap_before = swap_used_mb()
            sampler = TreeSampler(service.pid); sampler.start()
            tick = time.perf_counter()
            job = client.recon(run_id="perf", story_id=story, shot_id="shot_a", scene_id="room", event_id="e",
                               duration=analysis[0]["frames"] / 15, video=clip, dynamic_keywords=("person",),
                               publication_mode="reviewed_candidate_v1", review_lineage={"target_render_job_id": None})
            result = client.wait(job.job_id, interval=2, timeout=7200)
            recon_s = time.perf_counter() - tick
            client._request("POST", "/v1/world-publication", {"run_id": "perf", "story_id": story, "scene_id": "room",
                            "shot_id": "shot_a", "candidate_job_id": result["fuse"]["candidate_job_id"],
                            "film_review": film_review(clip, {"target_render_job_id": None})}, timeout=600)
            tick = time.perf_counter()
            rendered = render(client, {"run_id": "perf", "story_id": story, "scene_id": "room", "shot_id": "render",
                                       "event_id": "e", "duration": 2.0, "fps": 8, "camera_relation": "cut_recompose",
                                       "geometry_layers": "l1"}, Path(result["l0_dir"]), 20)
            render_s = time.perf_counter() - tick
            sampler.stop.set(); sampler.join()
            recon_log = hot / "jobs" / job.job_id / "recon.log"
            sam3 = [json.loads(l.split(" ", 1)[1]) for l in recon_log.read_text().splitlines() if l.startswith("AVWM_SAM3_REPORT ")] \
                if recon_log.is_file() else []
            runs.append({"run": i, "recon_s": recon_s, "render_s": render_s, "render_status": rendered["status"],
                         "peak_footprint_bytes": sampler.peak, "swap_delta_mb": swap_used_mb() - swap_before,
                         "recon_timings": result.get("timings"), "sam3_passes": sam3})
            (out / f"run_{i}.json").write_text(json.dumps(runs[-1], indent=2, default=str))
            print(json.dumps({k: runs[-1][k] for k in ("run", "recon_s", "render_s", "peak_footprint_bytes", "swap_delta_mb")}), flush=True)
    finally:
        os.killpg(service.pid, signal.SIGTERM)
        log.close()
    sec_per_frame = max((p.get("sec_per_frame_max") or 0) for r in runs for p in r["sam3_passes"]) if runs else None
    result = {"n_runs": len(runs), "clip": analysis[0]["clip"], "clip_frames": analysis[0]["frames"],
              "perf_clip_frames_preregistered": margins["perf_clip_frames"], "recon_wh": [1280, 720],
              "pi3x_settings": json.loads((REPO / "evidence/prereg/analysis_clips.json").read_text()).get("settings")
              or {"recon_wh": [1280, 720], "requested": {"pi3_pixel_limit": 255000, "pi3_vo_chunk": 48},
                  "effective": {"pi3_pixel_limit": 160000, "pi3_vo_chunk": 24, "pi3_vo_overlap": 6}, "low_vram": True},
              "dtype": os.environ.get("GEOMETRY_MPS_DTYPE", "float32"),
              "recon_max_s": max(r["recon_s"] for r in runs), "render_max_s": max(r["render_s"] for r in runs),
              "first_run": runs[0] if runs else None, "peak_footprint_bytes_max": max(r["peak_footprint_bytes"] for r in runs),
              "recommended_max_memory": recommended, "swap_delta_mb_max": max(r["swap_delta_mb"] for r in runs),
              "memory_source": "darwin_lifetime_max_phys_footprint", "sam3_sec_per_frame_max": sec_per_frame,
              "sam3_model": production_sam3(args.run_id).name, "runs": runs}
    (REPO / "evidence" / args.run_id / "perf.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps({k: v for k, v in result.items() if k not in ("runs", "first_run")}, indent=2, default=str))


if __name__ == "__main__":
    main()
