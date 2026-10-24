"""A10 certification: Pi3X with frame-chunked ConvHeads returns bit-identical outputs to the single pass.

    python tools/convhead_certify.py <run-id> [--device mps|cpu] [--chunk 2] [--clips 8] [--frames 24]

One model per process; for each clip (a seeded sample of the preregistered analysis clips) the same window is run
single-pass and chunked under common random numbers, and every output tensor is compared with torch.equal.
Memory is measured separately (one process per setting) because the allocator keeps a process-lifetime high-water mark.
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "geometry" / "backbones"), str(REPO / "geometry")]
SEED = 20261022
OUTPUTS = ("points", "local_points", "conf", "camera_poses")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--chunk", type=int, default=2)
    parser.add_argument("--clips", type=int, default=8)
    parser.add_argument("--frames", type=int, default=24)
    args = parser.parse_args()
    os.environ["GEOMETRY_DEVICE"] = args.device
    import torch
    import crn_launch
    from pi3.models.layers import conv_head
    from utils.media import crop_and_resize_video, load_video
    from utils.recon_and_seg.recon_pi3 import init_pi3, preprocess_frames_pi3

    pool = [c["clip"] for c in json.loads((REPO / "evidence/prereg/analysis_clips.json").read_text())["clips"]]
    chosen = sorted(random.Random(SEED).sample(pool, args.clips))
    model = init_pi3(str(REPO.parent / "cache/weights/pi3x"), device=args.device)
    rows = []
    for clip in chosen:
        video, _ = load_video(str(REPO.parent / "cache/clips" / clip / "clip.mp4"))
        video = crop_and_resize_video(video[:args.frames], 720, 1280, resample="lanczos")
        frames, (h, w) = preprocess_frames_pi3(video, pixel_limit=160000)
        images = frames[None].to(args.device)
        out, seconds = {}, {}
        for chunk in (0, args.chunk, 0):   # the repeated single pass shows the run-to-run floor of the device itself
            conv_head.FRAME_CHUNK = chunk
            crn_launch.install()
            torch.manual_seed(0)
            tick = time.perf_counter()
            with torch.inference_mode():
                pred = model(images)
            if args.device == "mps":
                torch.mps.synchronize()
            seconds.setdefault(chunk, []).append(time.perf_counter() - tick)
            out.setdefault(chunk, []).append({k: pred[k].float().cpu() for k in OUTPUTS})
        row = {"clip": clip, "frames": int(images.shape[1]), "wh": [w, h],
               "single_vs_single_identical": all(torch.equal(out[0][0][k], out[0][1][k]) for k in OUTPUTS),
               "chunked_identical": all(torch.equal(out[0][0][k], out[args.chunk][0][k]) for k in OUTPUTS),
               "max_abs_diff": {k: float((out[0][0][k] - out[args.chunk][0][k]).abs().max()) for k in OUTPUTS},
               "finite": all(bool(torch.isfinite(out[0][0][k]).all()) for k in OUTPUTS),
               "seconds_single": seconds[0], "seconds_chunked": seconds[args.chunk]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del out, pred
    result = {"device": args.device, "chunk": args.chunk, "seed": SEED, "clips": chosen, "rows": rows,
              "all_identical": all(r["chunked_identical"] for r in rows),
              "single_pass_reproducible": all(r["single_vs_single_identical"] for r in rows)}
    path = REPO / "evidence" / args.run_id / f"convhead_certify_{args.device}.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}))


if __name__ == "__main__":
    main()
