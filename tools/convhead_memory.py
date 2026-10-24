"""A10 memory: one Pi3X window (24 real frames, 532x294, fp32, MPS, real weights) at a given ConvHead frame chunk.

    python tools/convhead_memory.py <run-id> <chunk>        (one process per setting: the allocator high-water mark is per process)
Common random numbers; the outputs are saved beside the record so settings can be compared bit for bit.
Memory = lifetime peak physical footprint of this process and the maximum of torch.mps.driver_allocated_memory
sampled every 10 ms.
"""
import json, os, sys, threading, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "geometry/backbones"), str(REPO / "geometry")]
os.environ["GEOMETRY_DEVICE"] = "mps"
import numpy as np, torch
import crn_launch
from utils.media import crop_and_resize_video, load_video
from utils.recon_and_seg.recon_pi3 import init_pi3, preprocess_frames_pi3
from pi3.models.layers import conv_head
from atas_geometry.service.proc_darwin import lifetime_peak_footprint

run_id, chunk = sys.argv[1], int(sys.argv[2])
conv_head.FRAME_CHUNK = chunk   # 0 = upstream single pass
video, _ = load_video(str(REPO.parent / "cache/clips/tum_freiburg1_rpy/clip.mp4"))
video = crop_and_resize_video(video[:24], 720, 1280, resample="lanczos")
frames, (h, w) = preprocess_frames_pi3(video, pixel_limit=160000)
model = init_pi3(str(REPO.parent / "cache/weights/pi3x"), device="mps")
torch.mps.synchronize(); torch.mps.empty_cache()
before = torch.mps.driver_allocated_memory()
images = frames[None].to("mps")
crn_launch.install(); torch.manual_seed(0)

peak = {"driver": 0}; stop = threading.Event()
def sample():
    while not stop.is_set():
        peak["driver"] = max(peak["driver"], torch.mps.driver_allocated_memory()); time.sleep(0.01)
thread = threading.Thread(target=sample, daemon=True); thread.start()
tick = time.perf_counter()
with torch.inference_mode():
    pred = model(images)
torch.mps.synchronize(); seconds = time.perf_counter() - tick
stop.set(); thread.join()
arrays = {k: pred[k].float().cpu().numpy() for k in ("points", "local_points", "conf", "camera_poses")}
out = REPO / "evidence" / run_id / "convhead_memory"; out.mkdir(exist_ok=True); np.savez(out / f"chunk{chunk}.npz", **arrays)
record = {"chunk": chunk, "frames": 24, "wh": [w, h], "forward_s": round(seconds, 2),
      "driver_before_gib": round(before / 2**30, 2), "driver_peak_gib": round(peak["driver"] / 2**30, 2),
      "lifetime_peak_footprint_gib": round(lifetime_peak_footprint(os.getpid()) / 2**30, 2),
      "crn_draws": crn_launch.draw_summary()["draws"]}
with open(out / "records.jsonl", "a") as f:
    f.write(json.dumps(record) + "\n")
print("RESULT " + json.dumps(record))
