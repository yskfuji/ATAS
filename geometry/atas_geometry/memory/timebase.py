"""Exact frame timing of an input clip, and constant-frame-rate normalisation.

Reconstruction indexes frames by decode order and recapture converts a source time to
``round(t * fps)``. That is only correct for constant-frame-rate video, so a
variable-frame-rate clip is resampled by timestamp before reconstruction.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

# A clip is CFR when every frame interval is within this fraction of the median interval.
CFR_TOLERANCE = 0.02


def probe(path) -> dict:
    """Frame count and presentation times from the container's video packets (no decoding)."""
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        base = stream.time_base
        pts = sorted(float(packet.pts * base) for packet in container.demux(stream)
                     if packet.pts is not None and packet.size > 0)
    if not pts:
        raise ValueError(f"video has no frames: {path}")
    times = np.asarray(pts) - pts[0]
    deltas = np.diff(times)
    median = float(np.median(deltas)) if len(deltas) else 0.0
    cfr = bool(len(deltas) == 0 or (median > 0 and np.max(np.abs(deltas - median)) <= CFR_TOLERANCE * median))
    fps = (1.0 / median) if median > 0 else 0.0
    return {"frames": len(times), "fps": fps, "cfr": cfr, "duration": float(times[-1] + median),
            "pts_seconds": times.tolist()}


def normalize_cfr(source, destination, fps: float) -> dict:
    """Resample by timestamp: output frame i shows the source frame displayed at time i / fps."""
    import imageio_ffmpeg
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.stem + ".partial" + destination.suffix)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(source),
                    "-vf", f"fps={fps:.6f}:round=down", "-an", "-c:v", "libx264", "-crf", "12",
                    "-pix_fmt", "yuv420p", str(partial)], check=True)
    partial.replace(destination)
    return probe(destination)


def ensure_cfr(video, folder) -> tuple[Path, dict]:
    """Return (path to a CFR clip, timebase record). CFR input is used as is."""
    info = probe(video)
    record = {"source_frames": info["frames"], "source_fps": info["fps"], "source_cfr": info["cfr"]}
    if info["cfr"]:
        return Path(video), {**record, "normalized": False}
    target = Path(folder) / "cfr_input.mp4"
    normalized = normalize_cfr(video, target, info["fps"])
    if not normalized["cfr"]:
        raise ValueError("constant-frame-rate normalisation did not produce a CFR clip")
    (Path(folder) / "timebase.json").write_text(json.dumps({**record, "normalized": True,
        "normalized_frames": normalized["frames"], "normalized_fps": normalized["fps"]}, indent=2))
    return target, {**record, "normalized": True, "normalized_frames": normalized["frames"],
                    "normalized_fps": normalized["fps"]}
