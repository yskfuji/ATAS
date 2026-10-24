"""AC18 evidence: exact frame counts and timing of every evaluation clip, plus VFR normalisation.

    python tools/timebase_evidence.py <run-id> <clips-dir>
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "geometry"), str(REPO / "geometry/backbones"), str(REPO / "geometry/tests/mac")]


def ffprobe_frames(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                          "stream=nb_read_frames", "-of", "csv=p=0", str(path)], capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def main(run_id: str, clips_dir: Path):
    from atas_geometry.memory import timebase
    from utils.media import load_video
    from test_video_io import decode, make_vfr, source_index
    rows, index_errors = [], []
    for clip in sorted(p for p in clips_dir.iterdir() if (p / "clip.mp4").is_file()):
        info = timebase.probe(clip / "clip.mp4")
        expected = ffprobe_frames(clip / "clip.mp4")
        video, _ = load_video(str(clip / "clip.mp4"))
        pts = np.asarray(info["pts_seconds"])
        error = float(np.max(np.abs(pts * info["fps"] - np.arange(len(pts))))) if len(pts) else 0.0
        index_errors.append(error)
        rows.append({"clip": clip.name, "frames_probe": info["frames"], "frames_ffprobe": expected,
                     "frames_decoded": len(video), "cfr": info["cfr"], "pts_index_error_frames": error})
    vfr = []
    for pattern, durations in (("alternating", [1 / 30, 2 / 30] * 12), ("stall", [1 / 30] * 10 + [5 / 30] + [1 / 30] * 13)):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            clip = make_vfr(tmp, durations)
            info = timebase.probe(clip)
            shown = [source_index(f) for f in decode(clip)]
            path, record = timebase.ensure_cfr(clip, tmp / "job")
            fps, starts = record["normalized_fps"], np.asarray(info["pts_seconds"])
            errors = [abs(source_index(f) - shown[int(np.searchsorted(starts, i / fps + 1e-6, side="right") - 1)])
                      for i, f in enumerate(decode(path))]
            vfr.append({"pattern": pattern, "source_cfr": info["cfr"], "normalized_cfr": timebase.probe(path)["cfr"],
                        "max_alignment_error_frames": int(max(errors)), "frames": len(errors)})
    result = {"n_clips": len(rows), "all_counts_agree": all(r["frames_probe"] == r["frames_ffprobe"] == r["frames_decoded"] for r in rows),
              "max_index_error_frames": max(index_errors), "n_vfr_clips": len(vfr),
              "recapture_alignment_error_frames": max(v["max_alignment_error_frames"] for v in vfr),
              "vfr": vfr, "clips": rows}
    (REPO / "evidence" / run_id / "timebase.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k not in ("clips", "vfr")}))


if __name__ == "__main__":
    main(sys.argv[1], Path(sys.argv[2]))
