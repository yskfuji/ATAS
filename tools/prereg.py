"""Stage-2 preregistration: deterministic clip lists, CPU-only margins, then a hash lock.

    python tools/prereg.py lists   <clips-dir>               # seeded split of whole sequences
    python tools/prereg.py margins <clips-dir> <eval-out>     # from CPU-fp32 calibration results only
    python tools/prereg.py lock                               # evidence/prereg/prereg.lock (commit it before any MPS run)

The split and the margins are deterministic functions of the clip set and of CPU-only results, so
the gate can recompute both and seeing MPS data first could not have changed them.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "evidence/prereg"
SEED = 20261019
SPLIT = {"calibration": 6, "pilot": 5}
JF_PUBLISHED = 0.922
JF_SOURCE = ("Carion et al., SAM 3: Segment Anything with Concepts, arXiv:2511.16719v2, Table 5 "
             "(DAVIS17 J&F 92.2, first-frame mask prompts); floor = published - 0.03; our protocol "
             "(box + one interior point) is weaker than mask prompting, so the floor is conservative")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entry(clip: Path) -> dict:
    meta = json.loads((clip / "meta.json").read_text())
    return {"clip": clip.name, "dataset": meta["dataset"], "sequence": meta["sequence"], "frames": meta["clip_frames"],
            "clip_sha256": sha256(clip / "clip.mp4"), "gt_sha256": sha256(clip / "gt.npz")}


def split(clips_dir: Path) -> dict:
    names = sorted(p.name for p in clips_dir.iterdir() if (p / "meta.json").is_file() and not p.name.endswith("_history"))
    order = [names[i] for i in np.random.default_rng(SEED).permutation(len(names))]
    groups = {"calibration": order[:SPLIT["calibration"]],
              "pilot": order[SPLIT["calibration"]:SPLIT["calibration"] + SPLIT["pilot"]],
              "analysis": sorted(order[SPLIT["calibration"] + SPLIT["pilot"]:])}
    return groups


def lists(clips_dir: Path):
    PREREG.mkdir(parents=True, exist_ok=True)
    groups = split(clips_dir)
    for name, members in groups.items():
        (PREREG / f"{name}_clips.json").write_text(json.dumps({"rule": f"seeded permutation (seed {SEED}) of all "
            "sequence clips; first 6 calibration, next 5 pilot, rest analysis", "clips": [entry(clips_dir / m) for m in members]}, indent=2))
    pairs = [{"history": f"{m}_history", "new": m, "history_sha256": sha256(clips_dir / f"{m}_history" / "clip.mp4"),
              "new_sha256": sha256(clips_dir / m / "clip.mp4")}
             for m in groups["analysis"] if (clips_dir / f"{m}_history" / "clip.mp4").is_file()]
    dynamic = [p for p in pairs if any(w in p["new"] for w in ("walking", "sitting"))]
    world_pair = dynamic[0] if dynamic else None
    (PREREG / "e2e_pairs.json").write_text(json.dumps({"rule": "every analysis TUM sequence: the 48-frame stride-2 window "
        "immediately before its analysis clip is the history (first observation)", "pairs": pairs,
        "e2e_world_pair_rule": "AC26 HTTP loop: the first analysis pair (sorted) whose sequence shows people (walking/sitting)",
        "e2e_world_pair": world_pair}, indent=2))
    print(json.dumps({k: len(v) for k, v in groups.items()} | {"pairs": len(pairs)}))


def margins(clips_dir: Path, eval_out: Path) -> dict:
    calibration = json.loads((PREREG / "calibration_clips.json").read_text())["clips"]
    analysis = json.loads((PREREG / "analysis_clips.json").read_text())["clips"]
    rows = [json.loads((eval_out / c["clip"] / "compare_float32.json").read_text())["accuracy_cpu"] for c in calibration]
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("depth_absrel", "ate", "rpe")}
    result = {"margins": {"depth_absrel": max(0.01, 0.10 * mean["depth_absrel"]), "ate": 0.05 * mean["ate"],
                          "rpe": 0.05 * mean["rpe"]},
              "calibration_cpu_means": mean,
              "perf_clip_frames": int(np.ceil(np.percentile([c["frames"] for c in analysis], 90))),
              "jf_floor": round(JF_PUBLISHED - 0.03, 4), "jf_floor_source": JF_SOURCE,
              "computed_from": {"calibration_clips_sha256": sha256(PREREG / "calibration_clips.json"),
                                "cpu_compare_sha256": {c["clip"]: sha256(eval_out / c["clip"] / "compare_float32.json") for c in calibration}}}
    (PREREG / "margins.json").write_text(json.dumps(result, indent=2))
    return result


def lock():
    files = ["calibration_clips.json", "pilot_clips.json", "analysis_clips.json", "margins.json", "e2e_pairs.json"]
    extra = [REPO / "geometry/tests/mac/mps_fallback_allowlist.txt"]
    entries = {str((PREREG / f).relative_to(REPO)): sha256(PREREG / f) for f in files}
    entries.update({str(p.relative_to(REPO)): sha256(p) for p in extra})
    (PREREG / "prereg.lock").write_text(json.dumps({"stage": 2, "locked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                                    "files": entries}, indent=2) + "\n")
    print(json.dumps(entries, indent=2))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "lists":
        lists(Path(sys.argv[2]))
    elif action == "margins":
        print(json.dumps(margins(Path(sys.argv[2]), Path(sys.argv[3])), indent=2))
    else:
        lock()
