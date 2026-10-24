"""A12: recompute every rotation-bearing result with the corrected metric from the saved poses, keeping the old numbers.

    python tools/a12_recompute.py <run-id>

The originals were copied to *.pre_A12.json beforehand. This rewrites compare_float32.json (windowed parity, order and native
floors), the pre-SVD-fix comparison files (A7's pre-fix power check) and, through decisions.summarise, the anchored-pair
parity and decisions.json. Every parity record now carries rot_deg (stable formula) and rot_deg_pre_A12 (original formula).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "tools"), str(REPO / "geometry/backbones"), str(REPO / "geometry")]
import decisions  # noqa: E402
import geometry_eval as G  # noqa: E402


def main(run_id: str):
    eval_out = (REPO.parent / "cache/eval" / run_id).resolve()
    clips = REPO.parent / "cache/clips"
    names = [c["clip"] for c in json.loads((REPO / "evidence/prereg/analysis_clips.json").read_text())["clips"]]
    keys = ("depth_absrel", "rot_deg", "trans_frac_depth", "chamfer_frac_extent")
    for name in names:
        G.compare(clips / name, eval_out, "float32")
        base = eval_out / name
        pre = base / "compare_float32.pre_svdfix.json"
        seeds = [k for k in range(3) if (base / f"pre_svdfix_mps_float32_crn{k}" / "l0" / "cameras.npz").is_file()]
        if pre.is_file() and seeds:
            record = json.loads(pre.read_text())
            per_seed = [G.parity(base / G.run_name("cpu", "float32", "crn", k) / "l0", base / f"pre_svdfix_mps_float32_crn{k}" / "l0",
                                 base / "masks.npz") for k in seeds]
            record["parity"] = {key: max(p[key] for p in per_seed) for key in keys}
            record["parity_pre_A12"] = {"rot_deg": max(p["rot_deg_pre_A12"] for p in per_seed)}
            record["parity_by_seed"] = [{key: p[key] for key in keys} for p in per_seed]
            pre.write_text(json.dumps(record, indent=2))
    decisions.summarise(run_id, eval_out)


if __name__ == "__main__":
    main(sys.argv[1])
