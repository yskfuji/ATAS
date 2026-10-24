"""A11 pilot (exploratory, selection data only): EdgeTAM on the 20 seeded DAVIS-2017 *train* sequences of A8.

    python tools/edgetam_pilot.py <run-id> [--model edgetam_f16]

Same prompts as AC15 (first-frame GT box + one interior point), same J&F code; the SAM3 Metal f16 numbers come from
evidence/<run>/sam3_quant (A8). Timing taken while other work runs is not authoritative and is labelled as such.
Nothing here touches DAVIS val, which is reserved for the verdict of a preregistered rule.
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools")]
import sam3_eval  # noqa: E402
from sam3_quant import train_subset  # noqa: E402

VISUAL_BINARY = CACHE / "sam3.cpp/atas-build-visual/atas_sam3_track"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--model", default="edgetam_f16")
    args = parser.parse_args()
    sam3_eval.BINARY = VISUAL_BINARY
    sam3_eval.MODELS["metal_edgetam"] = (CACHE / f"weights/sam3cpp/{args.model}.ggml", True)
    archive = zipfile.ZipFile(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip")
    out = REPO / "evidence" / args.run_id / f"edgetam_pilot_{args.model}"
    out.mkdir(parents=True, exist_ok=True)
    reference = REPO / "evidence" / args.run_id / "sam3_quant"
    rows = []
    for name in train_subset(archive):
        target = out / f"{name}.json"
        if not target.is_file():
            images, labels = sam3_eval.load_sequence(archive, name)
            target.write_text(json.dumps(sam3_eval.run_sequence(images, labels, "metal_edgetam"), indent=2))
        edge = json.loads(target.read_text())
        base = json.loads((reference / f"{name}.json").read_text())["metal_f16"]
        rows.append({"sequence": name, "objects": edge.get("objects"), "frames": edge.get("frames"),
                     "jf_edgetam": edge.get("JF"), "jf_sam3": base.get("JF"),
                     "sec_per_frame_max_edgetam": edge.get("sec_per_frame_max"), "sec_per_frame_max_sam3": base.get("sec_per_frame_max"),
                     "peak_edgetam": edge.get("peak_footprint_bytes"), "peak_sam3": base.get("peak_footprint_bytes"),
                     "wall_edgetam": edge.get("wall_s"), "wall_sam3": base.get("wall_s"), "error": edge.get("error")})
        print(json.dumps(rows[-1]), flush=True)
    ok = [r for r in rows if r["jf_edgetam"] is not None]
    d = np.array([r["jf_edgetam"] - r["jf_sam3"] for r in ok])
    summary = {"model": args.model, "n": len(ok), "errors": [r["sequence"] for r in rows if r["error"]],
               "jf_edgetam_mean": float(np.mean([r["jf_edgetam"] for r in ok])), "jf_sam3_mean": float(np.mean([r["jf_sam3"] for r in ok])),
               "diff_mean": float(d.mean()), "diff_sd": float(d.std(ddof=1)), "diff_min": float(d.min()),
               "wall_ratio_sam3_over_edgetam": float(np.sum([r["wall_sam3"] for r in ok]) / np.sum([r["wall_edgetam"] for r in ok])),
               "peak_edgetam_gib_max": max(r["peak_edgetam"] for r in ok) / 2**30, "peak_sam3_gib_max": max(r["peak_sam3"] for r in ok) / 2**30,
               "timing_note": "measured while the CPU E2E reference was running; not authoritative",
               "status": "exploratory pilot on selection data (train); not a verdict"}
    (REPO / "evidence" / args.run_id / f"edgetam_pilot_{args.model}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
