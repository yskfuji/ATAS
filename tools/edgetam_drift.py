"""A11 step 1 (train only): per-frame J of EdgeTAM and SAM3 f16 on chosen DAVIS-2017 train sequences.

    python tools/edgetam_drift.py <run-id> seq [seq ...]

Shows where in a sequence EdgeTAM falls behind SAM3, which bounds the keyframe interval K a re-detection scheme needs.
"""
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools")]
import sam3_eval  # noqa: E402

VISUAL_BINARY = CACHE / "sam3.cpp/atas-build-visual/atas_sam3_track"


def main():
    run_id, names = sys.argv[1], sys.argv[2:]
    archive = zipfile.ZipFile(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip")
    sam3_eval.MODELS["metal_edgetam"] = (CACHE / "weights/sam3cpp/edgetam_f16.ggml", True)
    out = REPO / "evidence" / run_id / "edgetam_drift"
    out.mkdir(parents=True, exist_ok=True)
    for name in names:
        target = out / f"{name}.json"
        if target.is_file():
            continue
        images, labels = sam3_eval.load_sequence(archive, name)
        result = {}
        for variant, binary in (("metal_edgetam", VISUAL_BINARY), ("metal_f16", sam3_eval.BINARY)):
            sam3_eval.MODELS.setdefault("metal_f16", (CACHE / "weights/sam3cpp/sam3-f16.ggml", True))
            original, sam3_eval.BINARY = sam3_eval.BINARY, binary
            try:
                result[variant] = sam3_eval.run_sequence(images, labels, variant, per_frame=True, save_masks=out / f"{name}.{variant}.npz")
            finally:
                sam3_eval.BINARY = original
        target.write_text(json.dumps(result))
        print(name, {v: round(r["JF"], 3) for v, r in result.items()}, flush=True)


if __name__ == "__main__":
    main()
