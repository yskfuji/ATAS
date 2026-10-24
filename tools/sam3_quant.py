"""A8: choose the production SAM3 model among f16, q8_0 and q4_0 (sam3.cpp on Metal) by a preregistered rule.

    python tools/sam3_quant.py davis  <run-id>              # f16, q8_0, q4_0 on 20 seeded DAVIS-2017 *train* sequences
    python tools/sam3_quant.py timing <run-id> <clips-dir>  # AC22 workload, 6 interleaved fresh-process runs (quiet lane)
    python tools/sam3_quant.py select <run-id>              # -> sam3_quantisation.json

Selection data is disjoint from the AC15 verdict data (DAVIS-2017 val). Eligible: J&F equivalent to Metal f16
(paired t-TOST and BCa, margin 0.02), mean J&F >= the preregistered floor, >= 1.25x faster than f16 (worst of
runs 2-6; run 1 includes Metal compilation and is reported separately), peak footprint not above f16's.
The fastest eligible model is selected; otherwise f16.
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools"), str(REPO / "geometry"), str(REPO / "geometry/backbones")]
import sam3_eval  # noqa: E402

CANDIDATES = {"q8_0": CACHE / "weights/sam3cpp/sam3-q8_0.ggml", "q4_0": CACHE / "weights/sam3cpp/sam3-q4_0.ggml"}
ALL = {"f16": CACHE / "weights/sam3cpp/sam3-f16.ggml", **CANDIDATES}
MARGIN, SPEEDUP = 0.02, 1.25
SELECTION_SET, SEED, N_TRAIN, RUNS = "davis2017_train_20_seed20261019", 20261019, 20, 6


def train_subset(archive):
    names = sorted(archive.read("DAVIS/ImageSets/2017/train.txt").decode().split())
    return sorted(names[i] for i in np.random.default_rng(SEED).permutation(len(names))[:N_TRAIN])


def davis(run_id: str):
    for name, path in ALL.items():
        sam3_eval.MODELS[f"metal_{name}"] = (path, True)
    archive = zipfile.ZipFile(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip")
    out = REPO / "evidence" / run_id / "sam3_quant"
    out.mkdir(parents=True, exist_ok=True)
    for name in train_subset(archive):
        target = out / f"{name}.json"
        if target.is_file():
            continue
        images, labels = sam3_eval.load_sequence(archive, name)
        result = {f"metal_{n}": sam3_eval.run_sequence(images, labels, f"metal_{n}") for n in ALL}
        target.write_text(json.dumps(result, indent=2))
        print(json.dumps({name: {k: v.get("JF") for k, v in result.items()}}), flush=True)


def timing(run_id: str, clips: Path, repeats: int = RUNS):
    from utils.media import crop_and_resize_video, load_video
    from utils.recon_and_seg.seg_sam3_cpp import init_sam3_video, run_sam3_video
    analysis = [c for c in json.loads((REPO / "evidence/prereg/analysis_clips.json").read_text())["clips"] if c["dataset"] == "TUM RGB-D"]
    video, _ = load_video(str(clips / analysis[0]["clip"] / "clip.mp4"))
    video = crop_and_resize_video(video, 720, 1280, resample="lanczos")
    predictors = {name: init_sam3_video(str(CACHE / "sam3.cpp/atas-build/atas_sam3_track"), str(model)) for name, model in ALL.items()}
    runs = {name: [] for name in ALL}
    margins = json.loads((REPO / "evidence/prereg/margins.json").read_text())
    if len(video) != margins["perf_clip_frames"]:
        raise ValueError(f"timing workload has {len(video)} frames, preregistered {margins['perf_clip_frames']}")
    power = __import__("subprocess").run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[:1]
    if not power or "AC Power" not in power[0]:
        raise RuntimeError(f"timing requires mains power: {power}")
    for repeat in range(repeats):          # interleaved: f16, q8_0, q4_0, f16, ... ; every call is a fresh process
        for name in ALL:
            for keyword in ("person", "sky"):
                report = {}
                run_sam3_video(video, predictors[name], [keyword], collect_instances=False, report=report)
                runs[name].append({"repeat": repeat, "keyword": keyword, "sec_per_frame": report["track_s"] / report["frames"],
                                   "sec_per_frame_max": report["sec_per_frame_max"], "peak_footprint_bytes": report["peak_footprint_bytes"]})
    rows = {}
    for name, items in runs.items():
        steady = [r for r in items if r["repeat"] > 0]
        rows[name] = {"worst_sec_per_frame": max(r["sec_per_frame"] for r in steady),
                      "sec_per_frame_max": max(r["sec_per_frame_max"] for r in steady),
                      "median_sec_per_frame": statistics.median(r["sec_per_frame"] for r in steady),
                      "first_run_sec_per_frame": max(r["sec_per_frame"] for r in items if r["repeat"] == 0),
                      "peak_footprint_bytes": max(r["peak_footprint_bytes"] for r in items), "runs": items}
        print(json.dumps({name: {k: v for k, v in rows[name].items() if k != "runs"}}), flush=True)
    (REPO / "evidence" / run_id / "sam3_quant_timing.json").write_text(json.dumps(
        {"clip": analysis[0]["clip"], "frames": len(video), "repeats": repeats, "power": power,
         "power_source": "AC Power", "models": rows}, indent=2))


def compute_selection(quant: dict, timing_report: dict, floor: float) -> dict:
    """The A8 rule as a pure function of per-sequence J&F and timing data (also run by the gate)."""
    from stats_tost import tost
    base = {s: v["metal_f16"] for s, v in quant.items()}
    report = {"rule": "A8", "selection_set": SELECTION_SET, "margin": MARGIN, "required_speedup": SPEEDUP, "jf_floor": floor,
              "speed_statistic": "worst steady-state run (runs 2-6)", "candidates": {}, "eligible": []}
    f16_time = timing_report["f16"]["worst_sec_per_frame"]
    f16_peak = timing_report["f16"]["peak_footprint_bytes"]
    for name in CANDIDATES:
        seqs = sorted(s for s in base if s in quant and "JF" in base[s] and "JF" in quant[s].get(f"metal_{name}", {}))
        diffs = np.array([quant[s][f"metal_{name}"]["JF"] - base[s]["JF"] for s in seqs])
        test = tost(diffs, MARGIN) if len(seqs) >= 3 else None
        mean_jf = float(np.mean([quant[s][f"metal_{name}"]["JF"] for s in seqs])) if seqs else None
        speedup = f16_time / timing_report[name]["worst_sec_per_frame"]
        checks = {"all_selection_sequences": len(seqs) == N_TRAIN,
                  "equivalent": bool(test and test["t_tost_equivalent"] and test["bca_bootstrap_equivalent"]),
                  "floor": mean_jf is not None and mean_jf >= floor, "speedup": speedup >= SPEEDUP,
                  "memory": timing_report[name]["peak_footprint_bytes"] <= f16_peak}
        report["candidates"][name] = {"n_sequences": len(seqs), "mean_jf": mean_jf, "tost": test, "speedup_vs_f16": speedup,
                                      "peak_footprint_bytes": timing_report[name]["peak_footprint_bytes"], "checks": checks}
        if all(checks.values()):
            report["eligible"].append(name)
    report["selected"] = max(report["eligible"], key=lambda n: report["candidates"][n]["speedup_vs_f16"]) if report["eligible"] else "f16"
    report["f16"] = {"worst_sec_per_frame": f16_time, "peak_footprint_bytes": f16_peak,
                     "mean_jf_selection_set": float(np.mean([base[s]["JF"] for s in base if "JF" in base[s]]))}
    return report


def select(run_id: str):
    run = REPO / "evidence" / run_id
    floor = json.loads((REPO / "evidence/prereg/margins.json").read_text())["jf_floor"]
    timing = json.loads((run / "sam3_quant_timing.json").read_text())
    quant = {p.stem: json.loads(p.read_text()) for p in (run / "sam3_quant").glob("*.json")}
    report = compute_selection(quant, timing["models"], floor)
    report.update(timing_frames=timing["frames"], power_source=timing["power_source"])
    (run / "sam3_quantisation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({"selected": report["selected"], "eligible": report["eligible"],
                      "candidates": {k: {"mean_jf": v["mean_jf"], "speedup": v["speedup_vs_f16"], **v["checks"]} for k, v in report["candidates"].items()}}, indent=2))


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "davis":
        davis(sys.argv[2])
    elif action == "timing":
        timing(sys.argv[2], Path(sys.argv[3]))
    else:
        select(sys.argv[2])
