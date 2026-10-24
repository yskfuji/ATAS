"""A11 stage 2: the REAL hybrid (SAM3 on every K-th frame, EdgeTAM in between, mask hand-off) by a preregistered rule.

    python tools/hybrid_stage2.py run     b|a|c <run-id> [--dry] [--owner-quiet "<text>"]   # b: speed/memory; a: DAVIS-val J&F; c: mask agreement
    python tools/hybrid_stage2.py analyse b|a|c <run-id> [--dry]                             # the registered analysis (once)

Rule: evidence/prereg/hybrid_stage2_rule.json, pinned by hybrid_stage2_rule.lock. Same machinery as stage 1 (tools/hybrid_val.py):
lock check, attempts log, retries, no interim output, analyse once. Parts run strictly serially in the order b, a, c.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools"), str(REPO / "geometry/backbones"), str(REPO / "geometry")]
import hybrid_val as H1  # noqa: E402
import sam3_eval  # noqa: E402

RULE, LOCK = REPO / "evidence/prereg/hybrid_stage2_rule.json", REPO / "evidence/prereg/hybrid_stage2_rule.lock"
SAM3_BIN = CACHE / "sam3.cpp/atas-build/atas_sam3_track"
HYBRID_BIN = CACHE / "sam3.cpp/atas-build-hybrid/atas_sam3_track"
SAM3_MODEL = CACHE / "weights/sam3cpp/sam3-f16.ggml"
EDGETAM_MODEL = CACHE / "weights/sam3cpp/edgetam_f16.ggml"
DAVIS_ZIP = CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip"
sha256, log_attempt, with_retries = H1.sha256, H1.log_attempt, H1.with_retries
HYBRID_TIMEOUT_S = 3600


def environment() -> dict:
    """Like stage 1 but independent of sys.path: only distributions installed in site-packages are listed."""
    site = [p for p in sys.path if p.endswith("site-packages")]
    freeze = sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions(path=site))
    return {**H1.environment(), "pip_freeze_sha256": hashlib.sha256("\n".join(freeze).encode()).hexdigest()}


def stage1_values(rule_baseline_run: str, names) -> dict:
    d = REPO / "evidence" / rule_baseline_run / "hybrid_val"
    return {n: [json.loads((d / f"{n}.json").read_text())["sam3"]["JF"], json.loads((d / f"{n}.json").read_text())["hybrid_k4"]["JF"]] for n in names}


def check_lock(rule: dict, dry: bool = False):
    problems = []
    if not dry:
        if rule.get("status") != "locked":
            problems.append("rule is not marked locked")
        if not LOCK.is_file() or sha256(RULE) != LOCK.read_text().split()[0]:
            problems.append("rule file does not match hybrid_stage2_rule.lock")
    problems += [f"changed: {k}" for k, v in {**rule["frozen_files"], **rule["frozen_dylibs"], **rule["frozen_clips"]}.items() if sha256(H1.resolve(k)) != v]
    if sha256(DAVIS_ZIP) != rule["davis_archive_sha256"]:
        problems.append("DAVIS archive changed")
    values = stage1_values(rule["baseline_run"], rule["part_a"]["sequences"])
    if hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest() != rule["stage1_values_sha256"]:
        problems.append("stage-1 per-sequence values changed")
    env = environment()
    problems += [f"environment differs: {k}" for k in ("python", "pip_freeze_sha256", "os_build", "model") if env[k] != rule["environment"][k]]
    if problems:
        raise SystemExit("registered conditions not met: " + "; ".join(problems))


def hybrid_args(k):
    return ["--hybrid-k", str(k), "--edgetam-model", str(EDGETAM_MODEL)]


def sh(*cmd) -> str:
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


# ---------------------------------------------------------------- quiet-machine conditions (part b)
def machine_state() -> dict:
    batt, therm, rest = sh("pmset", "-g", "batt"), sh("pmset", "-g", "therm"), sh("pmset", "-g")
    load = [float(x) for x in re.findall(r"load averages?: ([\d.]+) ([\d.]+) ([\d.]+)", sh("uptime"))[0]]
    top = [ln.strip() for ln in sh("ps", "-Ao", "pcpu,comm", "-r").splitlines()[1:6]]
    return {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "ac_power": "AC Power" in batt, "low_power_mode": bool(re.search(r"lowpowermode\s+1", rest)),
            "thermal_warning": bool(re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)) and int(re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm).group(1)) < 100,
            "load_1min": load[0], "load_5min": load[1], "top_processes": top}


def wait_quiet(rule_b: dict, out: Path, label: str):
    """Before every timed run: AC power, no low-power mode, no thermal warning, 1-minute load below the ceiling (waits up to 15 min)."""
    deadline = time.time() + 900
    while True:
        state = machine_state()
        ok = state["ac_power"] and not state["low_power_mode"] and not state["thermal_warning"] and state["load_1min"] <= rule_b["load_ceiling_1min"]
        log_attempt(out, step=f"{label}:machine", **state, quiet=ok)
        if ok:
            return state
        if time.time() > deadline:
            log_attempt(out, step=label, status="INVALID_RUN", reason="machine not quiet")
            raise SystemExit(f"invalid run: machine not quiet before {label}: {state}")
        time.sleep(30)


# ---------------------------------------------------------------- a: DAVIS val J&F of the real hybrid
def run_a(run_id, rule, dry, quiet_note=None):
    out = REPO / "evidence" / run_id / ("hybrid_stage2a_dryrun" if dry else "hybrid_stage2a")
    if out.is_dir() and any(out.glob("*.json")) and not (out / "attempts.jsonl").is_file():
        raise SystemExit("results without an attempts log")
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps({**environment(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    archive = zipfile.ZipFile(DAVIS_ZIP)
    names = rule["sequences"] if not dry else rule["sequences"][:3]
    base = stage1_values(rule["baseline_run"], rule["sequences"])
    sam3_eval.MODELS["metal_f16"] = (SAM3_MODEL, True)
    sam3_eval.BINARY = HYBRID_BIN
    # Abort-before-scoring precondition: the hybrid build itself, without --hybrid-k, must reproduce the stored SAM3 baseline.
    for name in rule["baseline_check_sequences"][: (1 if dry else None)]:
        flag = out / f"baseline_check_{name}.ok"
        if flag.is_file():
            continue
        images, labels = sam3_eval.load_sequence(archive, name)
        r = with_retries(out, f"{name}:baseline_check", lambda: sam3_eval.run_sequence(images, labels, "metal_f16", timeout=HYBRID_TIMEOUT_S * 3))
        if r is None or abs(r["JF"] - base[name][0]) >= rule["baseline_tolerance"]:
            log_attempt(out, step=name, status="INVALID_RUN", reason="hybrid build does not reproduce the stored SAM3 baseline", got=(r or {}).get("JF"), expected=base[name][0])
            raise SystemExit(f"invalid run: hybrid build does not reproduce the SAM3 baseline on {name}")
        flag.write_text("ok")
    for i, name in enumerate(names):
        target = out / f"{name}.json"
        if target.is_file():
            continue
        images, labels = sam3_eval.load_sequence(archive, name)
        r = with_retries(out, f"{name}:hybrid", lambda: sam3_eval.run_sequence(images, labels, "metal_f16", extra_args=hybrid_args(rule["k"]), timeout=HYBRID_TIMEOUT_S))
        result = {"sequence": name, "frames": len(images), "hybrid": {"JF": None if r is None else r["JF"], "wall_s": (r or {}).get("wall_s"),
                  "peak_footprint_bytes": (r or {}).get("peak_footprint_bytes")}}
        target.write_text(json.dumps(result, indent=2))
        log_attempt(out, step=name, status="written", sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        print(f"done {name} ({i + 1}/{len(names)})", flush=True)


def analyse_a(run_id, rule, dry):
    d = REPO / "evidence" / run_id / ("hybrid_stage2a_dryrun" if dry else "hybrid_stage2a")
    target = REPO / "evidence" / run_id / ("hybrid_stage2a_dryrun.json" if dry else "hybrid_stage2a.json")
    if target.exists():
        raise SystemExit(f"{target} exists: the registered analysis is run once")
    attempts = d / "attempts.jsonl"
    if not attempts.is_file():
        raise SystemExit("no attempts log")
    written = {}
    for line in attempts.read_text().splitlines():
        row = json.loads(line) if line.strip() else {}
        if row.get("status") == "written":
            written[row["step"]] = row["sha256"]
    names = [p.stem for p in sorted(d.glob("*.json")) if p.stem not in ("attempts", "env")] if dry else rule["sequences"]
    if not dry and len(names) != 30:
        raise SystemExit("n must be 30")
    for n in names:
        if hashlib.sha256((d / f"{n}.json").read_bytes()).hexdigest() != written.get(n):
            raise SystemExit(f"{n}.json does not match its logged sha256 (edited or re-rolled)")
    rows = [json.loads((d / f"{n}.json").read_text()) for n in names]
    stage1 = stage1_values(rule["baseline_run"], names)
    diffs = np.array([((r["hybrid"]["JF"] if r["hybrid"]["JF"] is not None else 0.0) - stage1[r["sequence"]][0]) for r in rows])
    verdict = {"rule_sha256": sha256(RULE), **H1.decide(diffs, rule["margin"], rule["alpha"]),
               "hybrid_scored_zero": [r["sequence"] for r in rows if r["hybrid"]["JF"] is None],
               "jf_sam3_mean": float(np.mean([stage1[n][0] for n in names])), "jf_hybrid_mean": float(np.mean([r["hybrid"]["JF"] or 0.0 for r in rows]))}
    gap = np.array([(r["hybrid"]["JF"] or 0.0) - stage1[r["sequence"]][1] for r in rows])
    verdict["descriptive_vs_simulation"] = {"mean_real_minus_sim": float(gap.mean()), "sd": float(gap.std(ddof=1)) if len(gap) > 1 else None,
                                            "min": float(gap.min()), "max": float(gap.max())}
    verdict["PASS"] = bool(verdict["t_noninferior"] and verdict["bca_noninferior"])
    target.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


# ---------------------------------------------------------------- shared: frames of a clip, tracker runs
def write_clip_frames(clip: str, frames: int, folder: Path):
    from utils.media import crop_and_resize_video, load_video
    video, _ = load_video(str(CACHE / "clips" / clip / "clip.mp4"))
    video = crop_and_resize_video(video, 720, 1280, resample="lanczos")[:frames]
    folder.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(video):
        Image.fromarray(f).save(folder / f"{i:05d}.png", compress_level=1)
    return len(video)


def track(binary, model, frames_dir, out_dir, prompt, extra=()):
    command = [str(binary), "--model", str(model), "--frames", str(frames_dir), "--out", str(out_dir), "--prompt", prompt, "--threads", "8", *extra]
    tick = time.perf_counter()
    done = subprocess.run(command, capture_output=True, text=True, timeout=3 * 3600)
    wall = time.perf_counter() - tick
    if done.returncode != 0:
        raise RuntimeError(done.stderr[-1000:])
    records = [json.loads(l) for l in (out_dir / "tracks.jsonl").read_text().splitlines() if l.strip()]
    return wall, next(r["summary"] for r in records if "summary" in r), records


def frames_with_detection(records) -> int:
    return len({int(r["frame"]) for r in records if "mask" in r})


# ---------------------------------------------------------------- b: speed and memory
def run_b(run_id, rule, dry, quiet_note=None):
    out = REPO / "evidence" / run_id / ("hybrid_stage2b_dryrun" if dry else "hybrid_stage2b")
    if out.is_dir() and any(out.iterdir()):
        raise SystemExit(f"{out} is not empty: part b is run once; a repeat is a deviation and is not started automatically")
    if not dry and not quiet_note:
        raise SystemExit("part b needs --owner-quiet '<text>' (the owner's confirmation that other applications are closed)")
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps({**environment(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "owner_quiet_confirmation": quiet_note,
                                              "machine_at_start": machine_state()}, indent=2))
    repeats = 2 if dry else rule["repeats"]
    with tempfile.TemporaryDirectory(prefix="stage2b-") as tmp:
        tmp = Path(tmp)
        n = write_clip_frames(rule["clip"], rule["frames"], tmp / "frames")
        for r in range(repeats):
            for variant in (("sam3", "hybrid") if r % 2 == 0 else ("hybrid", "sam3")):
                label = f"{variant}:{r}"
                args = (SAM3_BIN, ()) if variant == "sam3" else (HYBRID_BIN, hybrid_args(rule["k"]))
                before = wait_quiet(rule, out, label)
                try:
                    wall, summary, records = track(args[0], SAM3_MODEL, tmp / "frames", tmp / f"o_{variant}_{r}", rule["prompt"], args[1])
                except (RuntimeError, subprocess.TimeoutExpired) as error:
                    log_attempt(out, step=label, status="INVALID_RUN", error=str(error)[-300:])
                    raise SystemExit(f"invalid run: a timing run failed ({label}); timings so far are kept in runs.jsonl")
                row = {"variant": variant, "run": r, "wall_s": wall, "track_s": summary["track_s"], "load_s": summary["load_s"],
                       "peak_footprint_bytes": summary["peak_footprint_bytes"], "detections": summary["detections"],
                       "frames_with_detection": frames_with_detection(records), "handoff_instances": summary.get("handoff_instances"),
                       "handoff_failed": summary.get("handoff_failed"), "empty_keyframes": summary.get("empty_keyframes"),
                       "keyframes": summary.get("keyframes"), "machine_before": before, "machine_after": machine_state()}
                with (out / "runs.jsonl").open("a") as f:     # append-only, written as soon as the run ends
                    f.write(json.dumps(row) + "\n")
                print(f"done {label}", flush=True)
    (out / "complete").write_text(json.dumps({"frames": n, "repeats": repeats}))


def analyse_b(run_id, rule, dry):
    d = REPO / "evidence" / run_id / ("hybrid_stage2b_dryrun" if dry else "hybrid_stage2b")
    target = REPO / "evidence" / run_id / ("hybrid_stage2b_dryrun.json" if dry else "hybrid_stage2b.json")
    if target.exists():
        raise SystemExit(f"{target} exists: the registered analysis is run once")
    if not (d / "complete").is_file():
        raise SystemExit("part b is not complete")
    runs = [json.loads(l) for l in (d / "runs.jsonl").read_text().splitlines() if l.strip()]
    steady = lambda v: [r for r in runs if r["variant"] == v and r["run"] >= 1]
    med = lambda v, k: float(statistics.median(r[k] for r in steady(v)))
    speedup_median = med("sam3", "wall_s") / med("hybrid", "wall_s")
    speedup_worst = min(r["wall_s"] for r in steady("sam3")) / max(r["wall_s"] for r in steady("hybrid"))
    peak_med = {v: med(v, "peak_footprint_bytes") for v in ("sam3", "hybrid")}
    peak_max = {v: max(r["peak_footprint_bytes"] for r in runs if r["variant"] == v) for v in ("sam3", "hybrid")}
    valid_output = med("hybrid", "frames_with_detection") >= rule["min_detection_frame_ratio"] * med("sam3", "frames_with_detection")
    verdict = {"rule_sha256": sha256(RULE), "n_runs_per_variant": len([r for r in runs if r["variant"] == "sam3"]),
               "wall_s_median_steady": {v: med(v, "wall_s") for v in ("sam3", "hybrid")},
               "first_run_wall_s": {v: next(r["wall_s"] for r in runs if r["variant"] == v and r["run"] == 0) for v in ("sam3", "hybrid")},
               "speedup_median_steady": speedup_median, "speedup_worst_case": speedup_worst,
               "peak_footprint_bytes_median_steady": peak_med, "peak_footprint_bytes_max_all_runs": peak_max,
               "peak_ratio_median_steady": peak_med["hybrid"] / peak_med["sam3"], "peak_ratio_max_all_runs": peak_max["hybrid"] / peak_max["sam3"],
               "output_valid": bool(valid_output),
               "frames_with_detection_median": {v: med(v, "frames_with_detection") for v in ("sam3", "hybrid")},
               "handoff": {k: [r.get(k) for r in runs if r["variant"] == "hybrid"] for k in ("handoff_instances", "handoff_failed", "empty_keyframes", "keyframes")},
               "detections": {v: [r["detections"] for r in runs if r["variant"] == v] for v in ("sam3", "hybrid")},
               "speedup_threshold": rule["speedup_min"], "peak_ratio_max": rule["peak_ratio_max"]}
    verdict["PASS"] = bool(speedup_median >= rule["speedup_min"] and verdict["peak_ratio_median_steady"] <= rule["peak_ratio_max"] and valid_output)
    verdict["not_demonstrated"] = bool(not valid_output)   # an invalid hybrid output is 'not demonstrated', not a speed result
    target.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


# ---------------------------------------------------------------- c: mask agreement on ten clips (descriptive; agreement is not accuracy)
def run_c(run_id, rule, dry, quiet_note=None):
    import cv2
    out = REPO / "evidence" / run_id / ("hybrid_stage2c_dryrun" if dry else "hybrid_stage2c")
    out.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps({**environment(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    clips = rule["repeat_control_clips"][:1] if dry else rule["clips"]
    kernel = np.ones((5, 5), np.uint8)    # the dilation geometry_eval.static_mask applies to dynamic masks downstream
    for i, clip in enumerate(clips):
        target = out / f"{clip}.json"
        if target.is_file():
            continue
        with tempfile.TemporaryDirectory(prefix="stage2c-") as tmp:
            tmp = Path(tmp)
            n = write_clip_frames(clip, rule["frames"], tmp / "frames")
            row = {"clip": clip, "frames": n}
            variants = [("sam3", SAM3_BIN, ()), ("hybrid", HYBRID_BIN, hybrid_args(rule["k"]))] + ([("sam3_repeat", SAM3_BIN, ())] if clip in rule["repeat_control_clips"] else [])
            masks = {}
            for variant, binary, extra in variants:
                res = with_retries(out, f"{clip}:{variant}", lambda b=binary, e=extra, v=variant: (lambda t: {"wall": t[0], "summary": t[1], "records": t[2]})(
                    track(b, SAM3_MODEL, tmp / "frames", tmp / f"o_{v}", rule["prompt"], e)))
                if res is None:
                    raise SystemExit(f"invalid run: {clip} {variant} failed after retries")
                stack = np.zeros((n, 720, 1280), bool)
                for rec in res["records"]:
                    if "mask" in rec:
                        stack[int(rec["frame"])] |= np.asarray(Image.open(tmp / f"o_{variant}" / rec["mask"])) > 127
                masks[variant] = stack
                row[f"{variant}_wall_s"] = res["wall"]
                row[f"{variant}_summary"] = {k: v for k, v in res["summary"].items() if k in ("detections", "keyframes", "handoff_instances", "handoff_failed", "empty_keyframes")}

            def iou_frames(a, b):
                inter, union = (a & b).sum((1, 2)), (a | b).sum((1, 2))
                return [float(x / y) if y else None for x, y in zip(inter, union)]
            s, h = masks["sam3"], masks["hybrid"]
            row["iou_per_frame"] = iou_frames(s, h)
            row["keyframe_flags"] = [f % rule["k"] == 0 for f in range(n)]
            row["sam3_dynamic_frac"], row["hybrid_dynamic_frac"] = float(s.mean()), float(h.mean())
            row["missed_fraction"] = float((s & ~h).sum() / max(s.sum(), 1))
            row["extra_fraction"] = float((h & ~s).sum() / max(h.sum(), 1))
            dil = np.stack([cv2.dilate(m.astype(np.uint8), kernel).astype(bool) for m in h])
            row["missed_fraction_after_5x5_dilation"] = float((s & ~dil).sum() / max(s.sum(), 1))
            if "sam3_repeat" in masks:
                row["repeat_control_iou_per_frame"] = iou_frames(s, masks["sam3_repeat"])
            target.write_text(json.dumps(row, indent=2))
        log_attempt(out, step=clip, status="written")
        print(f"done {clip} ({i + 1}/{len(clips)})", flush=True)


def analyse_c(run_id, rule, dry):
    d = REPO / "evidence" / run_id / ("hybrid_stage2c_dryrun" if dry else "hybrid_stage2c")
    target = REPO / "evidence" / run_id / ("hybrid_stage2c_dryrun.json" if dry else "hybrid_stage2c.json")
    if target.exists():
        raise SystemExit(f"{target} exists: the registered analysis is run once")
    rows = [json.loads(p.read_text()) for p in sorted(d.glob("*.json")) if p.stem not in ("attempts", "env")]
    mean = lambda xs: float(np.mean([v for v in xs if v is not None])) if any(v is not None for v in xs) else None
    per_clip = {}
    for r in rows:
        key = np.array(r["keyframe_flags"])
        iou = np.array([np.nan if v is None else v for v in r["iou_per_frame"]], float)
        per_clip[r["clip"]] = {"mean_iou": mean(r["iou_per_frame"]), "mean_iou_keyframes": float(np.nanmean(iou[key])) if np.isfinite(iou[key]).any() else None,
                               "mean_iou_non_keyframes": float(np.nanmean(iou[~key])) if np.isfinite(iou[~key]).any() else None,
                               "repeat_control_mean_iou": mean(r.get("repeat_control_iou_per_frame", [])) if "repeat_control_iou_per_frame" in r else None,
                               "missed_fraction": r["missed_fraction"], "missed_fraction_after_5x5_dilation": r["missed_fraction_after_5x5_dilation"],
                               "extra_fraction": r["extra_fraction"], "sam3_dynamic_frac": r["sam3_dynamic_frac"], "hybrid_dynamic_frac": r["hybrid_dynamic_frac"],
                               "sam3_wall_s": r["sam3_wall_s"], "hybrid_wall_s": r["hybrid_wall_s"], "hybrid_summary": r["hybrid_summary"]}
    ious = [v["mean_iou"] for v in per_clip.values() if v["mean_iou"] is not None]
    verdict = {"rule_sha256": sha256(RULE), "status": "descriptive, no decision; agreement with SAM3 is not accuracy", "n_clips": len(rows), "per_clip": per_clip,
               "mean_of_clip_mean_iou": float(np.mean(ious)) if ious else None, "min_clip_mean_iou": float(np.min(ious)) if ious else None,
               "mean_missed_fraction": float(np.mean([v["missed_fraction"] for v in per_clip.values()])),
               "mean_missed_fraction_after_5x5_dilation": float(np.mean([v["missed_fraction_after_5x5_dilation"] for v in per_clip.values()])),
               "mean_extra_fraction": float(np.mean([v["extra_fraction"] for v in per_clip.values()]))}
    target.write_text(json.dumps(verdict, indent=2))
    print(json.dumps({k: v for k, v in verdict.items() if k != "per_clip"}, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    action, part, run_id, dry = args[0], args[1], args[2], "--dry" in args
    quiet = args[args.index("--owner-quiet") + 1] if "--owner-quiet" in args else None
    full = json.loads(RULE.read_text())
    check_lock(full, dry)
    rule = {**full[{"a": "part_a", "b": "part_b", "c": "part_c"}[part]], "baseline_run": full["baseline_run"],
            "baseline_tolerance": full["baseline_tolerance"], "baseline_check_sequences": full["baseline_check_sequences"]}
    fn = {("run", "a"): run_a, ("run", "b"): run_b, ("run", "c"): run_c}.get((action, part))
    (fn(run_id, rule, dry, quiet) if fn else {("analyse", "a"): analyse_a, ("analyse", "b"): analyse_b, ("analyse", "c"): analyse_c}[(action, part)](run_id, rule, dry))
