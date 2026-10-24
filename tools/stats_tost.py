"""AC12/AC13 aggregation: per-clip parity limits, paired t-TOST and a BCa cluster bootstrap.

    python tools/stats_tost.py pilot    <eval-out>             -> evidence/prereg/pilot.json
    python tools/stats_tost.py analyse  <run-id> <eval-out> --dtype float32

The clip is the unit everywhere (frames are never pooled). Margins are read from the locked
evidence/prereg/margins.json; the primary decision is the paired t-TOST (Schuirmann 1987), the BCa
bootstrap of the mean difference by clip (10000 resamples, seed 20261019) is the sensitivity check.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
PREREG = REPO / "evidence/prereg"
METRICS = ("depth_absrel", "ate", "rpe")
ALPHA, SEED = 0.05, 20261019
LIMITS = {"float32": {"depth_absrel": 1e-3, "rot_deg": .05, "trans_frac_depth": 1e-3, "chamfer_frac_extent": 1e-3},
          "reduced": {"depth_absrel": 1e-2, "rot_deg": .25, "trans_frac_depth": 5e-3, "chamfer_frac_extent": 5e-3}}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clips(name):
    return [c["clip"] for c in json.loads((PREREG / f"{name}_clips.json").read_text())["clips"]]


def compares(eval_out: Path, names, dtype):
    return {n: json.loads((eval_out / n / f"compare_{dtype}.json").read_text()) for n in names}


def differences(rows):
    return {m: [r["accuracy_mps"][m] - r["accuracy_cpu"][m] for r in rows.values()] for m in METRICS}


def tost(diffs, margin):
    d = np.asarray(diffs, float)
    n, mean, sd = len(d), float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n) if sd > 0 else 1e-300
    t_lower, t_upper = (mean + margin) / se, (mean - margin) / se
    p_lower, p_upper = 1 - stats.t.cdf(t_lower, n - 1), stats.t.cdf(t_upper, n - 1)
    boot = stats.bootstrap((d,), np.mean, confidence_level=1 - 2 * ALPHA, n_resamples=10000, method="BCa",
                           random_state=np.random.default_rng(SEED))
    low, high = float(boot.confidence_interval.low), float(boot.confidence_interval.high)
    if not np.isfinite(low) or not np.isfinite(high):  # degenerate (all differences equal): the interval is the point
        low = high = mean
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "margin": margin, "t_lower": float(t_lower), "t_upper": float(t_upper),
            "p_lower": float(p_lower), "p_upper": float(p_upper), "t_tost_equivalent": bool(max(p_lower, p_upper) < ALPHA),
            "bca_ci90": [low, high], "bca_bootstrap_equivalent": bool(-margin < low and high < margin)}


def pilot(eval_out: Path):
    rows = compares(eval_out, clips("pilot"), "float32")
    result = {"clips": list(rows), "differences": differences(rows), "dtype": "float32"}
    (PREREG / "pilot.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["differences"]))


def analyse(run_id: str, eval_out: Path, dtype: str, selected: str):
    out = REPO / "evidence" / run_id
    (out / "stats").mkdir(parents=True, exist_ok=True)
    names = clips("analysis")
    margins = json.loads((PREREG / "margins.json").read_text())["margins"]
    rows = compares(eval_out, names, dtype)
    diffs = differences(rows)
    report = {"dtype": dtype, "n_clips": len(rows), "unit": "clip", "margins_sha256": sha256(PREREG / "margins.json"),
              "analysis_clips_sha256": sha256(PREREG / "analysis_clips.json"),
              "metrics": {m: tost(diffs[m], margins[m]) for m in METRICS},
              "per_clip": {n: {"cpu": r["accuracy_cpu"], "mps": r["accuracy_mps"]} for n, r in rows.items()}}
    (out / "stats" / "tost.json").write_text(json.dumps(report, indent=2))
    limits = LIMITS["float32" if dtype == "float32" else "reduced"]
    per_clip = {n: {k: r["parity"][k] for k in limits} for n, r in rows.items()}
    original = {n: [k for k, v in p.items() if not v <= limits[k]] for n, p in per_clip.items()}

    def terms(r, k):  # A7 (round 6): rotation may use a capped noise term; other quantities only L and 2*O
        order = (r.get("order_floor_cpu_10_vs_4_threads") or {}).get(k)
        noise = (r.get("native_noise_floor_median") or {}).get(k)
        if order is None or (k == "rot_deg" and noise is None):
            return None
        t = {"L": limits[k], "2O": 2 * order}
        if k == "rot_deg":
            t["N"] = min(0.1 * noise, LIMITS["reduced"]["rot_deg"])
        return t

    def judge(source_rows, source_parity):
        out, only_n, setters = {}, [], {}
        for n, p in source_parity.items():
            bad = []
            for k, v in p.items():
                t = terms(source_rows.get(n, rows.get(n, {})), k)
                if t is None or v is None or not v <= max(t.values()):
                    bad.append(k); continue
                setters.setdefault(n, {})[k] = max(t, key=t.get)
                if not v <= max(t["L"], t["2O"]):
                    only_n.append(n)
            out[n] = bad
        return out, sorted(set(only_n)), setters
    failures, only_via_n, setters = judge(rows, per_clip)
    amended = {n: {k: (max(terms(rows[n], k).values()) if terms(rows[n], k) else None) for k in limits} for n in rows}
    # Power check (round 7): judge only clips with a real pre-fix measurement; the rule must keep failing the
    # worst pre-fix clip and must not rescue every clip the original limits rejected.
    pre = {}
    for n in names:
        path = eval_out / n / f"compare_{dtype}.pre_svdfix.json"
        values = (json.loads(path.read_text()).get("parity") or {}) if path.is_file() else {}
        if all(values.get(k) is not None for k in limits):
            pre[n] = {k: values[k] for k in limits}
    pre_failures, _, _ = judge(rows, pre)
    pre_failed_original = [n for n, p in pre.items() if any(p[k] > limits[k] for k in limits)]
    pre_rescued = [n for n in pre_failed_original if not pre_failures[n]]
    worst = max(pre, key=lambda n: pre[n]["rot_deg"] / limits["rot_deg"]) if pre else None
    anchored = {}
    anchored_file = eval_out / f"anchored_parity_{dtype}.json"
    if anchored_file.is_file():
        anchored = json.loads(anchored_file.read_text())
        for n, p in anchored["per_pair"].items():
            failures[f"anchored:{n}"] = [k for k, v in p.items() if k in limits and not v <= limits[k]]
    # A12: the same rules with the pre-A12 (arccos) rotation numbers, reported side by side and never used for the verdict.
    def old_row(r):
        o = dict(r)
        o["order_floor_cpu_10_vs_4_threads"] = {**(r.get("order_floor_cpu_10_vs_4_threads") or {}), **(r.get("order_floor_cpu_10_vs_4_threads_pre_A12") or {})}
        o["native_noise_floor_median"] = {**(r.get("native_noise_floor_median") or {}), **(r.get("native_noise_floor_median_pre_A12") or {})}
        return o
    rows_old = {n: old_row(r) for n, r in rows.items()}
    per_clip_old = {n: {**p, "rot_deg": rows[n]["parity_pre_A12"]["rot_deg"]} for n, p in per_clip.items()}
    original_old = {n: [k for k, v in p.items() if not v <= limits[k]] for n, p in per_clip_old.items()}
    failures_old, _, _ = judge(rows_old, per_clip_old)
    if anchored:
        for n, p in anchored["per_pair"].items():
            failures_old[f"anchored:{n}"] = [k for k, v in {**p, "rot_deg": p["rot_deg_pre_A12"]}.items() if k in limits and not v <= limits[k]]
    pre_A12 = {"rotation_metric": "arccos (original)", "all_clips_pass": not any(failures_old.values()),
               "failures": {k: v for k, v in failures_old.items() if v}, "all_clips_pass_original": not any(original_old.values()),
               "failures_original": {k: v for k, v in original_old.items() if v},
               "worst_rot_deg": max(p["rot_deg"] for p in per_clip_old.values()),
               "order_floor_rot_deg_median_over_clips": float(np.median([r["order_floor_cpu_10_vs_4_threads_pre_A12"]["rot_deg"] for r in rows.values()])),
               "native_noise_rot_deg_median_over_clips": float(np.median([r["native_noise_floor_median_pre_A12"]["rot_deg"] for r in rows.values()]))}
    consistent = all(r.get("crn_installed_all") and r.get("crn_draw_sequences_equal") for r in rows.values())
    parity = {"dtype": dtype, "selected_dtype": selected, "n_clips": len(rows), "summary": "per_clip_max_over_frames",
              "random_numbers": "common" if consistent else "inconsistent", "crn_seeds_per_clip": [len(r["seeds"]) for r in rows.values()],
              "native_noise_floor_worst": {k: max((r.get("native_noise_floor_cpu_vs_cpu") or {}).get(k, 0) for r in rows.values())
                                           for k in ("depth_absrel", "rot_deg", "trans_frac_depth", "chamfer_frac_extent")},
              "units": "raw", "normaliser_mask": "l1_static", "limits": limits,
              "pi3x_settings": next(iter(rows.values()))["settings"],
              "clip_list_sha256": sha256(PREREG / "analysis_clips.json"),
              "paths": sorted(["windowed"] + (["anchored"] if anchored else [])),
              "anchored_pairs": len(anchored.get("per_pair", {})),
              "failures": {k: v for k, v in failures.items() if v},
              "all_clips_pass": not any(failures.values()),
              "rule": "A7 round 6: max(L,2O) for depth/trans/chamfer; max(L,2O,min(0.1N,0.25deg)) for rotation",
              "amended_limits": amended, "limit_set_by": setters, "clips_passing_only_via_N": len(only_via_n),
              "clips_passing_only_via_N_list": only_via_n,
              "pre_svdfix_measured_clips": len(pre), "pre_svdfix_failed_original": pre_failed_original,
              "pre_svdfix_rescued": pre_rescued, "pre_svdfix_worst_clip": worst,
              "pre_svdfix_worst_value_deg": pre[worst]["rot_deg"] if worst else None,
              "pre_svdfix_worst_fails": bool(worst and pre_failures[worst]),
              "order_floor_seeds": [r.get("order_floor_seeds", 0) for r in rows.values()],
              "failures_original": {k: v for k, v in original.items() if v},
              "all_clips_pass_original": not any(original.values()),
              "order_floor_clips": sum(1 for r in rows.values() if r.get("order_floor_cpu_10_vs_4_threads")),
              "noise_floor_clips": sum(1 for r in rows.values() if r.get("native_noise_floor_median")),
              "worst": {k: max(p[k] for p in per_clip.values()) for k in limits},
              "rotation_metric": "atan2_A12", "pre_A12": pre_A12,
              "fp16_storage_floor_note": "depth compared as stored float16 EXR: per-pixel rounding floor ~4.9e-4 relative",
              "per_clip": per_clip}
    (out / "parity.json").write_text(json.dumps(parity, indent=2))
    print(json.dumps({"tost": {m: {k: report["metrics"][m][k] for k in ("mean_diff", "margin", "t_tost_equivalent",
                     "bca_bootstrap_equivalent")} for m in METRICS}, "parity_pass": parity["all_clips_pass"],
                      "worst": parity["worst"]}, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "pilot":
        pilot(Path(sys.argv[2]))
    else:
        dtype = sys.argv[sys.argv.index("--dtype") + 1] if "--dtype" in sys.argv else "float32"
        selected = sys.argv[sys.argv.index("--selected") + 1] if "--selected" in sys.argv else dtype
        analyse(sys.argv[2], Path(sys.argv[3]), dtype, selected)
