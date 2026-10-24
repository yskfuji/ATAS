"""A11 stage 1: SAM3 keyframes + EdgeTAM in between (mask hand-off, K=4), simulated, on DAVIS-2017 val by a preregistered rule.

    python tools/hybrid_val.py run     <run-id>            # per sequence: SAM3 f16, EdgeTAM alone, simulated hybrid K=4 and K=8
    python tools/hybrid_val.py analyse <run-id>            # the one decision -> evidence/<run-id>/hybrid_val.json
    python tools/hybrid_val.py run|analyse <run-id> --dry  # train sequences / synthetic results, other output names; for tests

The rule is evidence/prereg/hybrid_rule.json, pinned by evidence/prereg/hybrid_rule.lock (its sha256, committed to git
before `run`). check_lock refuses to run unless the rule matches the lock, the rule is marked locked, and every frozen
artefact, the DAVIS archive, the baseline values and the Python environment still hash as registered.
`run` prints progress only (never a hybrid number); it aborts on an invalid run; it logs every attempt.
`analyse` reads only the per-sequence results, applies the rule once, and refuses to run twice.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import tempfile
import time
import zipfile
from importlib import metadata
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools")]
import edgetam_reinit_sim as sim  # noqa: E402
import sam3_eval  # noqa: E402
from stats_tost import tost  # noqa: E402

RULE, LOCK = REPO / "evidence/prereg/hybrid_rule.json", REPO / "evidence/prereg/hybrid_rule.lock"
SAM3_BINARY = CACHE / "sam3.cpp/atas-build/atas_sam3_track"
EDGETAM_BINARY = CACHE / "sam3.cpp/atas-build-visual/atas_sam3_track"
MAX_RETRIES = 2   # identical retries of a binary failure or timeout, each logged; then the registered failure rule applies


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve(key: str) -> Path:
    return REPO.parent / key if key.startswith("cache/") else REPO / key


def environment() -> dict:
    freeze = sorted(f"{d.metadata['Name'].lower()}=={d.version}" for d in metadata.distributions())
    return {"python": platform.python_version(), "pip_freeze_sha256": hashlib.sha256("\n".join(freeze).encode()).hexdigest(),
            "os_build": subprocess.run(["sw_vers", "-buildVersion"], capture_output=True, text=True).stdout.strip(),
            "model": subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True).stdout.strip(),
            "cli_threads": 8}


def check_lock(rule: dict, dry: bool = False):
    problems = []
    if not dry:
        if rule.get("status") != "locked":
            problems.append("rule is not marked locked")
        if not LOCK.is_file() or sha256(RULE) != LOCK.read_text().split()[0]:
            problems.append("rule file does not match evidence/prereg/hybrid_rule.lock")
    problems += [f"changed: {k}" for k, v in {**rule["frozen_files"], **rule["frozen_dylibs"]}.items() if sha256(resolve(k)) != v]
    if not dry:
        if sha256(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip") != rule["davis_archive_sha256"]:
            problems.append("DAVIS archive changed")
        base = {n: json.loads((REPO / "evidence" / rule["baseline_run"] / "sam3" / f"{n}.json").read_text())["metal_f16"]["JF"] for n in rule["sequences"]}
        if hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest() != rule["baseline_values_sha256"]:
            problems.append("AC15 baseline values changed")
    env = environment()
    for k in ("python", "pip_freeze_sha256", "os_build", "model"):
        if env[k] != rule["environment"][k]:
            problems.append(f"environment differs: {k}")
    if problems:
        raise SystemExit("registered conditions not met: " + "; ".join(problems))


def log_attempt(out: Path, **fields):
    with (out / "attempts.jsonl").open("a") as f:
        f.write(json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}) + "\n")


def with_retries(out: Path, label: str, fn):
    """Binary failure / timeout = infrastructure failure: retry identically, log every attempt. Any other exception propagates."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            result = fn()
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(result["error"])
            log_attempt(out, step=label, attempt=attempt, status="ok")
            return result
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            log_attempt(out, step=label, attempt=attempt, status="infrastructure_failure", error=str(error)[-300:])
    return None


def run(run_id: str, dry: bool = False):
    rule = json.loads(RULE.read_text())
    check_lock(rule, dry)
    name_dir = "hybrid_val_dryrun" if dry else "hybrid_val"
    out = REPO / "evidence" / run_id / name_dir
    masks_dir = CACHE / "eval" / run_id / f"{name_dir}_masks"
    if out.is_dir() and any(out.glob("*.json")) and not (out / "attempts.jsonl").is_file():
        raise SystemExit(f"{out} holds results without an attempts log: refusing to reuse unlogged results")
    out.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    (out / "env.json").write_text(json.dumps({**environment(), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, indent=2))
    archive = zipfile.ZipFile(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip")
    if dry:
        from sam3_quant import train_subset
        names = train_subset(archive)[:2]
        baseline = {n: json.loads((REPO / "evidence" / run_id / "sam3_quant" / f"{n}.json").read_text())["metal_f16"]["JF"] for n in names}
    else:
        names = archive.read("DAVIS/ImageSets/2017/val.txt").decode().split()
        assert names == rule["sequences"], "val list differs from the registered list"
        baseline = {n: json.loads((REPO / "evidence" / rule["baseline_run"] / "sam3" / f"{n}.json").read_text())["metal_f16"]["JF"] for n in names}
    sam3_eval.MODELS["metal_f16"] = (CACHE / "weights/sam3cpp/sam3-f16.ggml", True)
    sam3_eval.MODELS["metal_edgetam"] = (CACHE / "weights/sam3cpp/edgetam_f16.ggml", True)
    for i, name in enumerate(names):
        target = out / f"{name}.json"
        if target.is_file():
            continue
        images, labels = sam3_eval.load_sequence(archive, name)
        result = {"sequence": name, "frames": len(images)}
        sam3_eval.BINARY = SAM3_BINARY
        s = with_retries(out, f"{name}:sam3", lambda: sam3_eval.run_sequence(images, labels, "metal_f16", save_masks=masks_dir / f"{name}.sam3.npz"))
        if s is None:
            log_attempt(out, step=name, status="INVALID_RUN", reason="SAM3 baseline failed after retries")
            raise SystemExit(f"invalid run: SAM3 baseline failed on {name}; nothing was scored")
        if abs(s["JF"] - baseline[name]) >= rule["baseline_tolerance"]:
            log_attempt(out, step=name, status="INVALID_RUN", reason="baseline not reproduced", got=s["JF"], expected=baseline[name])
            raise SystemExit(f"invalid run: SAM3 baseline for {name} is not reproduced ({s['JF']} vs {baseline[name]}); aborting before any hybrid is run")
        result["sam3"] = {k: s.get(k) for k in ("JF", "J", "F", "objects", "wall_s")}
        sam3_eval.BINARY = EDGETAM_BINARY
        e = with_retries(out, f"{name}:edgetam", lambda: sam3_eval.run_sequence(images, labels, "metal_edgetam"))
        result["edgetam"] = {k: (e or {}).get(k) for k in ("JF", "J", "F", "objects", "wall_s")}
        masks = sim.load_masks(masks_dir / f"{name}.sam3.npz")
        with tempfile.TemporaryDirectory(prefix="hybrid-png-") as tmp:
            png = []
            for j, image in enumerate(images):
                path = Path(tmp) / f"{j:05d}.png"
                Image.fromarray(image).save(path, compress_level=1)
                png.append(path)
            for k in rule["keyframe_intervals_run"]:
                score = with_retries(out, f"{name}:hybrid_k{k}", lambda k=k: {"JF": sim.simulate(images, labels, masks, k, png, True)})
                result[f"hybrid_k{k}"] = {"JF": None if score is None else score["JF"]}
        target.write_text(json.dumps(result, indent=2))
        log_attempt(out, step=name, status="written", sha256=hashlib.sha256(target.read_bytes()).hexdigest())
        print(f"done {name} ({i + 1}/{len(names)})", flush=True)   # progress only: no result is printed (no interim look)


def decide(diffs, margin, alpha):
    n, mean = len(diffs), float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    t = (np.inf if mean + margin > 0 else -np.inf) if sd == 0 else (mean + margin) / (sd / np.sqrt(n))
    p = float(1 - stats.t.cdf(t, n - 1))
    ci = tost(diffs, margin)["bca_ci90"]
    return {"n": n, "mean_diff": mean, "sd_diff": sd, "t": float(t), "p_one_sided": p, "t_noninferior": bool(p < alpha),
            "bca_ci90": ci, "bca_noninferior": bool(ci[0] > -margin)}


def analyse(run_id: str, dry: bool = False):
    rule = json.loads(RULE.read_text())
    check_lock(rule, dry)
    name_dir = "hybrid_val_dryrun" if dry else "hybrid_val"
    d = REPO / "evidence" / run_id / name_dir
    target = REPO / "evidence" / run_id / ("hybrid_val_dryrun.json" if dry else "hybrid_val.json")
    if target.exists():
        raise SystemExit(f"{target} exists: the registered analysis is run once")
    names = [p.stem for p in sorted(d.glob("*.json")) if p.stem != "attempts"] if dry else rule["sequences"]
    rows = [json.loads((d / f"{n}.json").read_text()) for n in names]
    if not dry and not (d / "attempts.jsonl").is_file():
        raise SystemExit("no attempts log")
    primary, margin, alpha = f"hybrid_k{rule['primary_k']}", rule["margin"], rule["alpha"]
    usable = [r for r in rows if (r.get("sam3") or {}).get("JF") is not None]
    valid = len(usable) == len(names)   # baseline reproduction was enforced during `run`; a missing SAM3 result is an invalid run
    diffs = np.array([((r[primary]["JF"] if r[primary]["JF"] is not None else 0.0) - r["sam3"]["JF"]) for r in usable])
    verdict = {"valid_run": bool(valid), "rule_sha256": sha256(RULE), "primary": primary, "margin": margin, "alpha": alpha,
               **(decide(diffs, margin, alpha) if len(diffs) > 2 else {}),
               "hybrid_scored_zero": [r["sequence"] for r in usable if r[primary]["JF"] is None],
               "jf_sam3_mean": float(np.mean([r["sam3"]["JF"] for r in usable])) if usable else None,
               "jf_hybrid_mean": float(np.mean([(r[primary]["JF"] or 0.0) for r in usable])) if usable else None,
               "descriptive": {"edgetam_alone_mean_diff": float(np.mean([((r["edgetam"] or {}).get("JF") or 0.0) - r["sam3"]["JF"] for r in usable])) if usable else None,
                               "hybrid_k8_mean_diff": float(np.mean([((r.get("hybrid_k8") or {}).get("JF") or 0.0) - r["sam3"]["JF"] for r in usable])) if usable else None,
                               "worst_sequence": usable[int(np.argmin(diffs))]["sequence"] if len(diffs) else None,
                               "worst_diff": float(diffs.min()) if len(diffs) else None}}
    verdict["PASS"] = None if not valid else bool(verdict["t_noninferior"] and verdict["bca_noninferior"])
    target.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    args = sys.argv[1:]
    {"run": run, "analyse": analyse}[args[0]](args[1], dry="--dry" in args)
