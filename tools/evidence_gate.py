"""Compute every AC status from registered evidence only (docs/acceptance/acceptance.yaml).

    python tools/evidence_gate.py evidence/<run-id> [--out gate_report.json]

Nothing is read from a hand-written status. A missing, unregistered or hash-mismatched evidence
file makes its check FAIL. The overall verdict is "IN_SCOPE_PASS (BLOCKED: 7)" or "FAIL".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import guards  # noqa: E402

ACCEPTANCE = REPO / "docs/acceptance/acceptance.yaml"
LOCK = REPO / "docs/acceptance/acceptance.lock"
PREREG_LOCK = REPO / "evidence/prereg/prereg.lock"
MAC_TESTS = REPO / "geometry/tests/mac"
EXPECTED_BLOCKED = 7
IN_SCOPE = [f"AC{i:02d}" for i in list(range(1, 23)) + [25, 26]]


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args, check=True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=check)


def table(path: Path) -> list[list[str]]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            rows.append([part.strip() for part in line.split(" | ")])
    return rows


# --------------------------------------------------------------------------- statistics
def tost_power(n: int, ratio: float, alpha: float = 0.05) -> float:
    """Exact power of the paired t-TOST at true difference 0, margin 1, SD `ratio` (sigma/delta)."""
    from scipy import integrate, stats
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha, df)
    se_scale = ratio / math.sqrt(n)

    def integrand(s):
        # s is the sample SD; the mean must satisfy |mean| < 1 - t_crit * s / sqrt(n).
        half = 1 - t_crit * s / math.sqrt(n)
        if half <= 0:
            return 0.0
        density = stats.chi.pdf(s * math.sqrt(df) / ratio, df) * math.sqrt(df) / ratio
        return (2 * stats.norm.cdf(half / se_scale) - 1) * density
    upper = math.sqrt(n) / t_crit
    value, _ = integrate.quad(integrand, 0, upper, limit=200)
    return float(value)


def required_n(pilot: dict, pre: dict) -> int:
    """Smallest n with per-metric power >= 0.8^(1/k) using an 80% UCL of the pilot SD of each metric."""
    from scipy import stats
    metrics = pilot["differences"]  # {metric: [per-clip MPS - CPU differences]}
    k = len(metrics)
    target = 0.8 ** (1 / k)
    worst = 0
    for name, values in metrics.items():
        n0 = len(values)
        mean = sum(values) / n0
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n0 - 1))
        ucl = sd * math.sqrt((n0 - 1) / stats.chi2.ppf(0.20, n0 - 1))
        ratio = ucl / pre["margins"][name]
        n = 5
        while tost_power(n, ratio) < target:
            n += 1
            if n > 5000:
                raise ValueError(f"{name}: required n exceeds 5000 (ratio {ratio:.2f})")
        worst = max(worst, n)
    return max(worst, int(pre["min_analysis_clips"]))


# --------------------------------------------------------------------------- junit
def junit_matches(listed: str, passed: dict) -> list[str]:
    if listed.endswith(".py"):
        return [k for k in passed if k.split("::")[0] == listed]
    if "[" in listed:
        return [k for k in passed if k == listed]
    return [k for k in passed if k == listed or k.startswith(listed + "[")]


def check_junit(check, ev, gate):
    passed, skip_ok = ev["junit"]["passed"], set(gate["skip_allowlist"])
    problems = []
    for listed in check["tests"]:
        ids = junit_matches(listed, passed)
        if not ids:
            problems.append(f"{listed}: not in junit")
            continue
        for test_id in ids:
            if passed[test_id]:
                continue
            if listed.endswith(".py") and test_id in skip_ok and test_id in ev["junit"]["skipped"]:
                mps = test_id.rsplit("[", 1)[0] + "[mps]"
                if passed.get(mps):
                    continue
                problems.append(f"{test_id}: allowlisted CUDA variant without a passing [mps] variant")
            else:
                problems.append(f"{test_id}: not passed")
    return not problems, problems


# --------------------------------------------------------------------------- namespace
def load_run(run: Path):
    manifest_path = run / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{run}: no manifest.json")
    manifest = json.loads(manifest_path.read_text())
    registered, problems = {}, []
    for rel, digest in manifest["files"].items():
        path = run / rel
        if not path.is_file() or sha256(path) != digest:
            problems.append(f"hash mismatch or missing: {rel}")
        else:
            registered[rel] = path
    for path in run.rglob("*"):
        if path.is_file() and path.name != "manifest.json" and str(path.relative_to(run)) not in manifest["files"]:
            problems.append(f"unregistered evidence: {path.relative_to(run)}")
    ev = {}
    for rel, path in registered.items():
        if path.suffix == ".json":
            ev[rel] = json.loads(path.read_text())
    ev["manifest.json"] = manifest   # the manifest cannot register itself; the gate reads it directly
    if "junit.json" in ev:
        ev["junit"] = ev["junit.json"]
    return manifest, ev, problems


def preregistration(spec: dict) -> tuple[dict, list[str]]:
    pre, problems = dict(spec["preregistration"]), []
    if not PREREG_LOCK.is_file():
        return pre, ["preregistration stage-2 lock missing"]
    lock = json.loads(PREREG_LOCK.read_text())
    for rel, digest in lock["files"].items():
        path = REPO / rel
        key = Path(rel).stem + "_sha256"
        if not path.is_file() or sha256(path) != digest:
            problems.append(f"preregistered file changed: {rel}")
        pre[key] = digest
    pairs_file = REPO / "evidence/prereg/e2e_pairs.json"
    if pairs_file.is_file() and not any("e2e_pairs" in p for p in problems):  # only after its hash matched the lock
        world = (json.loads(pairs_file.read_text()).get("e2e_world_pair") or {})
        for key in ("new", "history", "new_sha256", "history_sha256"):
            if key in world:
                pre[f"e2e_world_pair_{key}"] = world[key]
    margins = REPO / "evidence/prereg/margins.json"
    if margins.is_file():
        for key, value in json.loads(margins.read_text()).items():
            if key in pre and not isinstance(pre[key], str):   # a string in the yaml is a description of the frozen value, not a value
                problems.append(f"preregistration key collision: {key}")
            pre[key] = value
    return pre, problems


def ordering(run: Path) -> list[str]:
    """The stage-2 lock commit must precede (be an ancestor of) the commit that added this run's manifest."""
    problems = []
    lock_commit = git("log", "--diff-filter=A", "--format=%H", "--", str(PREREG_LOCK.relative_to(REPO)), check=False).stdout.split()
    run_commit = git("log", "--diff-filter=A", "--format=%H", "--", str((run / "manifest.json").relative_to(REPO)), check=False).stdout.split()
    if not lock_commit:
        return ["prereg.lock is not committed"]
    if not run_commit:
        return ["evidence manifest is not committed"]
    lock_commit, run_commit = lock_commit[-1], run_commit[-1]
    if git("merge-base", "--is-ancestor", lock_commit, run_commit, check=False).returncode != 0:
        problems.append("preregistration lock is not an ancestor of the evidence commit")
    tree = git("ls-tree", "-r", "--name-only", lock_commit, "--", "evidence").stdout.split()
    # Only the preregistration and the pre-port upstream test baseline may precede the lock.
    if any(not t.startswith(("evidence/prereg/", "evidence/baseline/")) for t in tree):
        problems.append("MPS evidence existed in the preregistration lock commit's tree")
    return problems


def gate_values(spec, ev, pre) -> dict:
    gate = {"device_divergence_ids": [r[0] for r in table(MAC_TESTS / "device_divergence.txt")],
            "device_divergence_counterparts": [r[1] for r in table(MAC_TESTS / "device_divergence.txt")],
            "skip_allowlist": [r[0] for r in table(MAC_TESTS / "skip_allowlist.txt")],
            "skip_allowlist_cuda_mode": [r[0] for r in table(MAC_TESTS / "skip_allowlist_cuda_mode.txt")],
            "fallback_allowlist": [r[0] for r in table(MAC_TESTS / "mps_fallback_allowlist.txt")]
            if (MAC_TESTS / "mps_fallback_allowlist.txt").is_file() else []}
    named = []
    for ac in spec["acs"]:
        if ac["id"] in {"AC05", "AC06", "AC08", "AC14", "AC16", "AC17", "AC25"}:
            for check in ac["checks"]:
                if check["kind"] == "junit":
                    for listed in check["tests"]:
                        named += junit_matches(listed, ev.get("junit", {}).get("passed", {})) or [listed]
    gate["must_not_skip"] = sorted(set(named + gate["device_divergence_counterparts"]) - set(gate["skip_allowlist"]))
    if "perf_clip_frames" in pre:
        gate["sam3_sec_per_frame_budget"] = 0.5 * 1800 / float(pre["perf_clip_frames"])
    if "sam3_quant_timing.json" in ev and any(k.startswith("sam3_quant/") for k in ev):
        import sam3_quant  # the gate recomputes the A8 selection from per-sequence data
        quant = {Path(k).stem: v for k, v in ev.items() if k.startswith("sam3_quant/")}
        selection = sam3_quant.compute_selection(quant, ev["sam3_quant_timing.json"]["models"], pre["jf_floor"])
        gate["sam3_eligible"], gate["sam3_selected"] = selection["eligible"], selection["selected"]
    pilot = REPO / "evidence/prereg/pilot.json"
    if pilot.is_file() and "margins" in pre:
        gate["required_n"] = required_n(json.loads(pilot.read_text()), pre)
    return gate


def evaluate(check, ev, manifest, pre, gate):
    kind = check["kind"]
    if kind == "blocked":
        return "BLOCKED", [check["reason"]]
    if kind == "junit":
        if "junit" not in ev:
            return "FAIL", ["junit.json not registered"]
        ok, problems = check_junit(check, ev, gate)
        return ("PASS" if ok else "FAIL"), problems
    if kind == "json":
        if check["file"] not in ev:
            return "FAIL", [f"{check['file']} not registered"]
        namespace = {"d": ev[check["file"]], "ev": ev, "m": manifest, "pre": pre, "gate": gate, "set": set,
                     "all": all, "any": any, "len": len, "max": max, "min": min, "str": str}
        try:
            # One namespace as globals: names inside generator expressions are looked up in globals, not in a separate locals mapping.
            ok = bool(eval(check["assert"], {"__builtins__": {}, **namespace}))  # assertions come from the locked yaml
        except Exception as exc:  # a missing key is a failed check, never a pass
            return "FAIL", [f"{type(exc).__name__}: {exc}"]
        return ("PASS" if ok else "FAIL"), ([] if ok else [check["assert"]])
    path = check["file"].removeprefix("./")
    if kind == "ast_module":
        result = guards.ast_module(path, allow_darwin_block=bool(check.get("allow_darwin_block")))
    elif kind == "ast_identical":
        result = guards.ast_functions(path, check["functions"])
    elif kind == "ast_identical_except_device_calls":
        result = guards.ast_identical_except_device_calls(path)
    else:
        return "FAIL", [f"unknown check kind {kind}"]
    return ("PASS" if result["identical"] else "FAIL"), ([] if result["identical"] else [json.dumps(result)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    global_problems = []
    lock = json.loads(LOCK.read_text())
    for rel, digest in lock["files"].items():
        if sha256(REPO / rel) != digest:
            global_problems.append(f"locked file changed: {rel}")
    spec = yaml.safe_load(ACCEPTANCE.read_text())
    manifest, ev, problems = load_run(run)
    global_problems += problems
    pre, problems = preregistration(spec)
    global_problems += problems + ordering(run)
    gate = gate_values(spec, ev, pre)
    results, blocked_total = {}, 0
    for ac in spec["acs"]:
        outcomes = [evaluate(c, ev, manifest, pre, gate) for c in ac["checks"]]
        statuses = [s for s, _ in outcomes]
        blocked_total += statuses.count("BLOCKED")
        if "FAIL" in statuses:
            status = "FAIL"
        elif statuses and all(s == "BLOCKED" for s in statuses):
            status = "BLOCKED"
        elif "BLOCKED" in statuses:
            status = "PARTIAL"
        else:
            status = "PASS"
        results[ac["id"]] = {"title": ac["title"], "status": status,
                             "checks": [{"kind": c["kind"], "status": s, "details": d[:20]}
                                        for c, (s, d) in zip(ac["checks"], outcomes)]}
    blocked_checks = [ac["id"] for ac in spec["acs"] for c in ac["checks"] if c["kind"] == "blocked"]
    if len(blocked_checks) != EXPECTED_BLOCKED or set(blocked_checks) != {"AC23", "AC24"}:
        global_problems.append(f"blocked checks must be exactly {EXPECTED_BLOCKED} in AC23/AC24, found {blocked_checks}")
    in_scope_pass = all(results[a]["status"] == "PASS" for a in IN_SCOPE) and results["AC23"]["status"] == "PARTIAL" \
        and results["AC24"]["status"] == "BLOCKED"
    verdict = f"IN_SCOPE_PASS (BLOCKED: {blocked_total})" if in_scope_pass and not global_problems else "FAIL"
    report = {"verdict": verdict, "release": "BLOCKED", "global_problems": global_problems,
              "counts": {s: sum(1 for r in results.values() if r["status"] == s) for s in ("PASS", "FAIL", "PARTIAL", "BLOCKED")},
              "results": results, "gate_values": {k: v for k, v in gate.items() if k != "must_not_skip"},
              "acceptance_sha256": sha256(ACCEPTANCE), "run": str(run.relative_to(REPO))}
    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    print(json.dumps({"verdict": verdict, "counts": report["counts"], "global_problems": global_problems[:10]}, indent=2))
    for ac, r in results.items():
        print(f"{ac} {r['status']:8} {r['title']}")
    return 0 if verdict != "FAIL" else 1


if __name__ == "__main__":
    sys.exit(main())
