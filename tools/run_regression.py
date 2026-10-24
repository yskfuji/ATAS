"""AC21: run the upstream suites twice on this Mac and record per-test outcomes.

    python tools/run_regression.py <run-id>

cuda mode: GEOMETRY_DEVICE=cuda, upstream geometry tests only (the CUDA contract; these tests need no GPU).
mps mode:  default device (mps) with ATAS_TEST_GPU=0 so GPU-gated renderer tests run on MPS, plus tests/mac.
client:    the client suite (device independent).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = REPO / ".venv-macos/bin/python"


def node_id(case, root: str) -> str:
    module = case.get("classname", "")
    parts = module.split(".")
    # pytest writes classname "tests.mac.test_x" (or with a class suffix); map back to a file path.
    for cut in range(len(parts), 0, -1):
        candidate = Path(root, *parts[:cut]).with_suffix(".py")
        if candidate.is_file():
            suffix = parts[cut:]
            return "::".join([str(candidate.relative_to(root)), *suffix, case.get("name")])
    return f"{module}::{case.get('name')}"


def run(name, cwd, extra_env, args, out):
    xml = out / f"junit-{name}.xml"
    env = {**os.environ, **extra_env}
    for key in ("GEOMETRY_DEVICE", "ATAS_TEST_GPU"):
        if key not in extra_env:
            env.pop(key, None)
    completed = subprocess.run([str(PYTHON), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-rs",
                                f"--junitxml={xml}", *args], cwd=cwd, env=env, capture_output=True, text=True)
    (out / f"pytest-{name}.log").write_text(completed.stdout + completed.stderr)
    results = {}
    for case in ET.parse(xml).getroot().iter("testcase"):
        status = "passed"
        for child in case:
            if child.tag in {"failure", "error", "skipped"}:
                status = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
        results[node_id(case, cwd)] = status
    summary = {"exit_code": completed.returncode,
               **{k: sum(1 for s in results.values() if s == v) for k, v in
                  (("passed", "passed"), ("failed", "failed"), ("errors", "error"), ("skipped", "skipped"))},
               "failed_ids": sorted(k for k, s in results.items() if s == "failed"),
               "error_ids": sorted(k for k, s in results.items() if s == "error"),
               "skipped_ids": sorted(k for k, s in results.items() if s == "skipped")}
    return summary, results


def orphan_workers(since: float) -> list[str]:
    """Test-spawned owned workers left running after the suites (reparented to launchd)."""
    import psutil
    found = []
    for process in psutil.process_iter(["pid", "ppid", "cmdline", "create_time"]):
        cmd = " ".join(process.info["cmdline"] or [])
        if process.info["ppid"] == 1 and "atas_geometry.service.isolated_worker" in cmd and process.info["create_time"] >= since:
            found.append(f"{process.info['pid']}: {cmd[:120]}")
    return found


def main(run_id: str):
    out = REPO / "evidence" / run_id
    out.mkdir(parents=True, exist_ok=True)
    import time
    started = time.time()
    geometry, client = str(REPO / "geometry"), str(REPO / "client")
    cuda, _ = run("cuda", geometry, {"GEOMETRY_DEVICE": "cuda"}, ["--ignore=tests/mac"], out)
    mps, mps_results = run("mps", geometry, {"ATAS_TEST_GPU": "0"}, [], out)
    client_summary, client_results = run("client", client, {}, [], out)
    time.sleep(5)  # let reaped groups finish exiting
    regression = {"cuda_mode": cuda, "mps_mode": mps, "client": client_summary,
                  "orphan_workers_after_run": orphan_workers(started)}
    (out / "regression.json").write_text(json.dumps(regression, indent=2) + "\n")
    junit = {"passed": {k: s == "passed" for k, s in {**mps_results,
                        **{f"client/{k}": s for k, s in client_results.items()}}.items()},
             "skipped": sorted(k for k, s in mps_results.items() if s == "skipped")}
    (out / "junit.json").write_text(json.dumps(junit, indent=2) + "\n")
    print(json.dumps({k: ({x: v[x] for x in ("passed", "failed", "errors", "skipped")} if isinstance(v, dict) else v)
                      for k, v in regression.items()}))


if __name__ == "__main__":
    main(sys.argv[1])
