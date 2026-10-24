"""Evidence runs: a run directory, a live hardware fingerprint, and a sha256 manifest.

    python tools/evidence.py start  <run-id>          # creates evidence/<run-id>/, writes hw.json
    python tools/evidence.py finish <run-id> [--weights ...]  # hashes every file, writes manifest.json

Only files registered in manifest.json count as evidence (tools/evidence_gate.py re-hashes them).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
UPSTREAM = "b86aab046368"
PYTHON = REPO / ".venv-macos/bin/python"
WEIGHTS = {
    "pi3x/model.safetensors": "https://huggingface.co/yyfz233/Pi3X/resolve/main/model.safetensors",
    "pi3x/config.json": "https://huggingface.co/yyfz233/Pi3X/resolve/main/config.json",
    "sam3cpp/sam3-f16.ggml": "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-f16.ggml",
    "sam3cpp/sam3-f32.ggml": "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-f32.ggml",
    "sam3cpp/sam3-q8_0.ggml": "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-q8_0.ggml",
    "sam3cpp/sam3-q4_0.ggml": "https://huggingface.co/PABannier/sam3.cpp/resolve/main/sam3-q4_0.ggml",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True).stdout.strip()


def hardware() -> dict:
    """Captured live; deliberately excludes serial numbers and hardware UUIDs."""
    displays = json.loads(subprocess.run(["system_profiler", "SPDisplaysDataType", "-json"],
                                         capture_output=True, text=True).stdout or "{}")
    gpus = [{"model": d.get("sppci_model"), "cores": d.get("sppci_cores"),
             "metal": d.get("spdisplays_mtlgpufamilysupport")} for d in displays.get("SPDisplaysDataType", [])]
    from importlib import import_module
    sys.path.insert(0, str(REPO / "geometry"))
    probe = import_module("atas_geometry.service.unified_pool").probe_accelerator(
        str(PYTHON), "mps", env={**os.environ, **import_module("atas_geometry.device").worker_env(0)})
    sw = {k: subprocess.run(["sw_vers", f"-{k}"], capture_output=True, text=True).stdout.strip()
          for k in ("productName", "productVersion", "buildVersion")}
    return {"captured_at": time.time(), "cpu_brand": sysctl("machdep.cpu.brand_string"),
            "hw_model": sysctl("hw.model"), "memsize_bytes": int(sysctl("hw.memsize")),
            "physical_cpus": int(sysctl("hw.physicalcpu")), "os": sw, "machine": platform.machine(),
            "gpus": gpus, "python": platform.python_version(),
            "mps": {k: probe.get(k) for k in ("ok", "torch", "recommended_max_memory", "matmul_max_rel_err",
                                              "matmul_ms", "env_before_import")}}


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout


def start(run_id: str) -> Path:
    folder = REPO / "evidence" / run_id
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "run_started_at": time.time(), "argv": sys.argv}
    (folder / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    (folder / "hw.json").write_text(json.dumps(hardware(), indent=2) + "\n")
    return folder


def finish(run_id: str, extra: dict | None = None) -> dict:
    folder = REPO / "evidence" / run_id
    meta = json.loads((folder / "run_meta.json").read_text())
    freeze = subprocess.run([str(PYTHON), "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    if "uv" in os.environ.get("VIRTUAL_ENV", "") or not freeze:
        freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(PYTHON)], capture_output=True, text=True).stdout
    (folder / "pip_freeze.txt").write_text(freeze)
    patch = git("diff", f"{UPSTREAM}..HEAD", "--", ".", ":(exclude)evidence")
    weights = {name: {"sha256": sha256(CACHE / "weights" / name), "source": url}
               for name, url in WEIGHTS.items() if (CACHE / "weights" / name).is_file()}
    datasets = {}
    sums = CACHE / "datasets" / "SHA256SUMS"
    if sums.is_file():
        for line in sums.read_text().splitlines():
            digest, name, url = line.split("  ", 2)
            datasets[name] = {"sha256": digest, "source": url}
    files = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name != "manifest.json"}
    manifest = {**meta, "run_finished_at": time.time(), "upstream_commit": git("rev-parse", UPSTREAM).strip(),
                "fork_commit": git("rev-parse", "HEAD").strip(),
                "working_tree_clean": git("status", "--porcelain", "--", "geometry", "tools", "docs").strip() == "",
                "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
                "pip_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
                "weights": weights, "weights_selected_sam3": [n.split("sam3-")[1].removesuffix(".ggml") for n in weights if "sam3-" in n],
                "inputs": datasets, "files": files, **(extra or {})}
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["start", "finish", "hw"])
    parser.add_argument("run_id", nargs="?")
    args = parser.parse_args()
    if args.action == "hw":
        print(json.dumps(hardware(), indent=2))
    elif args.action == "start":
        print(start(args.run_id))
    else:
        manifest = finish(args.run_id)
        print(f"{len(manifest['files'])} files registered; fork {manifest['fork_commit'][:12]}")


if __name__ == "__main__":
    main()
