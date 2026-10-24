"""Static guards for the macOS port (AC01, AC05, AC23, AC25): compare the fork with upstream.

Usage: python tools/guards.py [--out cuda_scan.json]
Every check compares against the upstream commit through `git show`, never a stored copy.
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
UPSTREAM = "b86aab046368"
CUDA_RE = re.compile(r"""['"]cuda['"]|\.cuda\(|torch\.cuda\.|['"]cuda:|cuda:\{|nvidia-smi|CUDA_VISIBLE_DEVICES|SDPBackend\.FLASH_ATTENTION""")
SCANNED = ["geometry/atas_geometry", "geometry/backbones/utils/recon_and_seg", "geometry/backbones/scripts/preprocess",
           "geometry/backbones/pi3/models", "geometry/backbones/pi3/pipe/pi3x_vo_v2.py", "geometry/backbones/utils/misc.py"]
DECISION_PACKAGES = ("atas_geometry.reconstruct", "atas_geometry.register", "atas_geometry.world")
ACCEL_NAMES = {"backend", "torch_device", "compute_dtype", "autocast", "no_autocast", "worker_env", "device_type",
               "synchronize", "empty_cache", "set_device", "reset_peak_memory", "peak_memory_bytes",
               "peak_memory_source"}


def upstream_source(path: str) -> str:
    return subprocess.run(["git", "show", f"{UPSTREAM}:{path}"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout


def fork_source(path: str) -> str:
    return (REPO / path).read_text()


def dump(tree) -> str:
    return ast.dump(tree, include_attributes=False)


def expr(source: str) -> ast.expr:
    return ast.parse(source, mode="eval").body


def stmts(source: str) -> list[ast.stmt]:
    import textwrap
    return ast.parse(textwrap.dedent(source)).body


def _is_accel_import(node) -> bool:
    return (isinstance(node, ast.ImportFrom) and node.module == "atas_geometry"
            and [(a.name, a.asname) for a in node.names] == [("device", "accel")])


def _is_device_mapping_import(node) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "atas_geometry.service.device_mapping"


# Statement blocks whose upstream form spanned several statements (anchored_fusion.main).
STATEMENT_PATTERNS = [
    (stmts("accel.reset_peak_memory()"),
     stmts("import torch\nif torch.cuda.is_available():\n    torch.cuda.reset_peak_memory_stats()")),
    (stmts("peak = accel.peak_memory_bytes()\nif peak is not None:\n    fusion['gpu_reserved_peak_mib'] = round(peak / 2**20)"),
     stmts("import torch\nif torch.cuda.is_available():\n    fusion['gpu_reserved_peak_mib'] = round(torch.cuda.max_memory_reserved() / 2**20)")),
]


class _DeviceToUpstream(ast.NodeTransformer):
    """Rewrite each permitted accel call into the exact upstream CUDA expression it replaced."""

    def __init__(self):
        self.violations: list[str] = []

    def visit_Call(self, node):
        self.generic_visit(node)
        func = node.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "accel":
            name, args = func.attr, node.args
            if name not in ACCEL_NAMES:
                self.violations.append(f"accel.{name} is not a permitted device call")
                return node
            if name == "backend" and not args:
                return ast.Constant("cuda")
            if name == "compute_dtype" and not args:
                return expr("torch.bfloat16")
            if name == "autocast" and not args:
                return expr("torch.autocast('cuda', dtype=torch.bfloat16)")
            if name == "torch_device" and len(args) == 1:
                call = expr("torch.device(f'cuda:{device_index(X)}')")
                call.args[0].values[1].value.args[0] = args[0]
                return call
            if name == "set_device" and len(args) == 1:
                call = expr("torch.cuda.set_device(X)"); call.args[0] = args[0]; return call
            if name == "empty_cache" and not args:
                return expr("torch.cuda.empty_cache()")
            self.violations.append(f"accel.{name} with {len(args)} args has no upstream equivalent here")
        return node

    def visit_keyword(self, node):
        self.generic_visit(node)
        return node


def _expand_worker_env(tree):
    """**accel.worker_env(i)  ->  CUDA_VISIBLE_DEVICES=gpu_selector(i), CUDA_DEVICE_ORDER='PCI_BUS_ID'."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            keywords = []
            for keyword in node.keywords:
                value = keyword.value
                if (keyword.arg is None and isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
                        and isinstance(value.func.value, ast.Name) and value.func.value.id == "accel"
                        and value.func.attr == "worker_env" and len(value.args) == 1):
                    selector = expr("gpu_selector(X)"); selector.args[0] = value.args[0]
                    keywords += [ast.keyword("CUDA_VISIBLE_DEVICES", selector),
                                 ast.keyword("CUDA_DEVICE_ORDER", ast.Constant("PCI_BUS_ID"))]
                else:
                    keywords.append(keyword)
            node.keywords = keywords


def _rewrite_bodies(tree, drop, patterns=()):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            body = getattr(node, field, None)
            if not isinstance(body, list) or not body or not isinstance(body[0], ast.stmt):
                continue
            body = [s for s in body if not drop(s)]
            for fork, upstream in patterns:
                width = len(fork)
                i = 0
                while i + width <= len(body):
                    if [dump(s) for s in body[i:i + width]] == [dump(s) for s in fork]:
                        body[i:i + width] = copy.deepcopy(upstream)
                        i += len(upstream)
                    else:
                        i += 1
            setattr(node, field, body)


def ast_identical_except_device_calls(path: str, fork_text: str | None = None) -> dict:
    fork = ast.parse(fork_text if fork_text is not None else fork_source(path))
    upstream = ast.parse(upstream_source(path))
    _rewrite_bodies(fork, _is_accel_import, STATEMENT_PATTERNS)
    _expand_worker_env(fork)
    transformer = _DeviceToUpstream()
    fork = transformer.visit(fork)
    leftovers = sorted({n.id for n in ast.walk(fork) if isinstance(n, ast.Name) and n.id == "accel"})
    _rewrite_bodies(upstream, _is_device_mapping_import)
    _rewrite_bodies(fork, _is_device_mapping_import)
    same = dump(fork) == dump(upstream)
    return {"file": path, "identical": same and not transformer.violations and not leftovers,
            "violations": transformer.violations + [f"unsubstituted name {n}" for n in leftovers]}


def ast_module(path: str, allow_darwin_block: bool = False, fork_text: str | None = None) -> dict:
    fork = ast.parse(fork_text if fork_text is not None else fork_source(path)).body
    upstream = ast.parse(upstream_source(path)).body
    extra_imports, darwin_blocks, kept = [], [], []
    for node in fork:
        if allow_darwin_block and isinstance(node, ast.If) and dump(node.test) == dump(expr("sys.platform == 'darwin'")):
            darwin_blocks.append(node)
        elif allow_darwin_block and isinstance(node, ast.Import) and [a.name for a in node.names] == ["sys"] \
                and dump(node) not in {dump(u) for u in upstream}:
            extra_imports.append(node)
        else:
            kept.append(node)
    same = [dump(n) for n in kept] == [dump(n) for n in upstream]
    ok = same and len(darwin_blocks) <= (1 if allow_darwin_block else 0)
    return {"file": path, "identical": ok, "darwin_blocks": len(darwin_blocks), "extra_imports": len(extra_imports)}


def ast_functions(path: str, functions) -> dict:
    fork = {n.name: n for n in ast.parse(fork_source(path)).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    upstream = {n.name: n for n in ast.parse(upstream_source(path)).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    names = sorted(upstream) if functions == ["*"] else functions
    differing = [n for n in names if n not in fork or dump(fork[n]) != dump(upstream[n])]
    if functions == ["*"]:
        differing += sorted(set(fork) - set(upstream))
    return {"file": path, "identical": not differing, "differing": differing}


def rebinding_scan() -> list[str]:
    """Assignments to attributes of decision modules, or setattr/delattr on them, outside tests."""
    findings = []
    for file in sorted((REPO / "geometry/atas_geometry").rglob("*.py")):
        tree = ast.parse(file.read_text())
        aliases = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(DECISION_PACKAGES):
                        aliases.add(a.asname or a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                for a in node.names:
                    full = f"{node.module}.{a.name}"
                    if full.startswith(DECISION_PACKAGES) and (node.module.startswith(DECISION_PACKAGES) is False
                                                               or a.name[0].islower()):
                        aliases.add(a.asname or a.name)
        for node in ast.walk(tree):
            targets = []
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            elif isinstance(node, ast.Delete):
                targets = node.targets
            for target in targets:
                base = target
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(target, ast.Attribute) and isinstance(base, ast.Name) and base.id in aliases:
                    findings.append(f"{file.relative_to(REPO)}:{node.lineno}")
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"setattr", "delattr"}
                    and node.args and isinstance(node.args[0], ast.Name) and node.args[0].id in aliases):
                findings.append(f"{file.relative_to(REPO)}:{node.lineno}")
    return findings


def _scanned_files():
    for entry in SCANNED:
        path = REPO / entry
        yield from ([path] if path.is_file() else sorted(path.rglob("*.py")))


def cuda_literal_scan(allowlist_path: Path) -> dict:
    allow = []
    for line in allowlist_path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            location, _, reason = line.partition(" | ")
            file, _, text = location.partition("::")
            allow.append((file.strip(), text.strip(), reason.strip()))
    unallowlisted, used = [], set()
    for file in _scanned_files():
        rel = str(file.relative_to(REPO))
        for number, line in enumerate(file.read_text().splitlines(), 1):
            if not CUDA_RE.search(line):
                continue
            match = [i for i, (f, text, _) in enumerate(allow) if f == rel and text and (text == "*" or text in line)]
            if match:
                used.update(match)
            else:
                unallowlisted.append(f"{rel}:{number}: {line.strip()[:160]}")
    stale = [f"{allow[i][0]}::{allow[i][1]}" for i in range(len(allow)) if i not in used]
    missing_reason = [f"{f}::{t}" for f, t, r in allow if not r]
    return {"unallowlisted": unallowlisted, "stale_allowlist_entries": stale, "allowlist_entries_without_reason": missing_reason}


def missing_signatures() -> list[str]:
    """Upstream CUDA lines that vanished from a file that does not route through the device helper."""
    missing = []
    for file in _scanned_files():
        rel = str(file.relative_to(REPO))
        try:
            old = upstream_source(rel)
        except subprocess.CalledProcessError:
            continue  # new file
        new = file.read_text()
        uses_helper = "from atas_geometry import device as accel" in new or "accel." in new or "q.device.type" in new \
            or ".device.type" in new or "compute_device.type" in new or "torch.backends.mps" in new
        new_lines = {line.strip() for line in new.splitlines()}
        for line in old.splitlines():
            if CUDA_RE.search(line) and line.strip() not in new_lines and not uses_helper:
                missing.append(f"{rel}: {line.strip()[:160]}")
    return missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {
        "upstream_commit": UPSTREAM,
        **cuda_literal_scan(REPO / "geometry/tests/mac/cuda_allowlist.txt"),
        "missing_signatures": missing_signatures(),
        "gpu_pool": ast_functions("geometry/atas_geometry/service/gpu_pool.py", ["*"]),
        "worker_protocol": ast_module("geometry/atas_geometry/service/worker_protocol.py", allow_darwin_block=True),
        "decision_modules": [ast_module("geometry/atas_geometry/register/matching.py"),
                             ast_module("geometry/atas_geometry/world/reviewed_world.py"),
                             *[ast_identical_except_device_calls(p) for p in (
                                 "geometry/atas_geometry/reconstruct/initial_world.py",
                                 "geometry/atas_geometry/register/anchored_fusion.py",
                                 "geometry/atas_geometry/world/writeback.py")]],
        "decision_module_rebinding": rebinding_scan(),
        "linux_snapshot_unchanged": fork_source("geometry/environment.snapshot.txt") == upstream_source("geometry/environment.snapshot.txt"),
    }
    result["gpu_pool_ast_identical"] = result["gpu_pool"]["identical"]
    result["linux_functions_ast_identical"] = result["worker_protocol"]["identical"]
    text = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
