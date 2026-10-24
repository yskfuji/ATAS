"""AC10: Pi3X precision on MPS, measured inside the production code paths.

    python tools/pi3_dtype_probe.py run <clip-dir> <out-dir> --dtype float32|bfloat16|float16
    python tools/pi3_dtype_probe.py report <run-id> <out-dir> [--parity-reduced parity.json]

`run` executes the service's recon script (windowed Pi3X) in-process with forward hooks on every
attention and MLP submodule of Pi3X; the hooks record the largest |activation| and the number of
non-finite values before any downstream sanitisation. `report` applies the frozen rule: a dtype is
eligible only with zero non-finite values in every run, max|a| < 0.25 * 65504 for float16, and (for
reduced precision) a pass of AC12's reduced-precision parity; the fastest eligible dtype with at
least 1.25x speed-up over float32 is selected, otherwise float32.
"""
from __future__ import annotations

import json
import os
import runpy
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
FP16_MAX = 65504.0


def run(clip: Path, out: Path, dtype: str):
    os.environ["GEOMETRY_DEVICE"] = "mps"
    os.environ["GEOMETRY_MPS_DTYPE"] = dtype
    sys.path[:0] = [str(REPO / "geometry/backbones"), str(REPO / "geometry")]
    import torch
    sys.path.insert(0, str(REPO / "tools"))
    import crn_launch
    crn_launch.install()  # common random numbers across devices and precisions
    from utils.recon_and_seg import recon_pi3
    from atas_geometry.memory.timebase import probe
    stats = {}

    def hook(name):
        def record(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            detached = tensor.detach()
            finite = torch.isfinite(detached)
            entry = stats.setdefault(name, {"max_abs": 0.0, "nonfinite": 0, "calls": 0})
            entry["nonfinite"] += int((~finite).sum().item())
            entry["max_abs"] = max(entry["max_abs"], float(detached.float().abs().nan_to_num(0, 0, 0).max().item()))
            entry["calls"] += 1
        return record

    original = recon_pi3.init_pi3

    def instrumented(*args, **kwargs):
        model = original(*args, **kwargs)
        for name, module in model.named_modules():
            if name.endswith((".attn", ".mlp")):
                module.register_forward_hook(hook(name))
        return model
    recon_pi3.init_pi3 = instrumented
    frames = probe(clip / "clip.mp4")["frames"]
    folder = out / clip.name / f"probe_{dtype}"
    folder.mkdir(parents=True, exist_ok=True)
    sys.argv = ["recon_and_seg_single", "--video_path", str(clip / "clip.mp4"), "--output_folder", str(folder / "l0"),
                "--seg_keywords", "--recon_method", "pi3", "--pi3_model_id", str(CACHE / "weights/pi3x"),
                "--height", "720", "--width", "1280", "--num_frames", str(frames), "--pi3_pixel_limit", "255000",
                "--pi3_vo_chunk_size", "48", "--pi3_vo_overlap", "6", "--low_vram"]
    tick = time.perf_counter()
    error = None
    try:
        os.chdir(REPO / "geometry/backbones")
        runpy.run_module("scripts.preprocess.recon_and_seg_single", run_name="__main__")
    except BaseException as exc:  # a reduced-precision failure is a result, not a crash of the probe
        error = f"{type(exc).__name__}: {exc}"[:2000]
    timings = json.loads((folder / "l0" / "reconstruction_timings.json").read_text()) if (folder / "l0" / "reconstruction_timings.json").is_file() else {}
    result = {"clip": clip.name, "path": "windowed", "dtype": dtype, "error": error, "wall_s": time.perf_counter() - tick,
              "pi3_inference_s": timings.get("pi3_inference_and_rgbd"), "hooked_modules": len(stats),
              "nonfinite": sum(v["nonfinite"] for v in stats.values()),
              "max_abs": max((v["max_abs"] for v in stats.values()), default=None), "per_module": stats}
    (folder / "probe.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "per_module"}))


def report(run_id: str, out: Path, parity_reduced: dict):
    import numpy as np
    runs = [json.loads(p.read_text()) for p in sorted(out.glob("*/probe_*/probe.json"))]
    by_dtype = {}
    for r in runs:
        by_dtype.setdefault(r["dtype"], []).append(r)
    summary, eligible, speedup = {}, [], {}
    base = {r["clip"]: r["pi3_inference_s"] for r in by_dtype.get("float32", []) if r["pi3_inference_s"]}
    for dtype, rows in by_dtype.items():
        ok = all(r["error"] is None and r["nonfinite"] == 0 for r in rows)
        max_abs = max((r["max_abs"] or 0) for r in rows)
        if dtype == "float16":
            ok = ok and max_abs < 0.25 * FP16_MAX
        ratios = [base[r["clip"]] / r["pi3_inference_s"] for r in rows if r["clip"] in base and r["pi3_inference_s"]]
        speedup[dtype] = float(np.median(ratios)) if ratios else None
        parity_ok = dtype == "float32" or bool(parity_reduced.get(dtype))
        if ok and parity_ok:
            eligible.append(dtype)
        summary[dtype] = {"runs": len(rows), "nonfinite_total": sum(r["nonfinite"] for r in rows), "max_abs": max_abs,
                          "errors": [r["error"] for r in rows if r["error"]], "median_speedup_vs_fp32": speedup[dtype],
                          "reduced_parity_pass": parity_ok if dtype != "float32" else None}
    candidates = [d for d in eligible if d != "float32" and (speedup.get(d) or 0) >= 1.25]
    selected = max(candidates, key=lambda d: speedup[d]) if candidates else "float32"
    result = {"selected": selected, "eligible": eligible, "speedup_vs_fp32": speedup, "summary": summary,
              "rule": "zero non-finite; fp16 max|a| < 0.25*65504; reduced must pass AC12 reduced parity; fastest eligible with >=1.25x, else fp32",
              "runs": [{k: r[k] for k in ("clip", "path", "dtype", "nonfinite", "max_abs", "pi3_inference_s", "error")} for r in runs]}
    (REPO / "evidence" / run_id / "pi3_dtype_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ("selected", "eligible", "speedup_vs_fp32")}, indent=2))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[sys.argv.index("--dtype") + 1])
    else:
        parity = {}
        if "--parity-reduced" in sys.argv:
            for item in sys.argv[sys.argv.index("--parity-reduced") + 1:]:
                data = json.loads(Path(item).read_text())
                parity[data["dtype"]] = data["all_clips_pass"]
        report(sys.argv[2], Path(sys.argv[3]), parity)
