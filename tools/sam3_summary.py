"""AC15 summary: DAVIS J&F of sam3.cpp Metal f16 vs CPU f32 (TOST, margin 0.02), absolute floor, timing.

    python tools/sam3_summary.py <run-id>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
from stats_tost import tost  # noqa: E402

MARGIN = 0.02


def main(run_id: str):
    run = REPO / "evidence" / run_id
    variant = json.loads((run / "sam3_quantisation.json").read_text())["selected"]
    results = {p.stem: json.loads(p.read_text()) for p in (run / "sam3").glob("*.json") if p.name != "davis_results.json"}
    if variant != "f16":  # the production model's own val results; the CPU f32 reference is shared
        for name, row in results.items():
            row["metal_f16"] = json.loads((run / f"sam3_val_{variant}" / f"{name}.json").read_text())[f"metal_{variant}"]
    ok = {k: v for k, v in results.items() if "JF" in v["metal_f16"] and "JF" in v["cpu_f32"]}
    metal = np.array([v["metal_f16"]["JF"] for v in ok.values()])
    cpu = np.array([v["cpu_f32"]["JF"] for v in ok.values()])
    test = tost(metal - cpu, MARGIN)
    perf = json.loads((run / "perf.json").read_text())
    margins = json.loads((REPO / "evidence/prereg/margins.json").read_text())
    summary = {"variant": variant, "n_sequences": len(ok), "errors": sorted(set(results) - set(ok)), "jf_margin": MARGIN,
               "jf_davis17_val": float(metal.mean()), "jf_davis17_val_cpu_f32": float(cpu.mean()),
               "jf_floor": margins["jf_floor"], "jf_floor_source": margins["jf_floor_source"],
               "jf_tost": test, "jf_tost_equivalent_vs_f32_cpu": bool(test["t_tost_equivalent"] and test["bca_bootstrap_equivalent"]),
               "davis_sec_per_frame_max": float(max(v["metal_f16"]["sec_per_frame_max"] for v in ok.values())),
               "timing": {"clip_frames": perf["clip_frames"], "sec_per_frame_max": perf["sam3_sec_per_frame_max"],
                          "source": "perf.json service recon runs (person prompt, 1280x720)"},
               "peak_footprint_bytes": int(max(v["metal_f16"]["peak_footprint_bytes"] for v in ok.values())),
               "protocol": "first-frame GT box + one interior positive point per object; J region IoU, F boundary (0.008 diag)",
               "pytorch_sam3_reference": "BLOCKED: facebook/sam3 weights are gated"}
    (run / "sam3_parity.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "jf_tost"}, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
