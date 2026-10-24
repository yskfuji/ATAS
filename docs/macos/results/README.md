# Results summary (macOS port, run `mac-20261019`)

Aggregate results of the evidence run on one Apple M4 Pro (48 GB), copied from the local evidence directory with local paths and the list of
running processes removed. The full evidence run (rendered videos, per-clip reconstructions, caches) is not published; `manifest.json`
lists every file of that run with its sha256. Because the evidence directory itself is absent, `tools/evidence_gate.py` cannot be re-run from this
repository; `gate_report.json` is its output on the full run.

* `gate_report.json`: per-AC verdicts (21 PASS, AC12/AC15/AC26 FAIL, AC23 PARTIAL, AC24 BLOCKED)
* `parity.json`, `stats/tost.json`: MPS vs CPU numerics and ground-truth equivalence (AC12, AC13)
* `perf.json`, `render_remeasure.json`, `e2e_world*.json`, `regression.json`: performance, render time, world-state loop, test suites
* `hybrid_val*`, `hybrid_stage2*`: the SAM3 + EdgeTAM hybrid (stage 1 simulation, stage 2 a/b/c); rules are in `evidence/prereg/`
* `convhead_*`: the ConvHead frame-chunking certification and memory measurements
