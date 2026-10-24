#!/usr/bin/env bash
# Full evidence run in the order the preregistration requires. Usage: tools/run_evidence.sh <run-id> <phase>
#   phase A  (no MPS measurement): clips, masks, CPU calibration, stage-2 lock  -> commit, then phase B
#   phase S, then B1 and B2 in parallel (MPS): geometry lane and SAM3 lane; then B3 alone (timed runs)
#   phase C  statistics, manifest -> commit, then gate
set -euo pipefail
RUN="$1"; PHASE="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv-macos/bin/python"
CLIPS="$ROOT/../cache/clips"
EVAL="$ROOT/../cache/eval/$RUN"
cd "$ROOT"
list() { "$PY" -c "import json,sys; print(' '.join(c['clip'] for c in json.load(open('evidence/prereg/$1_clips.json'))['clips']))"; }
pairs() { "$PY" -c "import json; print(' '.join(p['new'] for p in json.load(open('evidence/prereg/e2e_pairs.json'))['pairs']))"; }

case "$PHASE" in
A)
  "$PY" tools/datasets.py "$CLIPS"
  "$PY" tools/prereg.py lists "$CLIPS"
  for c in $(list calibration); do
    for k in 0 1 2; do "$PY" tools/geometry_eval.py recon "$CLIPS/$c" "$EVAL" --device cpu --dtype float32 --mode crn --seed-index $k; done
    "$PY" tools/geometry_eval.py compare "$CLIPS/$c" "$EVAL" --dtype float32
  done
  "$PY" tools/prereg.py margins "$CLIPS" "$EVAL"
  "$PY" tools/prereg.py lock
  echo "phase A done: commit evidence/prereg before phase B"
  ;;
S)  # start the evidence run (hardware fingerprint, run window); before B1/B2
  "$PY" tools/evidence.py start "$RUN"
  ;;
B1)  # geometry lane (results are deterministic under CRN; timings here are not evidence)
  for c in $(list pilot) $(list analysis); do
    "$PY" tools/geometry_eval.py masks "$CLIPS/$c" "$EVAL"
    for k in 0 1 2; do for d in cpu mps; do
      "$PY" tools/geometry_eval.py recon "$CLIPS/$c" "$EVAL" --device $d --dtype float32 --mode crn --seed-index $k; done; done
    "$PY" tools/geometry_eval.py compare "$CLIPS/$c" "$EVAL" --dtype float32
  done
  "$PY" tools/stats_tost.py pilot "$EVAL"
  for c in $(list analysis); do
    for dt in float32 bfloat16 float16; do "$PY" tools/pi3_dtype_probe.py run "$CLIPS/$c" "$EVAL" --dtype $dt; done
    for d in cpu mps; do "$PY" tools/decisions.py initial "$CLIPS/$c" "$EVAL" --device $d --mode crn --seed-index 0; done
  done
  for c in $(pairs); do
    "$PY" tools/geometry_eval.py recon "$CLIPS/${c}_history" "$EVAL" --device cpu --dtype float32 --mode crn --seed-index 0
    for d in cpu mps; do "$PY" tools/decisions.py anchored "$CLIPS/$c" "$EVAL" --device $d; done
  done
  # Upstream noise floor (informative): native-RNG CPU repeats and their first-observation decisions.
  for c in $(list analysis); do
    for k in 0 1 2 3 4; do
      "$PY" tools/geometry_eval.py recon "$CLIPS/$c" "$EVAL" --device cpu --dtype float32 --mode native --seed-index $k
      "$PY" tools/decisions.py initial "$CLIPS/$c" "$EVAL" --device cpu --mode native --seed-index $k
    done
    "$PY" tools/geometry_eval.py compare "$CLIPS/$c" "$EVAL" --dtype float32
  done
  "$PY" tools/decisions.py summarise "$RUN" "$EVAL"
  ;;
B2)  # SAM3 DAVIS lane, may run beside B1
  "$PY" tools/sam3_eval.py "$RUN"
  ;;
B3)  # quiet lane: run alone, after B1 and B2, because AC22/AC15 timings come from here
  "$PY" tools/render_parity.py "$RUN"
  "$PY" tools/timebase_evidence.py "$RUN" "$CLIPS"
  "$PY" tools/perf.py "$RUN" "$CLIPS"
  "$PY" tools/e2e_world.py "$RUN" --device cpu --port 18321 --clips "$CLIPS"
  "$PY" tools/e2e_world.py "$RUN" --device mps --port 18322 --clips "$CLIPS"
  "$PY" tools/run_regression.py "$RUN"
  "$PY" tools/guards.py --out "evidence/$RUN/cuda_scan.json"
  ;;
C)
  "$PY" tools/pi3_dtype_probe.py report "$RUN" "$EVAL"
  "$PY" tools/stats_tost.py analyse "$RUN" "$EVAL" --dtype float32 --selected "$("$PY" -c "import json;print(json.load(open('evidence/$RUN/pi3_dtype_report.json'))['selected'])")"
  "$PY" tools/sam3_summary.py "$RUN"
  "$PY" tools/evidence.py finish "$RUN"
  echo "phase C done: commit evidence/$RUN, then run tools/evidence_gate.py evidence/$RUN"
  ;;
esac
