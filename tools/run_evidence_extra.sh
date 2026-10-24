#!/usr/bin/env bash
# Owner-authorised additions (A7, A8). Kept separate from run_evidence.sh, which is executing.
#   B4  CPU summation-order floor: 4-thread CPU runs at the 3 CRN seeds for every analysis clip (A7)
#   B5  SAM3 quantisation selection data: f16/q8_0/q4_0 on 20 seeded DAVIS-2017 train sequences (A8)
#   Q   quiet: interleaved quantisation timing on the AC22 workload (run with B3, alone)
#   C0  before phase C: recompare analysis clips (order floors) and select the SAM3 model
set -euo pipefail
RUN="$1"; PHASE="$2"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv-macos/bin/python"
CLIPS="$(cd "$ROOT/../cache/clips" && pwd)"
EVAL="$(cd "$ROOT/../cache/eval" && pwd)/$RUN"
cd "$ROOT"
list() { "$PY" -c "import json,sys; print(' '.join(c['clip'] for c in json.load(open('evidence/prereg/$1_clips.json'))['clips']))"; }
case "$PHASE" in
B4)
  for c in $(list analysis); do
    for k in 0 1 2; do "$PY" tools/geometry_eval.py recon "$CLIPS/$c" "$EVAL" --device cpu --dtype float32 --mode crn --seed-index $k --threads 4; done
  done
  ;;
B5)
  "$PY" tools/sam3_quant.py davis "$RUN"
  ;;
Q)
  "$PY" tools/sam3_quant.py timing "$RUN" "$CLIPS"
  ;;
C0)
  for c in $(list analysis); do "$PY" tools/geometry_eval.py compare "$CLIPS/$c" "$EVAL" --dtype float32 > /dev/null; done
  "$PY" tools/sam3_quant.py select "$RUN"
  ;;
esac
