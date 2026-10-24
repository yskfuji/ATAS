# macOS (Apple Silicon) geometry service

A port of the ATAS geometry service (Pi3X multi-view reconstruction + SAM3 video segmentation) to Apple Silicon. Branch `macos-geometry`.
Measured on a MacBook with an M4 Pro (48 GB). Read the status section before relying on it.

## What it is
* `GEOMETRY_DEVICE=mps|cpu|cuda` (default `mps` on macOS, `cuda` elsewhere). The CUDA path is kept as upstream wherever the code is
  checked statically (AST guards in `geometry/tests/mac`); no CUDA run was possible here, so CUDA behaviour is unmeasured.
* Unified-memory admission instead of per-GPU VRAM admission (`unified_pool.py`), macOS worker ownership (`proc_darwin.py`),
  F_FULLFSYNC durability, PyAV instead of decord, MPS fixes (float64 transfers, the 3x3 SVD of Pi3XVO runs on the CPU).
* SAM3 runs through [sam3.cpp](https://github.com/PABannier/sam3.cpp) (Metal or CPU) behind a small CLI (`geometry/native/sam3cpp`).

## Setup
    python3.12 -m venv .venv-macos && .venv-macos/bin/pip install -r geometry/requirements-macos.txt && .venv-macos/bin/pip install -e geometry
    brew install cmake ffmpeg
    # build the sam3.cpp CLI (see geometry/native/sam3cpp/CMakeLists.txt; the commit is pinned in SAM3CPP_COMMIT)
    export GEOMETRY_DEVICE=mps GEOMETRY_PYTHON=$PWD/.venv-macos/bin/python GEOMETRY_PI3_MODEL=<dir with Pi3X> \
           GEOMETRY_SAM3_CHECKPOINT=<sam3-f16.ggml> GEOMETRY_SAM3CPP_BIN=<build>/atas_sam3_track GEOMETRY_API_TOKEN=<token>
    geometry/run_geometry_service.sh            # curl http://127.0.0.1:18300/readyz
Weights are not part of this repository. **The Pi3X weights are CC BY-NC 4.0 (non-commercial)**; SAM3 is under Meta's SAM License;
see PUBLISHING_CHECKLIST.md.

## Status (honest summary; details in docs/acceptance and the report)
* Acceptance criteria were reconstructed (the original kit is missing), audited in nine rounds, frozen, and judged by a gate.
  Overall verdict form: `IN_SCOPE_PASS (BLOCKED: 7)` or `FAIL`; seven checks are BLOCKED by design (no CUDA machine, no 32 GB Mac,
  gated official SAM3 weights, no video-generation E2E).
* **AC12 (MPS == CPU-fp32 numerics) FAILS** on 6 of 33 windowed clips under the original and the amended limits (rotation on 5 clips; some of
  it is upstream run-to-run noise, `sintel_shaman_2` is not); the anchored path passes after the metric correction A12.
* **AC15 (SAM3 on DAVIS val) is below its absolute floor** (0.882 vs 0.892; the floor compares a box+point protocol with a published mask-prompt number).
* Render of a person world: 6.4 s per render, coverage >= 0.57 on 5 camera poses; recon of a 48-frame clip is dominated by SAM3 (~24 min on Metal).
* Opt-in features: [HYBRID.md](HYBRID.md) (SAM3 keyframes + EdgeTAM; 7.3x faster, 0.56x memory on the tested clip) and the ConvHead
  chunking (A10; driver peak 15.1 -> 8.1 GiB, bit-identical, default off).

## Tests
    cd geometry && ../.venv-macos/bin/python -m pytest tests/mac -q        # macOS-specific suite
    ../.venv-macos/bin/python tools/evidence_gate.py evidence/<run-id>        # the acceptance gate on a finished evidence run
