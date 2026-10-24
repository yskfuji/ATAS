# ATAS macOS geometry service: acceptance criteria — rationale and traceability

The machine-readable source of truth is `acceptance.yaml`. This file explains why each criterion is
there, where its thresholds come from, and which requirement it answers. A verdict is never written
here by hand: `tools/evidence_gate.py` computes verdicts from registered evidence.

## Requirement sources

| Key | Source | Content |
|---|---|---|
| QA.done.* | legacy/QA_REPORT.md "正しく完了したこと" | Claims the lost kit made. Each claim is re-implemented and re-proven here; none is trusted. |
| QA.not_done.* | legacy/QA_REPORT.md "完了していないこと" | m4pro_measure, m6_measure, pi3x_integration, sam3cpp_integration, world_fusion_integration, macos_patch, video_generator_e2e, quality_noninferiority, human_eval |
| ARCH.* | legacy/architecture.svg | screenplay, planner, world, geometry, generator, blender, review, publication |
| DESIGN.* | upstream docs/design.md | L0 immutable, L1 append-only/versioned, L2 never fused into L1, rejected registration never touches world, atomic pointer after manifest |

## Traceability (each requirement → at least one AC)

| Requirement | ACs |
|---|---|
| QA.not_done.m4pro_measure | AC07, AC20, AC22 |
| QA.not_done.m6_measure | AC24 (BLOCKED) |
| QA.not_done.pi3x_integration | AC10, AC11, AC12 |
| QA.not_done.sam3cpp_integration | AC15 |
| QA.not_done.world_fusion_integration (+recapture) | AC14, AC18 |
| QA.not_done.macos_patch | AC01, AC02, AC05, AC06, AC08 |
| QA.not_done.video_generator_e2e | AC24 (BLOCKED) |
| QA.not_done.quality_noninferiority | AC13 (geometry part), AC24 (video part, BLOCKED) |
| QA.not_done.human_eval | AC24 (BLOCKED) |
| QA.done.device_decision | AC01, AC09 |
| QA.done.sha256_observation | AC19 |
| QA.done.atomic_pointer | AC16, AC17 |
| QA.done.tracking_timebase | AC18 |
| QA.done.physics_calibration | AC24 (metric physics BLOCKED: upstream has no METRIC_CALIBRATION) |
| QA.done.mac_preflight | AC03 |
| QA.done.cuda_signature_scan | AC23 |
| QA.done.evidence_gate | AC19, gate itself |
| QA.done.contract_tests (19) | AC21 (`tests/contract/` re-created) |
| ARCH.world / ARCH.publication | AC04, AC16 |
| ARCH.geometry | AC01–AC03, AC07, AC12, AC14, AC15 |
| ARCH.screenplay / ARCH.planner / ARCH.generator / ARCH.review / ARCH.blender | AC24 (BLOCKED: client side needs LLM/generator APIs). Review *certificates* are still exercised as publication inputs in AC16/AC25. |
| DESIGN.L0_immutable, L1_append_only, L2_never_fused | AC25 (tree-hash, bytewise-subset, absence tests) |
| DESIGN.first_observation (t = 2 % median depth, ≥3 pairs, ≥80 % span, drift at 2t / 8t) | AC25 (upstream tests + constants + AST-identical decision code), AC26 (real run) |
| DESIGN.anchored_acceptance (≥30 SIFT, held-out history, ≤2 references) | AC25, AC26 |
| DESIGN.publication_preconditions, rejected_never_mutates | AC16, AC25 |
| ERD.WORLD_VERSION / OBSERVATION / STATIC_POINT_ASSET | AC16, AC25, AC26 |
| ERD.ADMISSION_DECISION / AUDIT_RECORD | AC25, AC26 (decision equality), AC19 (receipts) |
| ERD.METRIC_CALIBRATION | AC24 (BLOCKED) |
| ERD.FILM / SHOT / SHOT_ATTEMPT / ENTITY / ENTITY_STATE / EVENT_OBSERVATION / MODEL_RUN / DYNAMIC_MASK_TRACK | Proposed-only or client-side entities. The report's ERD comparison marks them; MODEL_RUN maps to AC19, DYNAMIC_MASK_TRACK to AC15. |

## Threshold rationale

* **AC12 parity (same code, CPU-fp32 vs MPS).** It isolates the device effect.
  * These thresholds are **engineering limits, not derived bounds**.
  * fp32 limits are AbsRel ≤ 1e-3, rotation ≤ 0.05°, and translation/Chamfer ≤ 0.1 % of scene
    extent. They allow for accumulation-order differences across about 24 transformer blocks, but
    not for algorithmic drift.
  * Reduced precision gets a 10× looser bound. That is about 2.5× the bf16 unit roundoff (2⁻⁸), not
    a derivation.
  * The summary is the **per-clip maximum over frames**, measured in raw reconstruction units, with
    the dtype and Pi3X settings bound to the preregistration. config.py uses 255 000 pixels;
    design.md's "~160k" figure refers to an older default.
* **AC13 equivalence (vs ground truth).** The primary rule is a paired t-TOST on per-clip
  differences (Schuirmann 1987; Lakens 2017). The clip is the unit, so frames are never pooled.
  A BCa cluster bootstrap by clip (Efron 1987; Field & Welsh 2007) is a sensitivity analysis, and
  both must pass. The 90 % CI corresponds to α = 0.05 per side (1 − 2α).
  * Margins: 10 % of the CPU-fp32 baseline AbsRel with a floor of 0.01, and 5 % for ATE/RPE.
    Note that the floor dominates whenever the baseline AbsRel is below 0.1. The idea is that the
    device must matter an order of magnitude less than the model's own error.
  * Independence of the margins: they are computed on a **disjoint calibration set**, from
    CPU-fp32 only. They are frozen as numbers, together with all clip lists and their sha256,
    before the first MPS run.
  * Sample size: the normal approximation n = (z₁₋α + z₁₋β/₂)² (σ_d/δ)² = 8.564 (σ_d/δ)²
    under-powers small n. The audit's exact simulation gave power 0.78 at n = 20.
  * Joint power: all k = 3 metrics must pass (intersection–union, Berger & Hsu 1996), so no α
    correction is needed but joint power drops. The target is therefore 0.8^(1/3) ≈ 0.928 per
    metric, which needs n ≥ 29 at σ_d/δ = 1.5.
  * The gate recomputes the required n from the pilot σ_d, using the exact t-based power. Pilot
    clips are excluded from the analysis.
  * J&F is not a geometry metric; it belongs to AC15.
* **ATE/RPE** as defined by the TUM benchmark (Sturm et al. 2012), after Umeyama Sim(3) alignment
  (Umeyama 1991). **J&F** as defined for DAVIS (Perazzi et al. 2016; Pont-Tuset et al. 2017).
* **AC14 render.** PSNR ≥ 40 dB means a mean squared error below 1e-4 for [0,1] images,
  which is visually indistinguishable. Coverage IoU ≥ 0.995 bounds holes and extra pixels.
* **AC15 SAM3.**
  * Speed: SAM3 runs inside the recon job, so its time budget is half of the AC22 recon headroom,
    divided by the clip's frame count. The published sam3.cpp M4 Pro figure is about 7.7 s/frame
    (f16, Metal).
  * Memory: SAM3 and Pi3X run sequentially in one job, so SAM3's peak footprint must leave the
    24 576 MiB recon admission free inside `recommended_max_memory`.
  * Accuracy: an absolute DAVIS-2017-val J&F floor, set at the published figure minus 0.03, keeps
    sam3.cpp from being judged only against itself.
* **AC22.** The **maximum** over at least 5 runs must leave 25 % headroom under the `from_env`
  hard deadlines: 2400 s recon (giving 1800 s) and 900 s render (giving 675 s). With 5 runs a
  "p95" would just be the maximum. The workload is pinned to ≥ 48 frames at 1280×720.
  * Memory is the macOS lifetime-maximum physical footprint. It counts Metal buffers, which RSS
    does not.
  * Swap growth of 256 MB or less shows the run does not rely on swap.
* **AC26 recapture.** "Recapture accuracy" from QA_REPORT is measured without a generator: the
  committed event is re-rendered at its source times and compared with the source frames over
  covered pixels. The floor is the CPU-fp32 value minus 1 dB.

## Known limits (stated, not hidden)

* Neither CUDA runtime parity nor M6 32 GB can be measured on this machine (AC23 runtime part,
  AC24).
* PyTorch SAM3 is not available as a reference because the weights are gated (AC24). Its place is
  taken by DAVIS ground truth plus a sam3.cpp f32 CPU run.
* Durability under power loss is argued from the syscall order (F_FULLFSYNC) and is not
  physically tested (AC17).

* The SAM3-keyframe + EdgeTAM hybrid and the ConvHead chunking (A10, A11) are opt-in features outside the acceptance criteria; their evidence and
  limits are in docs/macos/HYBRID.md and the A10/A11 rows of ACCEPTANCE_AUDIT.md.
* AC12's rotation metric is the stable atan2 form (A12); AC12 is FAIL on 6 of 33 windowed clips under both the original and the A7 limits.
