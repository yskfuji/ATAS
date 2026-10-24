# Acceptance-criteria audit log

Every round is run by a fresh agent that did not write the criteria. Each round reads `acceptance.yaml`,
`ACCEPTANCE.md`, `legacy/QA_REPORT.md`, `legacy/*.svg`, upstream `docs/design.md`, `config.py` and
`jobs.py`. The auditor modifies no files.

## Round 1 — 2026-10-19 — verdict: FAIL (20 findings)

| # | Severity | Finding (abridged) | Resolution in revision 2 |
|---|---|---|---|
| 1 | BLOCKER | design.md invariants (L0 immutable, L1 append-only, L2 never fused, first-observation rule, r=1 rule, publication preconditions) traced only to AC16 and not tested | **AC25** added. It covers upstream invariant tests (must-not-skip), new L0/L1/L2 tests, decision constants, and AST-identity of the decision functions. |
| 2 | BLOCKER | No real end-to-end fusion/recapture on MPS | **AC26** added. It requires a real Pi3X + sam3.cpp HTTP run, zero CPU/MPS decision flips over the analysis clips, and recapture PSNR/IoU against the source frames. |
| 3 | MAJOR | Margins computed from the same clips; the pilot uses MPS data | Margins now come from a disjoint calibration set (CPU only) and are frozen numerically with hashes. The pilot is excluded from the analysis, and clip lists are preregistered with sha256. |
| 4 | MAJOR | Normal-approximation n under-powers; joint power under IUT about 0.8^k | Exact t-based n with per-metric power 0.928 (k=3); `min_analysis_clips: 29`; the gate recomputes n. |
| 5 | MAJOR | Two decision rules; percentile bootstrap under-covers; floor note; J&F misplaced | t-TOST is primary, the BCa cluster bootstrap is a sensitivity check, and both must pass. The floor behaviour is documented, and J&F moved to AC15. |
| 6 | MAJOR | AC12 dtype/summary/settings not bound; 10× bound overstated | Dtype bound to AC10; per-clip maximum over frames; Pi3X settings bound to the preregistration; thresholds labelled as engineering limits. |
| 7 | MAJOR | AC22 workload unbound; AC06 default ambiguity; p95 with n=5 | Both default sets are asserted; `clip_frames`/`recon_wh` are pinned; the bound is on the max. |
| 8 | MAJOR | RSS misses Metal buffers; 12 GB arbitrary; AC07 too weak | Memory measured as lifetime max physical footprint; the SAM3 budget is derived from the 24576 MiB recon admission; AC07 asserts the upstream requirements. |
| 9 | MAJOR | sam3.cpp compared with itself | An absolute DAVIS-17 J&F floor (published value − 0.03, cited and frozen) is added; weights are hashed with provenance. |
| 10 | MAJOR | AC08 tests no sockets | `test_af_unix_roundtrip_real_server` added. |
| 11 | MAJOR | Overall success undefined; NOT_APPLICABLE loophole; untraced ARCH/ERD nodes | `IN_SCOPE_PASS (BLOCKED: n)` verdict format; NOT_APPLICABLE forbidden; screenplay, planner, review and the ERD entities traced. |
| 12 | MAJOR | PSNR inflated by grey pixels; fallback time not measurable | PSNR over the covered-pixel union; the fallback-op list comes from a run with `PYTORCH_ENABLE_MPS_FALLBACK=0`, checked against an allowlist. |
| 13–20 | MINOR | Manifest values; hw timestamp; SIGKILL points; dir fsync; AC11 tolerance; must-not-skip; AC10 parity binding; unit consistency | All adopted (see yaml). |

## Round 2 — 2026-10-19 — verdict: PASS WITH FIXES (8 findings)

| # | Severity | Finding (abridged) | Resolution in revision 3 |
|---|---|---|---|
| 1 | BLOCKER | Missing evidence could fall into BLOCKED, so `IN_SCOPE_PASS` could be printed with ACs unmeasured; AC23 could show PASS with a blocked check | Missing, unregistered or mismatched evidence = FAIL. Blocked checks are allowed only in AC23/AC24, and the gate asserts there are exactly 7. AC23 = PARTIAL. `IN_SCOPE_PASS` requires every other AC to PASS. |
| 2 | MAJOR | Reference values read from the evidence file under judgement (required_n, fallback allowlist, jf floor, budget, must-not-skip, psnr floor, dtype binding); `m` undefined | Assert namespace defined (`d`, `ev`, `m`, `pre`, `gate`). All reference values now come from `pre` (frozen), from `gate` (computed by the gate), or from a different evidence file. |
| 3 | MAJOR | Function-level AST pins can be bypassed by module-level rebinding; reference selection unpinned | `ast_module` for matching/reviewed_world. `ast_identical_except_device_calls` for initial_world, anchored_fusion and writeback, covering whole modules (no new top-level statements). worker_protocol allows exactly one darwin block. |
| 4 | MAJOR | Preregistration order unverifiable | The lock commit must be a git ancestor of every commit adding MPS evidence, and no MPS evidence may be in its tree. Prereg hashes are defined. *Residual risk:* the repository is local (no push, by owner decision), so ordering is attested by git ancestry, not by an external timestamp. |
| 5 | MAJOR | Workload too easy; per-frame budget loosens on short clips | `perf_clip_frames` = p90 of analysis clip lengths, frozen. Settings and dtype bound. The SAM3 budget uses the frozen frame count. |
| 6 | MINOR | AC26 status names wrong; pairs and HTTP unproven; no absolute recapture floor | Upstream statuses `unchanged/repaired/rejected`; preregistered e2e pairs; flips counted by type; `via_http`; floor `max(20 dB, CPU − 1 dB)`. |
| 7 | MINOR | n ≥ 29 confirmed by simulation (0.938 at n=29); the pilot SD is noisy | Sizing uses the 80 % UCL of the pilot SD. |
| 8 | MINOR | The 24576 MiB proxy is unstated | Stated as a deliberately conservative proxy. |

## Round 3 — 2026-10-19 — verdict: PASS WITH FIXES (7 findings, all with exact fixes)

| # | Severity | Finding (abridged) | Resolution in revision 4 |
|---|---|---|---|
| 1 | MAJOR | AC21 exemptions (`device_divergence_ids`, allowlist) were read from the evidence | Both now come from `gate`, parsed from locked files. Each divergence names a tests/mac counterpart, which must pass and must not be skipped. |
| 2 | MAJOR | AC26 clip_a could be `single_window_clip` (never checked) | `reason` must be one of the three checked outcomes, and frames must exceed first_window. |
| 3 | MAJOR | The device_call_rule folds any `accel.*` call; external rebinding | Closed `accel` name list with one-to-one substitution back to the upstream expression; other arguments are compared; a package-wide rebinding scan is added. |
| 4 | MAJOR | Local git ancestry can be rewritten | Impact neutralised: clip lists come from a deterministic seeded rule and margins are recomputed by the gate from CPU-only data. The residual risk is stated in the yaml preregistration block. |
| 5 | MINOR | `pre` keys undefined | Merge rule defined (yaml block, then margins.json, then lock hashes; a collision is an error). |
| 6 | MINOR | AC15 frame-count assertion unsatisfiable on DAVIS | Timing split into `timing.clip_frames`. |
| 7 | MINOR | Schema label stale; anchored Pi3X dtype path | Schema v4; AC10/AC12 cover both the windowed and the anchored path. |

## Conclusion

The auditor's round-3 confirmations held across rounds: the verdict rules, the gate-computed
references, the module-level AST guards, workload pinning, the AC26 floors, the two-mode
regression, and n ≥ 29 (simulation: power 0.938 at n = 29). Every remaining finding came with
an exact fix, and each was applied verbatim. Revision 4 is therefore **frozen** (stage-1 lock,
`acceptance.lock`).

Stage 2 locks the deterministic preregistration files: clip lists, margins and e2e pairs. It is
written after the CPU-only calibration run and before any MPS measurement.

## Amendments after freeze (before any analysis-clip measurement)

These amendments were made after the stage-1 lock, while only exploratory inputs existed: a
synthetic 48-frame smoke clip and unit tests. No analysis, calibration or pilot clip had been
measured. Numeric thresholds are unchanged.

| ID | Change | Evidence that forced it |
|---|---|---|
| A1 | AC07 test `test_parent_does_not_import_torch` replaced by `test_parent_admission_never_queries_torch_mps`. | Upstream's own HTTP process already imports torch (through `memory.fuse` via `jobs`); checked by importing `create_app` on the stashed, unmodified upstream tree. "No torch import" is therefore not achievable without restructuring upstream. The risk the guard targets (admission reading the parent's own MPS allocator) is now tested directly: every `torch.mps` query raises in the parent while it starts, serves `/readyz` and admits a job. |
| A3 | Junit matching clarified. A listed id without parameters means all of its parametrizations, with at least one required. File-level entries tolerate only locked-allowlist CUDA variants, and only when the matching `[mps]` variant passed. | The upstream GPU tests are parametrized `['cpu','cuda']`. The literal rule "any skip in a listed file is a FAIL" could never pass on hardware without CUDA, even though the `[mps]` variants that were added do run. This is a clarification: nothing that can run on this Mac becomes skippable. |
| A4 | AC21 gets a separate locked skip list for the CUDA-contract run (`skip_allowlist_cuda_mode.txt`, 14 GPU-gated ids). The gate additionally requires each of these ids, or its `[mps]` variant, to pass in the mps run. | Running upstream with `GEOMETRY_DEVICE=cuda` on a Mac without CUDA necessarily skips GPU-gated tests. The revision-6 rule (CUDA-run skips ⊆ the 3-entry mps list) was unsatisfiable. Discovered by dry-running the regression tool; no geometry measurement was involved. |
| A6 | Accuracy against ground truth follows the published video-depth protocol. Ground truth is capped at 70 and aligned predictions are clipped at 70, as in MonST3R's evaluation (`depth_metric.ipynb`, `max_depth=70` for Sintel and Bonn). The earlier ad-hoc cap was 1e3. | During CPU-only calibration, before any lock commit or MPS measurement, `sintel_temple_3` gave AbsRel 8.94 because its far background (GT median 352) dominated the median scale. That made the calibration mean 1.55 and the AbsRel margin meaningless. The cap is taken from the cited primary source, not tuned. |
| A5 | Device comparisons (AC10, AC12, AC13, AC26 decisions) run under **common random numbers**. The harness makes `torch.rand` draw from one seeded CPU generator for both devices; production code is unchanged. The native-RNG CPU-vs-CPU difference is reported as an upstream noise floor. | Upstream Pi3X applies a random depth-scale augmentation (`scale_aug = 0.8 + torch.rand(...)*0.4`) at inference whenever priors are injected, which Pi3XVO does from the second window on. Two **CPU** runs of the same synthetic clip differ by depth AbsRel 4.0e-3, more than MPS vs CPU (2.3–2.9e-3). The AC12 fp32 limit therefore could not be met by CPU against itself, a definition defect. Under CRN, CPU vs CPU is bit-identical and the synthetic MPS-vs-CPU difference is 2.5e-4 depth AbsRel, 0.048° rotation and 8.2e-6 for the conditioned forward pass. Found on synthetic and unit inputs; no analysis clip had been measured. Thresholds unchanged. |
| A2 | AC12 translation normaliser changed from camera-path extent to the median **static** scene depth (`trans_frac_depth`). The Chamfer normaliser is the robust static extent ‖P98 − P2‖. Both are computed over the L1 static mask (round 4). | The synthetic smoke clip had a camera-path extent of 0.06 units against a median depth of 2.2. Dividing by path extent turned a 0.008-unit difference into "13 %". The change was **triggered by an MPS observation**. The ill-conditioning is visible in CPU-only data: in the CPU-fp32 run, the camera-path extent was 0.0605 against a median depth of 2.209, a ratio of 0.027. The smoke difference still fails under the new normaliser (≈3.6e-3 > 1e-3), so the amendment does not turn a fail into a pass. |

**Exploratory observation, recorded and not acted on.** On the synthetic clip, MPS-fp32 vs
CPU-fp32 gave a per-frame depth AbsRel of up to 2.9e-3 and a rotation error of up to 0.079°.
Both exceed the AC12 fp32 limits of 1e-3 and 0.05°. The thresholds are **not** relaxed. AC12 will
be judged on the preregistered analysis clips only.

The same smoke test falsified one candidate explanation for a rejection: the pinhole-residual
rejection (2.65 px > 2 px) also occurs on CPU, so it comes from the planar synthetic input, not
from MPS.

**Bug found by AC09/AC03 tests.** Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0` alone makes torch
reject the default low watermark of 1.4. The probe, and with it `/readyz`, would therefore always
fail. Fixed by also setting `PYTORCH_MPS_LOW_WATERMARK_RATIO` in the default 1.4 : 1.7 proportion.

### Round 4 — 2026-10-19 — amendment review: PASS WITH FIXES

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | OK | A1 is a legitimate definition fix | — |
| 2 | MAJOR | The A1 test missed pre-bound names, `torch._C._mps_*` and `torch.backends.mps`, and did not prove the probe value is used | `torch._C._mps_*` and `torch.mps.*` now raise before `create_app` is imported. Added `test_admission_uses_probe_value`. |
| 3 | OK | A2's motivation is legitimate; the smoke difference still fails | The trigger and the CPU-only evidence are now stated. |
| 4 | MAJOR | Normalisers inflatable by sky/far points and outliers | L1 static mask with a robust extent ‖P98 − P2‖; `normaliser_mask == 'l1_static'` is asserted. |
| 5 | MINOR | Re-lock; exploratory evidence location | Re-locked as revision 5 (stage 1). Smoke outputs stay in the session scratchpad, outside `evidence/`. |

### Round 5 — 2026-10-19 — amendment A5 review: PASS WITH FIXES

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | OK | A5 is a legitimate paired-comparison method (common random numbers) | — |
| 2 | MAJOR | CRN might not reach worker processes; `random_numbers` was self-reported | Each process installs CRN from the environment and logs `crn_installed` plus a draw-sequence hash. AC12/AC26 require equal CPU/MPS sequences. |
| 3 | MINOR | Other random functions; training mode | rand_like/randn/randn_like/randperm/uniform_ are covered; a Pi3X forward in training mode aborts. |
| 4 | MAJOR | Production decisions are random upstream; one seed is fragile | Three CRN seeds (AC12 per-clip max, AC13 per-clip mean). Five native-RNG CPU repeats per analysis clip report the noise floor and decision stability (informative). |
| 5 | MINOR | State which ACs run in production mode | AC14, AC15, AC22 and the AC26 HTTP loop run with native RNG; stated in the yaml. |

## Implementation fix during measurement (no criterion changed)

On 2026-10-20 the analysis clip `sintel_temple_2` failed on MPS at all three CRN seeds. Pi3XVO's chunk-alignment
`torch.svd` on MPS did not converge on a 3x3 covariance with entries ~1e9 and well-separated singular values
(9.3e9, 7.3e8, 3.9e7). The CPU SVD of the same matrix converges. Fix: on MPS only, this 3x3 SVD runs on the
CPU. The CUDA and CPU paths are unchanged.

Every MPS measurement was redone with the fixed code. The pre-fix MPS results are kept as `pre_svdfix_*` and are
reported. CPU runs were reused, since their code path did not change.

The same investigation showed that this covariance has a relative error of 1.0e-3 when computed in CPU fp32,
against 8.2e-8 on MPS (both against float64). That gives the CPU-fp32 *reference* a 0.069° Umeyama-rotation
error on this step, while MPS has 0.000°. AC12 is judged as frozen. The report also shows a float64 analysis of
this step so readers can see which side carries the error.

## Owner-authorised amendments (2026-10-20)

| ID | Change | Basis |
|---|---|---|
| A7 | AC12 per-clip limit becomes max(L_k, 2·O_k, 0.1·N_k): the original limit, twice the CPU-fp32 summation-order floor, or a tenth of the upstream native-RNG noise floor. The original-limit verdict is reported alongside. | On `sintel_bandage_1`, CPU fp32 at 10 vs 4 threads (same random numbers) already differs by 0.039° in rotation, against an original limit of 0.05°. A float64 analysis of Pi3XVO's chunk alignment showed a 0.069° error in the CPU-fp32 reference and 0.000° on MPS. The original limits fell below the algorithm's own fp32 precision. The owner authorised the change after seeing the pre-fix results. The amended rule was fixed before any post-SVD-fix MPS analysis result was examined (only 3 of 114 redone MPS runs existed and none had been compared). |
| A8 | SAM3 quantisation selection rule (q8_0, q4_0 vs f16): DAVIS equivalence (margin 0.02), floor, ≥1.25× speed, no more memory. The selected model becomes production and the AC15 subject. | sam3.cpp is ~94 % of recon time (≈1000 s of a 48-frame clip). Owner's decision to evaluate quantised models; the rule was registered before any quantised run. |

### Round 6 — 2026-10-20 — A7/A8 review: PASS WITH FIXES (finding 1 before AC12 is evaluated)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCKER | A7 was written after the failing pre-fix results were seen and might admit the defect | The rule is applied to the stored pre-fix MPS results, which must fail (asserted). Per-clip detection thresholds and the term that set each one are reported. The pre-fix evidence also contains the `temple_2` crash. |
| 2 | MAJOR | 0.1·N could swamp limits | N removed for depth/translation/Chamfer; rotation term is min(0.1·N, 0.25°). At most 10 % of clips may pass only via N. |
| 3 | MAJOR | One thread pair at one seed is a weak order floor | O = median over the 3 CRN seeds of 10-vs-4 threads. *Deviation:* the 1-thread arm (~40 h) was not run; this is stated. |
| 4 | MAJOR | CPU-fp32 reference is off by 0.069° on the alignment step | Reported as supplementary float64 analysis. The SVD fix is a tracked change (in patch_sha256, MPS-guarded, CUDA path unchanged; `pi3x_vo_v2.py` is outside the AST-pinned decision modules). |
| 5 | MAJOR | A8 selected and judged on the same DAVIS val data; timing statistic | Selection on 20 seeded DAVIS-17 *train* sequences; AC15 judged on val. ≥5 fresh-process interleaved timing runs, max statistic, first run separate. The gate recomputes eligibility and selection; the selected .ggml is hashed. |
| 6 | MINOR | Disclosures | All listed items go into the report. |

### Round 7 — 2026-10-20 — confirmation: PASS WITH FIXES (finding 1 before AC12 is evaluated)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | BLOCKER | The pre-fix check counted missing values and the crash as failures, so it did not test A7's power | Only clips with a real pre-fix measurement are judged (at least 29 required). `pre_svdfix_rescued` lists clips that failed L but pass A7. Asserted: the worst pre-fix clip still fails A7, and fewer clips are rescued than originally failed. |
| 2 | MINOR | A8 timing workload and power not enforced | Timing aborts unless frames == perf_clip_frames and the source is AC Power; both are asserted. |
| 3 | MINOR | Selected .ggml not in AC19 | AC19 requires every `weights_selected_sam3` model hashed, with its source. |
| 4 | MINOR | Eligibility booleans self-reported | The gate recomputes A8 eligibility and selection from per-sequence J&F and timing (`sam3_quant.compute_selection`). |
| 5 | MINOR | The MPS-only SVD placement makes AC12 not strictly "same code" | Stated in the AC12 title. |

## Owner-authorised amendment A9 (2026-10-22)

| ID | Change | Basis |
|---|---|---|
| A9 | AC22's render pass condition moves from the perf-lane renders to at least 5 renders of the preregistered AC26 world (real person subject), each required to reach mean coverage >= 0.5. Perf-lane render times stay in perf.json as reported numbers. recon, memory and swap are unchanged. | The perf-lane renders (5 of 5) had mean coverage 0.0953-0.0958; 81 % of pixels were flat unobserved grey, and the colours of the source floor were absent. The perf clip `tum_freiburg1_floor` has no person, so SAM3's "person" prompt tracked a false object and the planner stand-in aimed the camera at it. The render time (3.4 s) therefore measured an almost empty frame. This is a defect of the measurement workload, found by inspecting an output, not a device problem. recon time and memory (same runs) are unaffected. The coverage floor 0.5 was fixed before the replacement world was rendered. |

### Round 8 — 2026-10-22 — A9 review: PASS WITH FIXES (finding 2 before AC22 is evaluated)

| # | Severity | Finding | Resolution |
|---|---|---|---|
| 1 | OK | Moving the render condition fixes a workload defect, not a result | Audit trail kept; the old numbers stay in perf.json and are reported. |
| 2 | BLOCKER | `pre['e2e_world_pair_new']` undefined (KeyError) | The gate sets `e2e_world_pair_{new,history,new_sha256,history_sha256}` from e2e_pairs.json only after its hash matched the lock; a missing key makes the check FAIL. The assert also binds source_world_sha256 and the world version to AC26's world. |
| 3 | MAJOR | 0.5 coverage can be met by the wrong picture | Subject visible on >= 80 % of frames (>= 2 % of frame area), centre in frame, grey fraction <= 0.5, all computed by the harness from the world and the file; the same rule must pass for two CPU renders. |
| 4 | MINOR | Distinct shot ids may mean identical work | Five distinct side offsets (>= 3 distinct poses required), cache hits must be 0. |
| 5 | MINOR | Disclosure | The report states the perf-lane coverage problem and its cause, both render-time sets, that A9 was authorised after the results were seen, and which runs supply recon, memory and swap. |


## Owner-authorised amendments and reviews (2026-10-23)

| # | Change | Why / status |
|---|---|---|
| A10 | Pi3X ConvHead frame chunking (`ATAS_CONVHEAD_FRAME_CHUNK`, `geometry/backbones/pi3/models/layers/conv_head.py`), off CUDA only. **Opt-in, default 0 (upstream single pass); not part of any AC.** | Exact by construction (every ConvHead op is per frame); bit-identical outputs on 8 clips (`convhead_certify_mps.json`); driver peak 15.07 -> 8.09 GiB at chunk 2, same time (`convhead_memory`). Default stays off because the collected performance evidence was measured without it; switching it on needs a before/after re-measurement. |
| A11 | SAM3-keyframe + EdgeTAM hybrid, evaluated by preregistered rules (`evidence/prereg/hybrid_rule.json`, `hybrid_stage2_rule.json`, each pinned by a `.lock`); opt-in `GEOMETRY_SAM3_HYBRID_K`, default off. **Outside the acceptance criteria**; no AC status depends on it. | Stage 1 (simulation, DAVIS val): PASS. Stage 2 (real implementation): a PASS, b PASS; c descriptive. See docs/macos/HYBRID.md for limits (DAVIS val consumed; contamination; no ATAS ground truth). |
| A12 | AC12 rotation metric: stable atan2 form; old numbers kept as `pre_A12`; limits unchanged. | docs/acceptance/A12.md. AC12 remains FAIL (6 of 33 windowed clips); the anchored path changes from FAIL to PASS. |

### Round 9 — 2026-10-23 — A11 stage 1, A12, A11 stage 2 reviews: all PASS WITH FIXES (independent auditors, fixes applied before locking / implementing)
| review | main findings | resolution |
|---|---|---|
| A11 stage 1 rule | BLOCKER: the lock did not pin the rule file itself; MAJOR: baseline reproduction and run validity, retry/crash policy, missing environment pins, simulation-vs-real wording, untested code paths; minor: labels of the train bounds, outlier dependence, interim printing | rule sha256 pinned by `hybrid_rule.lock`; reproduction check inside `run()`; invalid run distinct from fail; attempts log and bounded retries; dylibs, DAVIS archive, pip freeze, OS build pinned; progress-only printing; analyse refuses to run twice; dry runs and synthetic analysis tests; power table and Type-I check recorded |
| A12 | formula and diagnosis verified independently (SO(3) projection; exact rotations); MAJOR: the draft understated that the anchored verdict flips, that 6 (not 5) windowed clips fail, that shaman_2 is not explained by upstream noise, and that A7's rotation relaxation and its pre-fix power check rested on the artefact | A12.md rewritten with these; errata below; old numbers kept as pre_A12; tests added |
| A11 stage 2 rule | MAJOR: no output-validity gate in part b; baseline not re-run with the hybrid build; lock gaps (DAVIS archive, stage-1 values, clips, media.py, build provenance); quiet-machine conditions not enforced or recorded; part-b data not persisted; power not stated; wording over-claimed | all addressed before the lock (see hybrid_stage2_rule.json); a dry run found one more defect (env.json counted as a result), fixed before locking |

### Errata (A12 consequences for earlier rows)
* **A5 and A7:** the figures quoted as the 10-vs-4-thread rotation difference (0.048 deg synthetic, 0.039 deg on `sintel_bandage_1`) were arccos artefacts; the true difference is ~6e-5 deg. With the corrected metric the A7 terms 2O and min(0.1 N, 0.25 deg) never exceed L, so A7's rotation relaxation no longer binds (it never needed to).
* **A7 pre-fix power check:** under the corrected metric the pre-fix and post-SVD-fix rotation values are essentially identical (r = 0.99999998), so the gate condition `pre_svdfix_worst_fails` holds only because the same clips fail before and after; it is not evidence that the rule can detect the defect. This is a weakness of the A7 design exposed, not created, by A12.
