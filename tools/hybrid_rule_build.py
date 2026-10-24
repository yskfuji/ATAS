"""Write evidence/prereg/hybrid_rule.json (the A11 stage-1 rule) and, with --lock, mark it locked and write hybrid_rule.lock.

    python tools/hybrid_rule_build.py          # draft: hashes recomputed, status 'draft'
    python tools/hybrid_rule_build.py --lock   # status 'locked'; the lock file holds the rule's sha256 (commit both to git before `run`)
"""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools")]
from hybrid_val import environment, sha256  # noqa: E402

LOCKING = "--lock" in sys.argv
BASELINE_RUN = "mac-20261019"
archive = zipfile.ZipFile(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip")
seqs = archive.read("DAVIS/ImageSets/2017/val.txt").decode().split()
base = {n: json.loads((REPO / f"evidence/{BASELINE_RUN}/sam3/{n}.json").read_text())["metal_f16"]["JF"] for n in seqs}
frozen = {
    "cache/weights/sam3cpp/sam3-f16.ggml": CACHE / "weights/sam3cpp/sam3-f16.ggml",
    "cache/weights/sam3cpp/edgetam_f16.ggml": CACHE / "weights/sam3cpp/edgetam_f16.ggml",
    "cache/sam3.cpp/atas-build/atas_sam3_track": CACHE / "sam3.cpp/atas-build/atas_sam3_track",
    "cache/sam3.cpp/atas-build-visual/atas_sam3_track": CACHE / "sam3.cpp/atas-build-visual/atas_sam3_track",
    "cache/sam3.cpp/atas-build-handoff/atas_sam3_track": CACHE / "sam3.cpp/atas-build-handoff/atas_sam3_track",
    "geometry/native/sam3cpp/patches/mask_handoff.patch": REPO / "geometry/native/sam3cpp/patches/mask_handoff.patch",
    "geometry/native/sam3cpp/atas_sam3_track.cpp": REPO / "geometry/native/sam3cpp/atas_sam3_track.cpp",
    "tools/hybrid_val.py": REPO / "tools/hybrid_val.py",
    "tools/edgetam_reinit_sim.py": REPO / "tools/edgetam_reinit_sim.py",
    "tools/sam3_eval.py": REPO / "tools/sam3_eval.py",
    "tools/stats_tost.py": REPO / "tools/stats_tost.py",
}
dylibs = {}
for build in ("atas-build", "atas-build-visual", "atas-build-handoff"):
    for p in sorted((CACHE / "sam3.cpp" / build).rglob("*.dylib")):
        if not p.is_symlink():
            dylibs[str(p.relative_to(CACHE.parent))] = sha256(p)
rule = {
    "status": "locked" if LOCKING else "draft",
    "id": "A11 stage 1",
    "question": "Is the SIMULATED hybrid (SAM3 mask on every 4th frame, EdgeTAM between, re-initialised from SAM3's mask) non-inferior to SAM3 f16 on DAVIS-2017 val under the AC15 protocol? "
                "The simulation takes keyframe masks from SAM3's full-sequence tracker, so it is an UPPER BOUND for a real hybrid that runs SAM3 on keyframes only.",
    "data": "DAVIS-2017 val, 30 sequences, 480p; first-frame GT box + one interior point per object (AC15 protocol)",
    "sequences": seqs,
    "unit": "sequence: J&F averaged over frames 1..T-1, then over objects (as AC15; this differs from the official DAVIS per-object pooling)",
    "primary_k": 4,
    "keyframe_intervals_run": [4, 8],
    "primary_contrast": "d_i = JF_hybrid_k4,i - JF_sam3_f16,i, paired by sequence",
    "margin": 0.02,
    "margin_provenance": "equals AC15's jf_margin (A8), fixed before this analysis and not tuned to the hybrid",
    "alpha": 0.05,
    "test": "intersection-union: one-sided paired t non-inferiority (H0: mean(d) <= -margin, df = n-1) AND lower end of the BCa 90% two-sided CI (= 95% one-sided; 10000 resamples, seed 20261019) of mean(d) > -margin. Both required; disagreement = not non-inferior. If sd(d) = 0 the t statistic is +/-inf by the sign of mean+margin.",
    "type_I_check": {"method": "simulation, normal d with sd 0.032, n 30, 400 replicates, BCa 1500 resamples", "size_at_boundary_both_required": 0.043,
                     "size_t_only": 0.045, "size_bca_only": 0.058, "power_at_train_estimate_both_required": 0.767},
    "baseline": "SAM3 f16 JF per sequence from evidence/<baseline_run>/sam3 (AC15). Hash recipe: sha256 of json.dumps({sequence: JF}, sort_keys=True). "
                "The SAM3 rerun that produces the keyframe masks must reproduce each value within baseline_tolerance; this also proves the metric code is unchanged since AC15 (sam3_eval.py was edited after AC15 for per-frame output and mask saving).",
    "baseline_run": BASELINE_RUN,
    "baseline_tolerance": 1e-6,
    "baseline_values_sha256": hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest(),
    "run_validity": {"invalid_run": "a SAM3 baseline failure after retries, or a baseline that is not reproduced, aborts the run (nothing is scored; it is not a fail of the hybrid). "
                                    "After an abort a rerun is allowed only with the cause documented in attempts.jsonl and the same frozen files; the abort stays in the log.",
                     "infrastructure_failure": "non-zero exit or timeout (900 s per EdgeTAM segment) of a binary: identical retry up to 2 times, every attempt logged; then the hybrid is scored JF=0 for that sequence (conservative). "
                                               "Any other Python exception aborts the run. If the machine sleeps or a process is killed the run is resumed with the same command; sequences already written are kept and the log shows it.",
                     "reuse": "run refuses to start if the output directory holds results without an attempts.jsonl; deleting a result to re-roll it is a deviation and is recorded as one"},
    "no_peeking": "run prints progress only; analyse runs once after all 30 sequences are written (it refuses to run twice); n is fixed at 30, no early stopping and no stopping rule; no change to K, prompts, models, binaries, code or environment after the lock",
    "deviation_protocol": "any deviation from this rule is reported with the original verdict; the original rule's verdict is the one that stands; exclusions: none",
    "run_conditions": "quiet machine (no other heavy jobs), CLI threads 8, no fixed random seeds are used by the pipeline (it is deterministic); hardware and OS build are recorded in env.json",
    "if_pass": "licenses stage 2 only: implement the hybrid in the pipeline, then (a) re-test the REAL implementation's J&F against the same margin on data chosen and registered beforehand, (b) measure end-to-end speed (>= 1.25x, quiet machine) and memory, (c) measure the downstream effect on the static map. It does not adopt the hybrid by itself.",
    "if_fail": "a fail of the simulated upper bound is evidence against the hybrid on DAVIS; K and the method are not retuned on val; a retry needs data not used for selection. "
               "(A pass of the upper bound does not show the real hybrid passes; a fail of the real hybrid in stage 2 would be a separate finding.)",
    "not_claimed": "DAVIS J&F non-inferiority says nothing about the downstream static-map effect or about ATAS-domain footage",
    "descriptive_only": ["EdgeTAM alone vs SAM3 (val)", "hybrid K=8 vs SAM3 (val)", "worst sequence"],
    "power": {"n": 30, "method": "noncentral t, one-sided alpha 0.05 (t test alone; the both-required test is 1-2 points lower)",
              "values": {"train estimate: mean -0.0062, sd 0.032": 0.746, "mean -0.010, sd 0.032": 0.511, "mean -0.0062, sd 0.045": 0.498,
                         "mean -0.012, sd 0.045": 0.244, "true mean 0, sd 0.032": 0.955},
              "note": "point estimates from the selection data (20 DAVIS train sequences; the sd rests on n=20)."},
    "selection_data": {"set": "DAVIS-2017 train, 20 sequences, seed 20261019 (A8)",
                       "K4": {"mean_diff": -0.0062, "sd": 0.032, "t_lower90": -0.0186, "bca_lower90": -0.0141},
                       "K8": {"mean_diff": -0.0170, "sd": 0.048, "t_lower90": -0.0356, "bca_lower90": -0.0301},
                       "outlier_dependence": "drift-turn (+0.116 over SAM3) lifts K=4's mean from -0.0127 to -0.0062; without it the sd is 0.014; the headroom to -0.02 rests on one recovered SAM3 failure; the estimate is subject to a winner's curse"},
    "order_of_events": "SAM3 f16 and f32 per-sequence J&F on val (mean 0.882; AC15 floor 0.892 not met) were produced and seen before this rule (AC15). EdgeTAM and the hybrid have never been run on val. K, hand-off vs box re-prompt and the margin were fixed on train.",
    "limits_stated_in_advance": [
        "simulation: keyframe masks come from SAM3's full-sequence tracker, so the result is an upper bound for a hybrid that runs SAM3 on keyframes only",
        "contamination: EdgeTAM trained on DAVIS (arXiv 2501.07256 Sec 4.1); SAM3's media pool lists DAVIS2017 (arXiv 2511.16719 Table 25, column inferred); which DAVIS split each saw is not stated, so val is not known to be clean for either model",
        "SAM3 weights are a third-party ggml conversion; identity with facebook/sam3 is unverified",
        "mask hand-off: logits +-6, obj_score 10, zero object pointer, frame index clamp; not verified against EdgeTAM's reference mask-prompt path (SAM2 uses +-10 and a decoder-derived pointer); likely a small effect",
        "val is consumed by this test whether it passes or fails",
        "speed is not measured here"],
    "frozen_files": {k: sha256(v) for k, v in frozen.items()},
    "frozen_dylibs": dylibs,
    "davis_archive_sha256": sha256(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip"),
    "environment": environment(),
}
path = REPO / "evidence/prereg/hybrid_rule.json"
path.write_text(json.dumps(rule, indent=2))
if LOCKING:
    (REPO / "evidence/prereg/hybrid_rule.lock").write_text(f"{sha256(path)}  hybrid_rule.json\n")
print("rule", rule["status"], sha256(path), "| frozen", len(rule["frozen_files"]), "dylibs", len(dylibs))
