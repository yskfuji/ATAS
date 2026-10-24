"""Write evidence/prereg/hybrid_stage2_rule.json (A11 stage 2) and, with --lock, mark it locked and write the .lock (sha256)."""
import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools")]
from hybrid_val import sha256  # noqa: E402
import hybrid_stage2 as S  # noqa: E402

LOCKING = "--lock" in sys.argv
archive = zipfile.ZipFile(S.DAVIS_ZIP)
seqs = archive.read("DAVIS/ImageSets/2017/val.txt").decode().split()
CLIPS = ["tum_freiburg3_walking_halfsphere", "tum_freiburg3_walking_rpy", "tum_freiburg3_walking_static", "tum_freiburg3_walking_xyz",
         "tum_freiburg3_sitting_halfsphere", "tum_freiburg3_sitting_rpy", "sintel_shaman_2", "sintel_bandage_2", "sintel_ambush_2", "sintel_bandage_1"]
frozen = {
    "cache/weights/sam3cpp/sam3-f16.ggml": CACHE / "weights/sam3cpp/sam3-f16.ggml",
    "cache/weights/sam3cpp/edgetam_f16.ggml": CACHE / "weights/sam3cpp/edgetam_f16.ggml",
    "cache/sam3.cpp/atas-build/atas_sam3_track": CACHE / "sam3.cpp/atas-build/atas_sam3_track",
    "cache/sam3.cpp/atas-build-hybrid/atas_sam3_track": CACHE / "sam3.cpp/atas-build-hybrid/atas_sam3_track",
    "cache/sam3.cpp/src-handoff/sam3.cpp": CACHE / "sam3.cpp/src-handoff/sam3.cpp",
    "cache/sam3.cpp/src-handoff/sam3.h": CACHE / "sam3.cpp/src-handoff/sam3.h",
    "cache/sam3.cpp/atas-build-hybrid.cmake.log": CACHE / "sam3.cpp/atas-build-hybrid.cmake.log",
    "cache/sam3.cpp/atas-build-hybrid.make.log": CACHE / "sam3.cpp/atas-build-hybrid.make.log",
    "geometry/native/sam3cpp/patches/mask_handoff.patch": REPO / "geometry/native/sam3cpp/patches/mask_handoff.patch",
    "geometry/native/sam3cpp/atas_sam3_track.cpp": REPO / "geometry/native/sam3cpp/atas_sam3_track.cpp",
    "geometry/native/sam3cpp/CMakeLists.txt": REPO / "geometry/native/sam3cpp/CMakeLists.txt",
    "geometry/backbones/utils/media.py": REPO / "geometry/backbones/utils/media.py",
    "tools/hybrid_stage2.py": REPO / "tools/hybrid_stage2.py",
    "tools/hybrid_val.py": REPO / "tools/hybrid_val.py",
    "tools/edgetam_reinit_sim.py": REPO / "tools/edgetam_reinit_sim.py",
    "tools/sam3_eval.py": REPO / "tools/sam3_eval.py",
    "tools/stats_tost.py": REPO / "tools/stats_tost.py",
}
dylibs = {str(p.relative_to(CACHE.parent)): sha256(p) for build in ("atas-build", "atas-build-hybrid")
          for p in sorted((CACHE / "sam3.cpp" / build).rglob("*.dylib")) if not p.is_symlink()}
BASELINE_RUN = "mac-20261019"
values = S.stage1_values(BASELINE_RUN, seqs)
rule = {
    "status": "locked" if LOCKING else "draft",
    "id": "A11 stage 2",
    "question": "Does the REAL hybrid (SAM3 only on frames 0, K, 2K, ...; EdgeTAM re-initialised from SAM3's masks propagates in between) (a) keep SAM3's J&F within the margin when "
                "tracking objects known at frame 0 on DAVIS-2017 val, and (b) run the SAM3 'person' tracker stage on one 48-frame TUM clip at least 1.25x faster within +10% peak memory, "
                "while still producing detections on at least 90% of the frames SAM3 does? Not tested: text-mode detection accuracy, new objects appearing between keyframes, the 'sky' pass, ATAS footage accuracy.",
    "stage1_result": "stage 1 (simulated upper bound, K=4): mean diff -0.0093, BCa90 [-0.0156, -0.0023], PASS (evidence/<run>/hybrid_val.json, rule sha256 7316e074...)",
    "baseline_run": BASELINE_RUN,
    "k": 4,
    "decision": "stage 2 PASSES iff part a PASSES and part b PASSES. A part-b output-validity failure is reported as 'not demonstrated' (not as a speed result). Part c is descriptive and cannot change the decision.",
    "order_and_isolation": "parts run strictly serially in the order b, a, c, each alone on the machine (no other GPU or heavy job, including my own scripts); the owner closes other applications before b and confirms in the command line (--owner-quiet), the confirmation is stored in env.json",
    "baseline_tolerance": 1e-6,
    "baseline_check_sequences": seqs[:3],
    "baseline_values_provenance": "SAM3 f16 J&F per sequence = stage-1 evidence/<run>/hybrid_val/<seq>.json -> sam3.JF (themselves checked against AC15 within 1e-6). They are not re-run for all 30 sequences; instead part a first runs the hybrid BUILD without --hybrid-k on the first three val sequences and aborts (invalid run, nothing scored) unless it reproduces the stored values within baseline_tolerance",
    "stage1_values_sha256": hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest(),
    "davis_archive_sha256": sha256(S.DAVIS_ZIP),
    "part_a": {
        "what": "real hybrid J&F on the 30 DAVIS-2017 val sequences, AC15 protocol (first-frame GT box + one interior point; objects are known at frame 0; box mode), vs SAM3 f16",
        "sequences": seqs, "k": 4, "margin": 0.02, "alpha": 0.05,
        "test": "identical to stage 1: one-sided paired t non-inferiority AND BCa 90% lower bound > -margin, both required; sd 0 handled as in stage 1; unit = sequence; missing/failed hybrid = 0",
        "power": {"method": "noncentral t, n=30, one-sided alpha 0.05, margin 0.02 (joint t+BCa within ~2 points)",
                  "true mean diff -0.0093": {"sd 0.0225": 0.82, "sd 0.03": 0.60}, "-0.012": {"sd 0.0225": 0.60, "sd 0.03": 0.41},
                  "-0.015": {"sd 0.0225": 0.32, "sd 0.03": 0.23}, "-0.018": {"sd 0.0225": 0.12, "sd 0.03": 0.10},
                  "note": "size is controlled (<= 0.05); a FAIL at a true gap of -0.015 occurs ~68% of the time, so a fail is not evidence of inferiority"},
        "failure_scoring": "binary failure or timeout (3600 s) -> identical retry up to 2x, all logged; then hybrid scored 0 for that sequence (no rerun); any other exception aborts the run",
        "expected_gap_to_simulation": "the SAM3 tracker at keyframes sees a 4-frame-strided video whereas stage 1 took SAM3's every-frame masks; a gap of a few thousandths is expected and is reported (paired real-minus-simulation)",
        "frame0": "EdgeTAM is prompted at frame 0 with the GT box + interior point (as AC15 and stage 1), not with a hand-off; hand-off happens at keyframes 4, 8, ...",
    },
    "part_b": {
        "what": "speed and memory of the SAM3 person-tracker stage: 48 frames of tum_freiburg3_walking_xyz at 1280x720, prompt 'person', sam3.cpp text tracker, f16",
        "clip": "tum_freiburg3_walking_xyz", "frames": 48, "prompt": "person", "k": 4, "repeats": 6,
        "design": "6 pairs of fresh processes, order alternating (sam3,hybrid / hybrid,sam3 ...), wall time of the whole process (model load included); run 0 of each variant (Metal shader compilation) is reported separately; the first complete 6-pair run is the one that counts",
        "speedup_min": 1.25, "speedup_definition": "median wall time of SAM3-alone runs 1-5 / median wall time of hybrid runs 1-5; the median (not A8's worst run) is used because one ratio of two noisy 5-run samples must not hinge on a single outlier; the worst-case ratio is reported",
        "expected": "from per-frame times (SAM3 ~13 s, EdgeTAM 0.5-1 s per frame) the SAM3-stage ratio is expected near 3 (K=4); one functional dry run (not a registered measurement, machine state not enforced) measured ~7 (SAM3 alone 1450 s, hybrid 198 s for the same 48 frames); a median ratio inside [1.19, 1.31] would be reported as borderline",
        "peak_ratio_max": 1.10, "peak_definition": "median over steady runs (1-5) of lifetime-max physical footprint (proc_pid_rusage, as A8) of the hybrid / same for SAM3-alone; max over all runs is reported; the footprint covers the process only (not the Python parent, not the page cache) and has not been calibrated against a known Metal allocation",
        "min_detection_frame_ratio": 0.9, "output_validity": "median over steady runs of frames-with-any-detection of the hybrid must be >= 0.9 x that of SAM3-alone; otherwise part b is 'not demonstrated' (a fast but empty output does not pass)",
        "load_ceiling_1min": 1.5,
        "conditions": "before every timed run the harness requires AC power, low-power mode off, no CPU speed limit (pmset -g therm) and a 1-minute load average <= 1.5 (waits up to 15 min, else the run is invalid); machine state before and after each run is stored with the run; the owner's confirmation that other applications are closed is stored in env.json",
        "persistence": "each run is appended to runs.jsonl as soon as it ends; part b refuses to start if its output directory is not empty; an aborted or repeated b run is reported with all timings",
        "limits": "one clip, one machine (M4 Pro 48 GB), 48 frames; the baseline is SAM3 on every frame of the person pass, not the AC12 lane (which runs SAM3 on every 4th frame and holds masks); production also runs a 'sky' pass that this part does not time",
        "sam3_share_of_recon": "A8's rationale records sam3.cpp at ~94% of the recon time of a 48-frame clip (~1000 s, two passes); other estimates in this project are lower; the share is not re-measured here, so no recon-level speed-up is claimed",
    },
    "part_c": {
        "what": "agreement of the hybrid's dynamic ('person') masks with SAM3's on ten clips, whole 48-frame clips at 1280x720 (SAM3 text tracker on every frame vs hybrid). AGREEMENT WITH SAM3 IS NOT ACCURACY: no claim about ATAS accuracy follows from it; high agreement only shows the hybrid copies SAM3, low agreement could mean either is right",
        "clips": CLIPS,
        "clip_choice": "the six TUM clips with people in the analysis set (all of them) and the four Sintel clips with the highest SAM3-only dynamic fraction (masks.npz of the AC12 lane, before the hybrid existed): shaman_2 0.7475, bandage_2 0.3170, ambush_2 0.2932, bandage_1 0.2914. Large foregrounds are easier for K=4 re-initialisation, so this selection is optimistic",
        "frames": 48, "prompt": "person", "k": 4, "repeat_control_clips": ["tum_freiburg3_walking_xyz", "sintel_shaman_2"],
        "metrics": "per frame IoU of the union dynamic mask (frames where either is non-empty), overall and split by keyframe / non-keyframe; share of SAM3-dynamic pixels the hybrid misses, also after the 5x5 dilation the pipeline applies to dynamic masks (geometry_eval.static_mask); share of hybrid-dynamic pixels SAM3 does not have; SAM3-vs-second-SAM3-run IoU on two clips as the repeatability ceiling (one functional dry run gave IoU 1.000); wall times and hand-off counters per clip (single runs, descriptive)",
        "status": "descriptive only: no pass/fail",
    },
    "behaviour_stated_in_advance": [
        "new objects appearing between keyframes are first seen at the next keyframe (up to K-1 = 3 frames late); part a cannot detect this (objects are known at frame 0); part c counts it only as 'missed' against SAM3",
        "when SAM3 finds nothing at a keyframe the hybrid outputs nothing for the following K-1 frames; an instance that disappears is not re-handed-off; EdgeTAM can emit masks for up to K-1 frames after an object leaves the frame",
        "propagated frames report the SAM3 score of the last keyframe (not a confidence); iou_score is EdgeTAM's",
        "each keyframe creates a new EdgeTAM state before the old one is released (a transient double state; conservative for the peak ratio)",
        "SAM3's text tracker counts steps: hotstart_delay 15, max_keep_alive 30 and recondition_every 16 are divided by K (ceil, min 1) in hybrid mode (recondition_every is unused by the text tracker); this scaling is also applied in box mode (part a); the unscaled alternative is untested; the values were fixed before any hybrid output in text mode was inspected (functional dry runs on one clip precede the lock; they changed nothing except logged bug fixes)",
        "duplicate pending instances during hotstart, if any, are all handed to EdgeTAM; hand-off counters are reported",
    ],
    "no_peeking": "run prints progress only; analyse runs once per part; n fixed at 30 for part a, no early stopping; no change to K, models, binaries, code or environment after the lock",
    "run_validity": "a baseline-check failure, a SAM3 or hybrid-build failure that cannot be attributed to infrastructure, a machine that is not quiet in part b, or any failed b run is an INVALID RUN (not a fail of the hybrid): it aborts, is logged with its cause in attempts.jsonl, keeps its timings, and a rerun needs the cause documented and the frozen files unchanged; at most two reruns per part; all voided runs are reported. Infrastructure failure = non-zero exit or timeout of a binary: identical retry up to 2x then (part a) hybrid scored 0. After the lock, a code defect found in a run is a deviation: it is reported with the original verdict and the fix is not applied silently",
    "deviation_protocol": "any deviation is reported with the original verdict; exclusions: none; the owner is told in the same message",
    "if_pass": "the hybrid may be offered as an opt-in setting (GEOMETRY_SAM3_HYBRID_K, default off) for the person-tracking pass; the text path's detection accuracy, late detection of new objects (item above) and the sky pass remain unvalidated; making it the default needs a downstream check on ATAS-domain footage with ground truth, which this rule does not provide. A pass of part a plus stage 1 is not two independent confirmations (same 30 sequences, DAVIS val is consumed) and is no evidence about ATAS footage",
    "if_fail": "no evidence that the hybrid is non-inferior / fast enough on these tests; at a true gap of -0.015 part a fails ~68% of the time, a failed part b can also reflect machine state; the hybrid stays off; no retuning on val",
    "limits_stated_in_advance": [
        "part a reuses the 30 val sequences of stage 1: the tests are not independent; part a measures the implementation's gap to the simulation on the same data. DAVIS val is consumed. No implementation choice was selected on val numbers; K=4 was fixed before stage 1",
        "contamination: EdgeTAM trained on DAVIS (arXiv 2501.07256 Sec 4.1); SAM3's media pool lists DAVIS2017 (arXiv 2511.16719 Table 25, column inferred); which split each saw is not stated",
        "SAM3 weights are a third-party ggml conversion; identity with facebook/sam3 is unverified",
        "no ground truth exists for the ATAS workload (parts b, c): accuracy there is not measured",
        "speed is measured on one clip on one machine; the SAM3 share of a real recon is not re-measured",
        "the hybrid's SAM3 build (atas-build-hybrid) differs from the baseline build (atas-build); the abort-before-scoring check covers the first three sequences only, and is the only equivalence check",
        "before registration the real hybrid was run once, in box mode on one DAVIS train sequence (motocross-bumps: J&F 0.908 vs simulation 0.907, SAM3 0.920) and in functional dry runs; no val run existed"],
    "frozen_files": {k: sha256(v) for k, v in frozen.items()},
    "frozen_dylibs": dylibs,
    "frozen_clips": {f"cache/clips/{c}/clip.mp4": sha256(CACHE / "clips" / c / "clip.mp4") for c in CLIPS},
    "environment": S.environment(),
}
path = REPO / "evidence/prereg/hybrid_stage2_rule.json"
path.write_text(json.dumps(rule, indent=2))
if LOCKING:
    (REPO / "evidence/prereg/hybrid_stage2_rule.lock").write_text(f"{sha256(path)}  hybrid_stage2_rule.json\n")
print("rule", rule["status"], sha256(path), "| frozen", len(rule["frozen_files"]), "dylibs", len(dylibs), "clips", len(rule["frozen_clips"]))
