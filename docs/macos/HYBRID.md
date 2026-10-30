# SAM3-keyframe + EdgeTAM hybrid (opt-in, default off)

SAM3 (large, slow) produces masks on every K-th frame; EdgeTAM (small, fast) is re-initialised from SAM3's mask at each keyframe
("mask hand-off") and propagates the frames in between. Implemented in the sam3.cpp command-line front end
(`geometry/native/sam3cpp/atas_sam3_track.cpp`, `--hybrid-k K --edgetam-model E.ggml`) and enabled in the geometry service by
environment variables. Nothing changes unless you set them.

## Build (needs the patched sam3.cpp)
1. Check out PABannier/sam3.cpp at the commit in `geometry/native/sam3cpp/SAM3CPP_COMMIT`.
2. Apply, in this order (`patch -p1` in the checkout): `geometry/native/sam3cpp/patches/mask_handoff.patch` (adds
   `sam3_tracker_add_instance_from_mask`), `visual_no_object_gating.patch` (an optional no-object gate for EdgeTAM, off by default),
   then `tracker_memory_release.patch` (required: without it the tracker aborts after about 1365 masklet-frames; see README.md).
3. `cmake -S geometry/native/sam3cpp -B build -DSAM3CPP_SOURCE_DIR=<patched checkout> -DGGML_METAL=ON -DATAS_MASK_HANDOFF=ON && cmake --build build --target atas_sam3_track`
4. Weights: `sam3-f16.ggml` and `edgetam_f16.ggml` are not distributed here (see PUBLISHING_CHECKLIST.md for their licences).

## Enable
    GEOMETRY_SAM3CPP_BIN=<build>/atas_sam3_track      # the patched build
    GEOMETRY_SAM3_HYBRID_K=4                          # 0 or unset = off; K >= 2
    GEOMETRY_SAM3_EDGETAM_MODEL=<path>/edgetam_f16.ggml
    GEOMETRY_SAM3_HYBRID_KEYWORDS=person              # default; the hybrid applies only to calls whose keywords are all listed
K without the EdgeTAM model is refused at start-up. The 'sky' pass and other keywords run SAM3 as before.
    GEOMETRY_SAM3_HANDOFF_DROP_EMPTY=1                # optional, default off: see "Optional: drop empty hand-offs" below

## What was measured (A11; evidence in evidence/prereg and evidence/<run>)
| claim | evidence | result |
|---|---|---|
| simulated hybrid (K=4) non-inferior to SAM3 on DAVIS-2017 val (margin 0.02 J&F) | stage 1, rule locked before the run | mean diff -0.0093, BCa90 [-0.0156, -0.0023]: PASS |
| real hybrid, same protocol and sequences | stage 2a | mean diff -0.0051, BCa90 [-0.0129, +0.0053]: PASS |
| speed and memory, 48 frames of a TUM clip, 'person', fresh processes, quiet machine | stage 2b | 7.3x faster (1456 s -> 199 s), peak footprint 0.56x (6.2 -> 3.5 GB): PASS |

| agreement of the hybrid's 'person' masks with SAM3's, ten clips, 48 frames, 1280x720 (descriptive; **agreement is not accuracy**) | stage 2c | mean IoU 0.835 (SAM3 vs a second SAM3 run: 1.000). Six clips 0.86-0.99; three clips 0.60-0.65 (`sintel_bandage_1`, `sintel_ambush_2`, `tum_freiburg3_walking_rpy`), where the hybrid misses 27-30% of SAM3's mask pixels on the two Sintel clips (fast animated motion) and keyframe masks already differ from SAM3's every-frame tracker. Wall time 5-7x lower on every clip (single runs) |

## Caution from stage 2c
In scenes with large, fast motion the hybrid can disagree strongly with SAM3. Whether SAM3 or the hybrid is right there is not measured (no ground
truth); the risk is a dynamic person leaking into the static map. Check your own footage (compare masks, or the downstream map) before using it.

## Optional: drop empty hand-offs (`GEOMETRY_SAM3_HANDOFF_DROP_EMPTY=1`; measured after the fact, not adopted as a default)
On ATAS-style footage the hybrid can spread a person's mask onto static objects next to the person. The cause: SAM3 sometimes outputs an
empty or tiny duplicate mask at a keyframe, and EdgeTAM, started from it as a present object, grows it onto nearby structure. With this
option the CLI never hands an empty SAM3 mask to EdgeTAM (`--handoff-min-coverage 0`, text prompts only). Measured on VIPSeg val (58
videos with at most 10 labelled persons) and DAVIS, one machine (`docs/macos/results/hybrid_drop_empty.json`):

| measure | unfixed | drop empty |
|---|---|---|
| VIPSeg: leak pixels outside the labelled persons and outside SAM3's own every-frame output | 1,699,904 | 586,409 (-66%) |
| VIPSeg: labelled-person pixels missed (pooled) | 5.63% | 5.64% (worst video +0.10 pp) |
| VIPSeg: labelled persons whose recall fell by more than 0.05 | - | 0 of 123 |
| DAVIS lab-coat, person 4 recall | 0.337 | **0.136** |
| DAVIS hike, person 1 recall | 0.729 | 0.693 |

The known weakness: a person SAM3 itself loses at a keyframe can be covered in the unfixed hybrid only by EdgeTAM masks grown from empty
hand-offs (lab-coat person 4), and is lost with this option. Stronger variants were worse and stay off: also dropping small fragments
(`--handoff-min-coverage 0.001`) deletes small distant people, and that together with the no-object options (`--handoff-empty-absent`,
`--gate-no-object`) failed a locked evaluation (it dropped real small objects on DAVIS). The option was chosen after the locked evaluations of these fixes had ended, so the table is a
post hoc measurement; there is no downstream (map) check.

## What is NOT validated
* Text-mode detection accuracy and ATAS footage (no ground truth); agreement with SAM3 on ten clips is reported in stage 2c, and agreement is not accuracy.
* New objects that appear between keyframes are first seen at the next keyframe (up to K-1 frames late); a person SAM3 misses at a keyframe is lost for K frames.
* DAVIS val is consumed (stages 1 and 2a share the 30 sequences); EdgeTAM was trained on DAVIS and SAM3's media list includes DAVIS2017, so the numbers may be optimistic.
* One clip, one machine (M4 Pro 48 GB); recon-level speed-up is not claimed (SAM3's share of a full recon was not re-measured).
Making it the default needs a downstream check with ground truth that these tests do not provide.
