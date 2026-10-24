# Publishing checklist for the macOS (Apple Silicon) port

Status: prepared locally on branch `macos-geometry`. Nothing has been pushed; the `origin` remote still points at the upstream
`ziplab/ATAS`. Publishing is the owner's decision (target repository, account, timing).

## What may be published
* Code and tests: `geometry/`, `client/`, `tools/`, `examples/`, `docs/`, the preregistrations (`evidence/prereg/`), the
  acceptance documents (`docs/acceptance/`), and aggregate result JSON from `evidence/<run>/` (not the bulky or data-derived parts).
* The sam3.cpp patch (`geometry/native/sam3cpp/patches/mask_handoff.patch`) and the CLI (`atas_sam3_track.cpp`, `CMakeLists.txt`).

## What must NOT be published
| item | why |
|---|---|
| model weights (Pi3X, SAM3 ggml, EdgeTAM ggml) and `cache/` | licences below; users download or convert them themselves |
| datasets and anything derived frame-by-frame from them (DAVIS-2017, TUM RGB-D, MPI Sintel clips, renders, masks, `render_parity/`, `eval` outputs) | dataset licences (DAVIS: non-commercial research; Sintel: terms of its authors); size |
| `evidence/<run>/` raw folders (several hundred MB) | publish only the aggregate JSON listed in the report |
| `.venv-macos/`, `geometry/cache/`, scratch outputs | environment / local state |

## Licences found (checked 2026-10-23; re-verify before release)
| component | licence | consequence |
|---|---|---|
| this repository (ATAS) | MIT (`LICENSE`) | keep |
| `geometry/backbones/` (vendored third-party code) | Apache-2.0 (Meta, `geometry/backbones/LICENSE`); one file header states CC BY-NC-SA 4.0 (non-commercial; Naver-derived code inherited from upstream) | inherited from upstream; mention in README |
| Pi3 code | BSD-3-Clause | |
| **Pi3X weights** | **CC BY-NC 4.0, "strictly non-commercial"** (Hugging Face model card `yyfz233/Pi3X`) | anyone running the reconstruction with these weights is bound by the non-commercial clause; say so in the README; do not redistribute |
| SAM3 weights (`facebook/sam3`) | Meta SAM License (gated; 'proprietary free' category) | do not redistribute; the third-party ggml conversion on `PABannier/sam3.cpp` states no licence of its own; derive it from the official weights or link to it |
| sam3.cpp | MIT (per third-party listing; confirm from its repository's `LICENSE`) | the patch is a derivative: MIT, keep the notice |
| EdgeTAM code and checkpoints | Apache-2.0 (README of `facebookresearch/EdgeTAM`) | |
| DAVIS-2017 | non-commercial research | evaluation scripts only; no data |

## Personal data and local paths
* Commit author identity is a GitHub no-reply address (26 commits) plus the upstream author; no personal e-mail is recorded.
* Tracked file with a local absolute path: `evidence/baseline/geometry-junit.xml` (`/Users/<name>/...`). Replace the prefix by `<repo>`
  before publishing. Untracked evidence JSON (`hw.json`, run metadata, logs) contains local paths too: scrub or leave out.
* `hw.json` does not record serial numbers (by design, AC20).

## Before pushing (owner steps)
1. Choose the target: a new repository under the owner's account (recommended), or a pull request to upstream (needs the upstream's agreement; the patch touches `geometry/` widely).
2. Decide the branch history: keep `macos-geometry` as is, or squash into a few commits with a clean message.
3. Run `git grep -I "/Users/"` and the secret scan of the owner's choice on the final tree; check `git status` for untracked large files.
4. Add a short `docs/macos/README.md` (setup, supported devices, environment variables, what is BLOCKED) and link the report.
5. Publish only after the owner has read the final report (honest status: AC12 FAIL, AC15 below its floor, AC23/AC24 BLOCKED).
