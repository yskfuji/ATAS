# Publishing checklist for the macOS (Apple Silicon) port

Status: published as a fork (account `yskfuji`) with a stacked set of branches (`macos/01-…` to `macos/05-…`, tip `macos-geometry`); the history was
rebuilt on top of the upstream initial commit and no raw evidence, data or weights are included (see below). Licences kept as upstream.

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

## Personal data and local paths (checked before publication)
* Commit author identity is a GitHub no-reply address; no personal e-mail is recorded.
* The pre-port test baselines under `evidence/baseline/` had a local path prefix, the machine's host name and the user name in a temp path;
  all three were replaced. The published results summary (`docs/macos/results`) has local paths and the lists of running processes removed.
* `hw.json` does not record serial numbers (by design, AC20). A tree-wide search for the user name, home paths, e-mail addresses and
  key/token patterns found nothing else.

## After publication
* Before any further release re-run a tree-wide search for local paths, host names and tokens, and check the model-weight licences again.
* Upstream (`ziplab/ATAS`) was asked first, through an issue (#2); its maintainers declined official macOS support for now (performance, validation and maintenance cost) and suggested a community fork. The stacked branches `macos/01-…` to `macos/05-…` stay in this fork; no pull request was opened.
