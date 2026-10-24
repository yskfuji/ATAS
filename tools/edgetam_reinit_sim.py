"""A11 step 2 (train only): offline simulation of "SAM3 keyframes + EdgeTAM in between".

    python tools/edgetam_reinit_sim.py <run-id> --k 4 8 16 [--sequences a b ...]

Needs the SAM3 f16 masks of each sequence saved by tools/edgetam_drift.py. For keyframe interval K, frames s = 0, K, 2K, ...
are keyframes: the output at s > 0 is SAM3's mask, and EdgeTAM is re-initialised there from a box and one interior point derived
from that mask (the same prompt kind as the first frame; sam3.cpp takes no mask prompt). The K-1 frames after a keyframe come
from EdgeTAM. Frame 0 uses the ground-truth prompt for both, as in AC15. SAM3 keyframe masks here come from SAM3's *full*
tracker run, so the result is an upper bound for a hybrid that runs SAM3 only on keyframes.
"""
import argparse
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
sys.path[:0] = [str(REPO / "tools")]
import sam3_eval  # noqa: E402

BINARY = CACHE / "sam3.cpp/atas-build-visual/atas_sam3_track"
HANDOFF_BINARY = CACHE / "sam3.cpp/atas-build-handoff/atas_sam3_track"   # sam3.cpp + patches/mask_handoff.patch
MODEL = CACHE / "weights/sam3cpp/edgetam_f16.ggml"
SEGMENT_TIMEOUT_S = 900   # one EdgeTAM segment takes seconds; a longer run is an infrastructure failure


def load_masks(path):
    z = np.load(path)
    t, h, w = (int(v) for v in z["shape"])
    return {int(k): np.unpackbits(z[k])[:t * h * w].reshape(t, h, w).astype(bool) for k in z.files if k != "shape"}


def propagate(images_png, start, stop, prompts, masks=None):
    """EdgeTAM over frames start..stop-1 -> {object: [mask per frame]}.
    prompts {object: (x0,y0,x1,y1,px,py)} on frame `start`; or, with `masks` {object: bool array}, the masks themselves
    (mask hand-off: the memory is initialised from the mask, no PVS decoding)."""
    with tempfile.TemporaryDirectory(prefix="sim-") as tmp:
        tmp = Path(tmp)
        (tmp / "frames").mkdir()
        for i, f in enumerate(range(start, stop)):
            (tmp / "frames" / f"{i:05d}.png").symlink_to(images_png[f])
        if masks is None:
            (tmp / "boxes.txt").write_text("\n".join(f"{o} " + " ".join(str(v) for v in p) for o, p in prompts.items()) + "\n")
            command = [str(BINARY), "--init-boxes", str(tmp / "boxes.txt")]
        else:
            lines = []
            for o, m in masks.items():
                Image.fromarray((m * 255).astype(np.uint8)).save(tmp / f"mask_{o}.png")
                lines.append(f"{o} {tmp / f'mask_{o}.png'}")
            (tmp / "masks.txt").write_text("\n".join(lines) + "\n")
            command = [str(HANDOFF_BINARY), "--init-masks", str(tmp / "masks.txt")]
        done = subprocess.run(command + ["--model", str(MODEL), "--frames", str(tmp / "frames"), "--out", str(tmp / "out"),
                                         "--threads", "8"], capture_output=True, text=True, timeout=SEGMENT_TIMEOUT_S)
        if done.returncode != 0:
            raise RuntimeError(done.stderr[-1500:])
        records = [json.loads(l) for l in (tmp / "out" / "tracks.jsonl").read_text().splitlines() if l.strip()]
        mapping = {r["instance_id"]: r["object_id"] for r in records if r.get("init")}
        h, w = np.asarray(Image.open(images_png[start])).shape[:2]
        out = {o: [np.zeros((h, w), bool) for _ in range(stop - start)] for o in prompts}
        for r in records:
            if "mask" in r and r["instance_id"] in mapping:
                out[mapping[r["instance_id"]]][r["frame"]] |= np.asarray(Image.open(tmp / "out" / r["mask"])) > 127
    return out


def prompt_from(mask):
    ys, xs = np.nonzero(mask)
    px, py = sam3_eval.interior_point(mask)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1, px, py)


def simulate(images, labels, sam3_masks, k, tmp_png, handoff=False):
    t = len(images)
    objects = sorted(sam3_masks)
    pred = {o: [None] * t for o in objects}
    for s in range(0, t, k):
        stop = min(s + k, t)
        prompts = {}
        for o in objects:
            m = (labels[0] == o) if s == 0 else sam3_masks[o][s]
            if m.any():
                prompts[o] = prompt_from(m)
        if handoff and s > 0:
            got = propagate(tmp_png, s, stop, prompts, {o: sam3_masks[o][s] for o in prompts}) if prompts else {}
        else:
            got = propagate(tmp_png, s, stop, prompts) if prompts else {}
        for o in objects:
            for i, f in enumerate(range(s, stop)):
                if f == s and s > 0:
                    pred[o][f] = sam3_masks[o][s]          # keyframe output is SAM3's
                elif o in got:
                    pred[o][f] = got[o][i]
                else:
                    pred[o][f] = np.zeros(labels[0].shape, bool)
    jf = []
    for o in objects:
        js = [sam3_eval.j_measure(pred[o][i], labels[i] == o) for i in range(1, t)]
        fs = [sam3_eval.f_measure(pred[o][i], labels[i] == o) for i in range(1, t)]
        jf.append((np.mean(js) + np.mean(fs)) / 2)
    return float(np.mean(jf))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--k", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--sequences", nargs="*")
    parser.add_argument("--handoff", action="store_true", help="initialise keyframes from the SAM3 mask itself (patched sam3.cpp)")
    args = parser.parse_args()
    run = REPO / "evidence" / args.run_id
    archive = zipfile.ZipFile(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip")
    names = args.sequences or sorted(p.name.split(".")[0] for p in (run / "edgetam_drift").glob("*.metal_f16.npz"))
    out = run / ("edgetam_reinit_sim_handoff" if args.handoff else "edgetam_reinit_sim")
    out.mkdir(exist_ok=True)
    for name in names:
        target = out / f"{name}.json"
        done = json.loads(target.read_text()) if target.is_file() else {}
        images, labels = sam3_eval.load_sequence(archive, name)
        masks = load_masks(run / "edgetam_drift" / f"{name}.metal_f16.npz")
        with tempfile.TemporaryDirectory(prefix="simpng-") as tmp:
            png = []
            for i, image in enumerate(images):
                path = Path(tmp) / f"{i:05d}.png"
                Image.fromarray(image).save(path, compress_level=1)
                png.append(path)
            for k in args.k:
                if str(k) not in done:
                    done[str(k)] = simulate(images, labels, masks, k, png, args.handoff)
                    target.write_text(json.dumps(done))
        print(name, done, flush=True)


if __name__ == "__main__":
    main()
