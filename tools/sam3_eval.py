"""AC15: sam3.cpp video tracking on DAVIS 2017 val against ground truth, Metal f16 vs CPU f32.

    python tools/sam3_eval.py <run-id> [--limit N]

Protocol (preregistered): each annotated object of the first frame is prompted with its ground-truth
bounding box plus one positive point at the mask's interior point farthest from the boundary
(sam3.cpp exposes box/point prompts, not mask prompts; the published SAM 3 DAVIS17 J&F of 92.2 uses
first-frame masks, so this protocol is the harder one). J is region IoU and F the boundary
F-measure with tolerance 0.008 x image diagonal, as in the DAVIS 2017 evaluation (Pont-Tuset et al.).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
CACHE = REPO.parent / "cache"
BINARY = CACHE / "sam3.cpp/atas-build/atas_sam3_track"
MODELS = {"metal_f16": (CACHE / "weights/sam3cpp/sam3-f16.ggml", True),
          "cpu_f32": (CACHE / "weights/sam3cpp/sam3-f32.ggml", False)}


def seg2bmap(seg):
    """Boundary map of a binary mask (DAVIS evaluation definition, same-size output)."""
    seg = seg.astype(bool)
    e = np.zeros_like(seg); s = np.zeros_like(seg); se = np.zeros_like(seg)
    e[:, :-1] = seg[:, 1:]; s[:-1, :] = seg[1:, :]; se[:-1, :-1] = seg[1:, 1:]
    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]; b[:, -1] = seg[:, -1] ^ s[:, -1]; b[-1, -1] = 0
    return b


def f_measure(pred, gt, bound_th=0.008):
    from skimage.morphology import disk, binary_dilation
    radius = bound_th if bound_th >= 1 else math.ceil(bound_th * np.linalg.norm(gt.shape))
    fg, gb = seg2bmap(pred), seg2bmap(gt)
    fg_d, gb_d = binary_dilation(fg, disk(radius)), binary_dilation(gb, disk(radius))
    n_fg, n_gt = fg.sum(), gb.sum()
    if n_fg == 0 and n_gt > 0:
        return 0.0
    if n_fg > 0 and n_gt == 0:
        return 0.0
    if n_fg == 0 and n_gt == 0:
        return 1.0
    precision = (fg & gb_d).sum() / n_fg
    recall = (gb & fg_d).sum() / n_gt
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def j_measure(pred, gt):
    union = (pred | gt).sum()
    return 1.0 if union == 0 else float((pred & gt).sum() / union)


def interior_point(mask):
    from scipy.ndimage import distance_transform_edt
    distance = distance_transform_edt(mask)
    y, x = np.unravel_index(int(np.argmax(distance)), mask.shape)
    return float(x) + .5, float(y) + .5


def load_sequence(archive: zipfile.ZipFile, name: str):
    root = "DAVIS/JPEGImages/480p/" + name + "/"
    frames = sorted(n for n in archive.namelist() if n.startswith(root) and n.endswith(".jpg"))
    annotations = [n.replace("JPEGImages", "Annotations").replace(".jpg", ".png") for n in frames]
    images = [np.asarray(Image.open(io.BytesIO(archive.read(n))).convert("RGB")) for n in frames]
    labels = [np.asarray(Image.open(io.BytesIO(archive.read(n)))) for n in annotations]
    return images, labels


def run_sequence(images, labels, variant, per_frame=False, save_masks=None, extra_args=(), timeout=None):
    model, gpu = MODELS[variant]
    objects = [int(v) for v in np.unique(labels[0]) if v != 0]
    with tempfile.TemporaryDirectory(prefix="davis-") as tmp:
        tmp = Path(tmp)
        (tmp / "frames").mkdir()
        for i, image in enumerate(images):
            Image.fromarray(image).save(tmp / "frames" / f"{i:05d}.png", compress_level=1)
        rows = []
        for object_id in objects:
            mask = labels[0] == object_id
            ys, xs = np.nonzero(mask)
            px, py = interior_point(mask)
            rows.append(f"{object_id} {xs.min()} {ys.min()} {xs.max() + 1} {ys.max() + 1} {px} {py}")
        (tmp / "boxes.txt").write_text("\n".join(rows) + "\n")
        command = [str(BINARY), "--model", str(model), "--frames", str(tmp / "frames"), "--out", str(tmp / "out"),
                   "--init-boxes", str(tmp / "boxes.txt"), "--threads", "8"] + ([] if gpu else ["--no-gpu"]) + list(extra_args)
        tick = time.perf_counter()
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        wall = time.perf_counter() - tick
        if completed.returncode != 0:
            return {"error": completed.stderr[-2000:]}
        records = [json.loads(l) for l in (tmp / "out" / "tracks.jsonl").read_text().splitlines() if l.strip()]
        mapping = {r["instance_id"]: r["object_id"] for r in records if r.get("init")}
        summary = next(r["summary"] for r in records if "summary" in r)
        predicted = {o: [np.zeros(labels[0].shape, bool) for _ in images] for o in objects}
        for r in records:
            if "mask" in r and r["instance_id"] in mapping:
                predicted[mapping[r["instance_id"]]][r["frame"]] |= np.asarray(Image.open(tmp / "out" / r["mask"])) > 127
    if save_masks is not None:   # predicted masks, bit-packed, for offline re-initialisation experiments
        np.savez_compressed(save_masks, **{str(o): np.packbits(np.stack(predicted[o])) for o in objects}, shape=np.array([len(images), *labels[0].shape]))
    per_object = {}
    for o in objects:
        js = [j_measure(predicted[o][i], labels[i] == o) for i in range(1, len(images))]
        fs = [f_measure(predicted[o][i], labels[i] == o) for i in range(1, len(images))]
        per_object[o] = {"J": float(np.mean(js)), "F": float(np.mean(fs))}
        if per_frame:
            per_object[o]["J_by_frame"], per_object[o]["F_by_frame"] = js, fs
    j = float(np.mean([v["J"] for v in per_object.values()])); f = float(np.mean([v["F"] for v in per_object.values()]))
    return {"objects": len(objects), "frames": len(images), "J": j, "F": f, "JF": (j + f) / 2, "per_object": per_object,
            "wall_s": wall, "sec_per_frame_max": summary["sec_per_frame_max"],
            "peak_footprint_bytes": summary["peak_footprint_bytes"], "load_s": summary["load_s"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--variant", default=None, help="evaluate one Metal model (e.g. q8_0) on val into sam3_val_<variant>/")
    args = parser.parse_args()
    archive = zipfile.ZipFile(CACHE / "datasets/archives/DAVIS-2017-trainval-480p.zip")
    val = archive.read("DAVIS/ImageSets/2017/val.txt").decode().split()
    if args.limit:
        val = val[:args.limit]
    variants = list(MODELS)
    out = REPO / "evidence" / args.run_id / "sam3"
    if args.variant:
        MODELS[f"metal_{args.variant}"] = (CACHE / f"weights/sam3cpp/sam3-{args.variant}.ggml", True)
        variants = [f"metal_{args.variant}"]
        out = REPO / "evidence" / args.run_id / f"sam3_val_{args.variant}"
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in val:
        if (out / f"{name}.json").is_file():
            continue
        images, labels = load_sequence(archive, name)
        results[name] = {v: run_sequence(images, labels, v) for v in variants}
        (out / f"{name}.json").write_text(json.dumps(results[name], indent=2))
        print(json.dumps({name: {v: r.get("JF") for v, r in results[name].items()}}), flush=True)
    if not args.variant:
        (out / "davis_results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
