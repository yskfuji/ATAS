"""SAM3 video segmentation through sam3.cpp (Metal or CPU), same contract as seg_sam3_official.

``run_sam3_video`` returns ``(dynamic_mask[N,H,W] bool, seg_frames)`` where each frame lists
``{"id", "keyword", "score", "box_xyxy", "mask"}`` with ids unique across keywords, exactly as
the official implementation builds them.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Sam3Cpp:
    binary: str
    model: str
    use_gpu: bool = True
    threads: int = 8
    score_threshold: float = 0.5


def init_sam3_video(binary: str, model: str, use_gpu: bool = True, threads: int = 8, **_ignored) -> Sam3Cpp:
    for path in (binary, model):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    print(f"Using sam3.cpp ({'Metal' if use_gpu else 'CPU'}) model={model}")
    return Sam3Cpp(binary=str(binary), model=str(model), use_gpu=bool(use_gpu), threads=int(threads))


def _write_frames(video: np.ndarray, folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(video):
        Image.fromarray(frame).save(folder / f"{index:05d}.png", compress_level=1)


def run_tracker(config: Sam3Cpp, frames_dir: Path, out_dir: Path, *, prompts=None, init_boxes=None) -> list[dict]:
    command = [config.binary, "--model", config.model, "--frames", str(frames_dir), "--out", str(out_dir),
               "--threads", str(config.threads), "--score-threshold", str(config.score_threshold)]
    if not config.use_gpu:
        command.append("--no-gpu")
    if init_boxes is not None:
        command += ["--init-boxes", str(init_boxes)]
    for prompt in prompts or ():
        command += ["--prompt", prompt]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"sam3.cpp failed ({completed.returncode}): {completed.stderr[-4000:]}")
    return [json.loads(line) for line in (out_dir / "tracks.jsonl").read_text().splitlines() if line.strip()]


def run_sam3_video(video: np.ndarray, predictor: Sam3Cpp, keywords: List[str], collect_instances: bool = True,
                   report: dict | None = None):
    num_frames, height, width, _ = video.shape
    keywords = [keyword.strip() for keyword in keywords]
    print(f"Running sam3.cpp inference with keywords={keywords}")
    dynamic_mask = np.zeros((num_frames, height, width), dtype=np.bool_)
    seg_frames = [[] for _ in range(num_frames)]
    with tempfile.TemporaryDirectory(prefix="atas-sam3cpp-") as tmp:
        tmp = Path(tmp)
        _write_frames(video, tmp / "frames")
        records = run_tracker(predictor, tmp / "frames", tmp / "out", prompts=keywords)
        summary = next((r["summary"] for r in records if "summary" in r), {})
        if report is not None:
            report.update(summary)
        offset = 0
        for prompt_index, keyword in enumerate(keywords):
            rows = [r for r in records if r.get("prompt_index") == prompt_index and "mask" in r]
            max_id = -1
            for row in rows:
                mask = np.asarray(Image.open(tmp / "out" / row["mask"])) > 127
                if mask.shape != (height, width):
                    raise ValueError(f"sam3.cpp mask {mask.shape} does not match frame {(height, width)}")
                frame = int(row["frame"])
                dynamic_mask[frame] |= mask
                max_id = max(max_id, int(row["instance_id"]))
                if collect_instances:
                    seg_frames[frame].append({"id": int(row["instance_id"]) + offset, "keyword": keyword,
                                              "score": float(row["score"]),
                                              "box_xyxy": np.asarray(row["box_xyxy"], dtype=np.float32),
                                              "mask": mask})
            if max_id >= 0:
                offset += max_id + 1
    print(f"sam3.cpp summary: {summary}")
    print(f"Dynamic mask coverage: {dynamic_mask.astype(np.int64).sum() / np.prod(dynamic_mask.shape) * 100:.3f}%")
    return dynamic_mask, seg_frames
