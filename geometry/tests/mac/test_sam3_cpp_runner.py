"""AC15: the sam3.cpp adapter reproduces the official SAM3 contract consumed downstream."""
import json
import os
import sys
import textwrap
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.mac.conftest import GEOMETRY, SAM3CPP_BIN, SAM3CPP_MODEL

sys.path.insert(0, str(GEOMETRY / "backbones"))
from utils.recon_and_seg import seg_sam3_cpp  # noqa: E402

H, W, N = 24, 32, 3


def fake_binary(tmp_path, detections):
    """An executable speaking atas_sam3_track's protocol; detections: {prompt_index: [(frame, id, score, box)]}."""
    script = tmp_path / "fake_sam3.py"
    script.write_text(textwrap.dedent(f"""\
        import json, sys
        from pathlib import Path
        import numpy as np
        from PIL import Image
        args = sys.argv[1:]
        out = Path(args[args.index('--out') + 1]); (out / 'masks').mkdir(parents=True)
        prompts = [args[i + 1] for i, a in enumerate(args) if a == '--prompt']
        frames = sorted(Path(args[args.index('--frames') + 1]).glob('*.png'))
        assert len(frames) == {N}, frames
        rows = []
        detections = {json.dumps({str(k): v for k, v in detections.items()})}
        for p, prompt in enumerate(prompts):
            for frame, inst, score, box in detections.get(str(p), []):
                x0, y0, x1, y1 = box
                mask = np.zeros(({H}, {W}), np.uint8); mask[y0:y1, x0:x1] = 255
                name = f"p{{p:02d}}_f{{frame:05d}}_i{{inst:04d}}.png"
                Image.fromarray(mask).save(out / 'masks' / name)
                rows.append(dict(prompt_index=p, prompt=prompt, frame=frame, instance_id=inst, score=score,
                                 iou_score=score, box_xyxy=[x0, y0, x1, y1], mask='masks/' + name))
        rows.append({{'summary': {{'frames': len(frames), 'passes': len(prompts), 'detections': len(rows),
                     'load_s': 0.1, 'track_s': 0.2, 'sec_per_frame_max': 0.1, 'use_gpu': '--no-gpu' not in args,
                     'peak_footprint_bytes': 123}}}})
        (out / 'tracks.jsonl').write_text(''.join(json.dumps(r) + '\\n' for r in rows))
        """))
    wrapper = tmp_path / "fake_sam3"  # a #! line cannot hold an interpreter path with spaces
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    wrapper.chmod(0o755)
    script = wrapper
    model = tmp_path / "model.ggml"
    model.write_bytes(b"x")
    return seg_sam3_cpp.init_sam3_video(str(script), str(model))


def video():
    return np.zeros((N, H, W, 3), np.uint8)


def test_schema_conversion(tmp_path):
    from scripts.preprocess.avwm_entity_mask_export import export_entity_masks
    predictor = fake_binary(tmp_path, {0: [(f, 0, .9, [2, 3, 10, 12]) for f in range(N)] + [(1, 1, .8, [20, 2, 30, 8])],
                                       1: [(f, 0, .95, [12, 14, 20, 22]) for f in range(N)]})
    report = {}
    mask, frames = seg_sam3_cpp.run_sam3_video(video(), predictor, ["cat", " dog "], report=report)
    assert mask.shape == (N, H, W) and mask.dtype == np.bool_
    assert mask[0, 3:12, 2:10].all() and mask[0, 14:22, 12:20].all() and mask[1, 2:8, 20:30].all()
    assert not mask[0, 2:8, 20:30].any()
    ids = sorted({(d["keyword"], d["id"]) for f in frames for d in f})
    # Official contract: ids are unique across keywords (offset by the previous keyword's max id + 1).
    assert ids == [("cat", 0), ("cat", 1), ("dog", 2)]
    first = frames[0][0]
    assert set(first) == {"id", "keyword", "score", "box_xyxy", "mask"} and first["box_xyxy"].dtype == np.float32
    assert first["mask"].shape == (H, W) and report["peak_footprint_bytes"] == 123
    export_entity_masks(frames, tmp_path / "l0", frame_shape=(H, W))
    saved = json.loads((tmp_path / "l0" / "instance_tracks.json").read_text())
    assert saved["frame_count"] == N and saved["tracks"] and all(not t["identity_verified"] for t in saved["tracks"].values())


def test_empty_tracks(tmp_path):
    from scripts.preprocess.avwm_entity_mask_export import export_entity_masks
    predictor = fake_binary(tmp_path, {})
    mask, frames = seg_sam3_cpp.run_sam3_video(video(), predictor, ["unicorn"])
    assert not mask.any() and frames == [[] for _ in range(N)]
    export_entity_masks(frames, tmp_path / "l0", frame_shape=(H, W))
    assert json.loads((tmp_path / "l0" / "instance_tracks.json").read_text())["segmentation_status"] == "no_tracks_detected"


def test_backend_switch(tmp_path, monkeypatch):
    from pathlib import Path as P
    from atas_geometry import device as accel
    from atas_geometry.reconstruct import vista4d as v
    captured = {}
    class Process:
        def __init__(self, command, **kwargs): captured["command"] = command; self.stdout = iter(["ok\n"])
        def wait(self): return 0
    monkeypatch.setattr(v.subprocess, "Popen", Process)
    kwargs = dict(python_bin=P("python"), repo_dir=tmp_path, video=tmp_path / "v.mp4", output_dir=tmp_path / "l0",
                  keywords=["cat"], gpu_index=0, log_path=tmp_path / "log", pi3_model_id="pi3x",
                  sam3_checkpoint=tmp_path / "m.ggml", width=1280, height=720, num_frames=48,
                  pi3_pixel_limit=255000, pi3_vo_chunk_size=48, pi3_vo_overlap=6, low_vram=True)
    monkeypatch.delenv("GEOMETRY_DEVICE", raising=False)
    monkeypatch.setenv("GEOMETRY_SAM3CPP_BIN", "/opt/bin/atas_sam3_track")
    for backend, no_gpu, expected in [("", "0", ["--sam3_implementation", "cpp"]), ("cpp", "1", ["--sam3_cpp_no_gpu"]),
                                      ("official", "0", None)]:
        monkeypatch.setenv("GEOMETRY_SAM3_BACKEND", backend)
        monkeypatch.setenv("GEOMETRY_SAM3CPP_NO_GPU", no_gpu)
        try:
            v.run_recon_and_seg(**kwargs)
        except Exception:
            pass
        command = captured["command"]
        if expected is None:
            assert "--sam3_implementation" not in command
        else:
            joined = " ".join(command)
            assert " ".join(expected) in joined and "--sam3_cpp_binary /opt/bin/atas_sam3_track" in joined
            assert command[command.index("--sam3_cpp_model") + 1] == str(tmp_path / "m.ggml")
    monkeypatch.setenv("GEOMETRY_DEVICE", "cuda"); monkeypatch.delenv("GEOMETRY_SAM3_BACKEND")
    assert accel.sam3_backend() == "official"


@pytest.mark.skipif(not (SAM3CPP_BIN.exists() and SAM3CPP_MODEL.exists()), reason="sam3.cpp build/weights absent")
def test_real_binary_tracks_a_person(tmp_path):
    from skimage import data
    image = np.array(Image.fromarray(data.astronaut()).resize((256, 256)))
    clip = np.stack([np.roll(image, 4 * i, axis=1) for i in range(2)])
    predictor = seg_sam3_cpp.init_sam3_video(str(SAM3CPP_BIN), str(SAM3CPP_MODEL))
    mask, frames = seg_sam3_cpp.run_sam3_video(clip, predictor, ["person"])
    assert all(len(f) >= 1 for f in frames) and .1 < mask.mean() < .9
