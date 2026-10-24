"""Frame-chunked ConvHead (Pi3X point/conf heads off CUDA): same values as one pass, upstream body kept for CUDA."""
import ast
import subprocess
import sys

import pytest
import torch
import torch.nn as nn

from tests.mac.conftest import GEOMETRY

sys.path.insert(0, str(GEOMETRY / "backbones"))
from pi3.models.layers import conv_head as CH  # noqa: E402

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
WEIGHTS = GEOMETRY.parents[1] / "cache/weights/pi3x/model.safetensors"


def _head(dim_out):
    torch.manual_seed(11)
    return CH.ConvHead(num_features=4, dim_in=1024, projects=nn.Identity(), dim_out=dim_out, dim_proj=1024,
                       dim_upsample=[256, 128, 64], dim_times_res_block_hidden=2, num_res_blocks=2,
                       res_block_norm="group_norm", last_res_blocks=0, last_conv_channels=32, last_conv_size=1).eval()


def _run(head, hidden, chunk, device, **kwargs):
    old = CH.FRAME_CHUNK
    CH.FRAME_CHUNK = chunk
    try:
        with torch.inference_mode():
            return head(hidden, **kwargs)
    finally:
        CH.FRAME_CHUNK = old


def _same(a, b):
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a.shape == b.shape and torch.equal(a, b)


DEVICES = ["cpu", pytest.param("mps", marks=mps)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_out", [[2, 1], [1]])
@pytest.mark.parametrize("chunk", [1, 2, 4, 5])
def test_chunked_equals_single_pass_bitwise(device, dim_out, chunk):
    """Includes a ragged tail (7 frames, chunk 2/4/5) and both output types (ModuleList -> list, one block -> list)."""
    head = _head(dim_out).to(device)
    hidden = torch.randn(7, 8 * 11, 1024, generator=torch.Generator().manual_seed(5)).to(device)
    ref = _run(head, hidden, 0, device, patch_h=8, patch_w=11)
    got = _run(head, hidden, chunk, device, patch_h=8, patch_w=11)
    assert _same(ref, got)


@pytest.mark.parametrize("device", DEVICES)
def test_chunked_with_image_argument(device):
    head = _head([1]).to(device)
    hidden = torch.randn(5, 8 * 11, 1024, generator=torch.Generator().manual_seed(6)).to(device)
    image = torch.rand(5, 3, 112, 154, generator=torch.Generator().manual_seed(7)).to(device)
    assert _same(_run(head, hidden, 0, device, image=image), _run(head, hidden, 2, device, image=image))


def test_few_frames_and_disabled_take_the_single_pass(monkeypatch):
    head = _head([1])
    calls = []
    original = CH.ConvHead._forward_frames
    monkeypatch.setattr(CH.ConvHead, "_forward_frames", lambda self, *a, **k: calls.append(a[0].shape[0]) or original(self, *a, **k))
    hidden = torch.randn(3, 4 * 4, 1024)
    _run(head, hidden, 4, "cpu", patch_h=4, patch_w=4)
    _run(head, hidden, 0, "cpu", patch_h=4, patch_w=4)
    assert calls == [3, 3]
    calls.clear()
    _run(head, torch.randn(7, 4 * 4, 1024), 3, "cpu", patch_h=4, patch_w=4)
    assert calls == [3, 3, 1]


def test_cuda_tensor_takes_the_upstream_single_pass(monkeypatch):
    """A CUDA tensor never reaches the slicing branch (no CUDA needed: the branch condition only reads device.type)."""
    seen = []
    monkeypatch.setattr(CH.ConvHead, "_forward_frames", lambda self, hidden, *a: seen.append(hidden.shape[0]) or "single")
    fake = type("T", (), {"shape": (24, 1, 1), "device": type("D", (), {"type": "cuda"})()})()
    assert CH.ConvHead.forward(_head([1]), fake, None, 4, 4) == "single" and seen == [24]


def test_frame_body_is_upstream_forward():
    """The per-frame body is upstream's forward, unchanged apart from its name."""
    path = "geometry/backbones/pi3/models/layers/conv_head.py"
    upstream = subprocess.run(["git", "show", f"b86aab046368:{path}"], cwd=GEOMETRY.parent, capture_output=True,
                              text=True, check=True).stdout
    def method(source, name):
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "ConvHead")
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
        fn.name = "forward"
        return ast.dump(fn)
    assert method((GEOMETRY.parent / path).read_text(), "_forward_frames") == method(upstream, "forward")


@mps
@pytest.mark.skipif(not WEIGHTS.is_file(), reason="Pi3X weights absent")
def test_real_weights_whole_model_bitwise_on_mps():
    """Pi3X with its real weights: heads chunked into 2 frames vs one pass, bit-identical outputs (random noise fixed)."""
    from utils.recon_and_seg.recon_pi3 import init_pi3
    model = init_pi3(str(GEOMETRY.parents[1] / "cache/weights/pi3x"), device="mps")
    images = torch.rand(1, 6, 3, 112, 154, generator=torch.Generator().manual_seed(8)).to("mps")
    out = {}
    for chunk in (0, 2):
        old, CH.FRAME_CHUNK = CH.FRAME_CHUNK, chunk
        try:
            torch.manual_seed(0)  # the model draws a scale augmentation at inference
            with torch.inference_mode():
                pred = model(images)
        finally:
            CH.FRAME_CHUNK = old
        out[chunk] = {k: pred[k].float().cpu() for k in ("points", "local_points", "conf", "camera_poses")}
    for key in out[0]:
        assert torch.equal(out[0][key], out[2][key]), key
