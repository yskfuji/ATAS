"""The evaluation harness's common random numbers (tools/crn_launch.py) do what A5 relies on."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import crn_launch  # noqa: E402


@pytest.fixture(autouse=True)
def restore():
    saved = {n: getattr(torch, n) for n in ("rand", "randn", "rand_like", "randn_like", "randperm")}
    uniform = torch.Tensor.uniform_
    yield
    for n, f in saved.items():
        setattr(torch, n, f)
    torch.Tensor.uniform_ = uniform


def draws(device):
    crn_launch.install(seed=5)
    t = torch.empty(4, device=device)
    return [torch.rand((2, 3), device=device).cpu(), torch.randn(3, device=device).cpu(),
            torch.rand_like(t).cpu(), torch.randn_like(t).cpu(), torch.randperm(6, device=device).cpu(),
            t.uniform_(-1, 1).cpu()], crn_launch.draw_summary()


def test_same_seed_same_stream_and_restart():
    a, sa = draws("cpu")
    b, sb = draws("cpu")
    assert all(torch.equal(x, y) for x, y in zip(a, b)) and sa["draw_sequence_sha256"] == sb["draw_sequence_sha256"]
    assert sa["crn_installed"] and sa["draws"] == 6


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
def test_cpu_and_mps_receive_identical_numbers():
    a, sa = draws("cpu")
    b, sb = draws("mps")
    assert all(torch.equal(x, y.to(x.dtype)) for x, y in zip(a, b))
    assert sa["draw_sequence_sha256"] == sb["draw_sequence_sha256"]


def test_scale_augmentation_expression_is_covered():
    crn_launch.install(seed=1)
    first = 0.8 + torch.rand((1,), device="cpu") * 0.4  # the upstream scale_aug draw
    crn_launch.install(seed=1)
    assert torch.equal(first, 0.8 + torch.rand((1,), device="cpu") * 0.4)


def test_training_mode_aborts(monkeypatch):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backbones"))
    from utils.recon_and_seg import recon_pi3
    class Tiny(torch.nn.Module):
        def forward(self, x): return x
    monkeypatch.setattr(recon_pi3, "init_pi3", lambda *a, **k: Tiny())
    crn_launch.install(seed=1)
    model = recon_pi3.init_pi3()
    with pytest.raises(RuntimeError, match="training mode"):
        model(torch.ones(1))
    model.eval()
    assert torch.equal(model(torch.ones(1)), torch.ones(1))
