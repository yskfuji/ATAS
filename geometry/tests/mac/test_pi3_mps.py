"""AC11: Pi3 attention on MPS matches CPU in fp32; the CUDA bf16 flash path is unchanged."""
import ast
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.mac.conftest import GEOMETRY

sys.path.insert(0, str(GEOMETRY / "backbones"))
from pi3.models.layers import attention as A  # noqa: E402
from torch.nn.attention import SDPBackend  # noqa: E402

mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS required")
TOLERANCE = 1e-5


def _compare(module, *inputs, **kwargs):
    torch.manual_seed(0)
    module = module.eval()
    with torch.no_grad():
        expected = module(*inputs, **kwargs)
        moved = module.to("mps")
        actual = moved(*[x.to("mps") for x in inputs],
                       **{k: (v.to("mps") if isinstance(v, torch.Tensor) else v) for k, v in kwargs.items()}).cpu()
    abs_err = float((actual - expected).abs().max())
    rel_err = abs_err / float(expected.abs().max())
    return abs_err, rel_err


@mps
@pytest.mark.parametrize("cls", ["FlashAttention", "MemEffAttention", "Attention"])
def test_attention_mps_matches_cpu(cls):
    torch.manual_seed(1)
    module = getattr(A, cls)(dim=128, num_heads=8, qkv_bias=True)
    x = torch.randn(2, 300, 128)
    abs_err, rel_err = _compare(module, x)
    assert abs_err <= TOLERANCE and rel_err <= TOLERANCE, (abs_err, rel_err)


@mps
@pytest.mark.parametrize("cls", ["FlashAttentionRope", "MemEffAttentionRope"])
def test_rope_attention_mps_matches_cpu(cls):
    torch.manual_seed(2)
    module = getattr(A, cls)(dim=128, num_heads=8, qkv_bias=True, rope=None)
    x = torch.randn(2, 300, 128)
    abs_err, rel_err = _compare(module, x, xpos=None)
    assert abs_err <= TOLERANCE and rel_err <= TOLERANCE, (abs_err, rel_err)


def test_cuda_bf16_still_forces_flash():
    """On a CUDA tensor every patched condition reduces to upstream's, and the non-bf16 kernel list is upstream's."""
    cuda_q = SimpleNamespace(device=SimpleNamespace(type="cuda"))
    assert A._fallback_backends(cuda_q) == [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
    path = "geometry/backbones/pi3/models/layers/attention.py"
    upstream = subprocess.run(["git", "show", f"b86aab046368:{path}"], cwd=GEOMETRY.parent, capture_output=True,
                              text=True, check=True).stdout
    fork = (GEOMETRY.parent / path).read_text()
    # Substitute what a CUDA tensor evaluates the new expressions to; the result must be upstream.
    fork_tree = ast.parse(fork)
    fork_tree.body = [n for n in fork_tree.body if not (isinstance(n, ast.FunctionDef) and n.name == "_fallback_backends")]
    reduced = ast.unparse(fork_tree)
    reduced = reduced.replace(" and (q.device.type == 'cuda')", "").replace(" and q.device.type == 'cuda'", "")
    reduced = reduced.replace("nn.attention.sdpa_kernel(_fallback_backends(q))",
                              "nn.attention.sdpa_kernel([SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION])")
    assert ast.dump(ast.parse(reduced)) == ast.dump(ast.parse(upstream))
    assert fork.count("SDPBackend.FLASH_ATTENTION):") == upstream.count("SDPBackend.FLASH_ATTENTION):") == 4


def test_cpu_uses_tiled_exact_kernel():
    assert A._fallback_backends(torch.zeros(1))[0] == SDPBackend.FLASH_ATTENTION


@mps
@pytest.mark.skipif(not (GEOMETRY.parents[1] / "cache/weights/pi3x/model.safetensors").is_file(), reason="Pi3X weights absent")
def test_conditioned_forward_on_mps():
    """The history-conditioned path (with_prior, as anchored registration calls it) runs on MPS and matches CPU."""
    from utils.recon_and_seg.recon_pi3 import init_pi3
    torch.manual_seed(3)
    n, h, w = 3, 112, 154
    images = torch.rand(1, n, 3, h, w)
    depths = torch.rand(1, n, h, w) + 1.0
    poses = torch.eye(4).repeat(1, n, 1, 1)
    poses[0, 1, 0, 3] = .1
    intrinsics = torch.tensor([[100., 0, w / 2], [0, 100., h / 2], [0, 0, 1]]).repeat(1, n, 1, 1)
    mask = torch.tensor([[True, True, False]])
    out = {}
    sys.path.insert(0, str(GEOMETRY.parent / "tools"))
    import crn_launch
    for device in ("cpu", "mps"):
        # One instance per device: Pi3X caches position grids on the device of its first call.
        crn_launch.install()  # upstream draws a scale augmentation at inference; both devices must see the same numbers
        model = init_pi3(str(GEOMETRY.parents[1] / "cache/weights/pi3x"), device=device)
        with torch.inference_mode():
            pred = model(images.to(device), depths=depths.to(device), poses=poses.to(device), intrinsics=intrinsics.to(device),
                         mask_add_depth=mask.to(device), mask_add_pose=mask.to(device), mask_add_ray=mask.to(device), with_prior=True)
        out[device] = {k: pred[k].float().cpu() for k in ("local_points", "camera_poses")}
        del model, pred
    for key in out["cpu"]:
        ref, got = out["cpu"][key], out["mps"][key]
        assert torch.isfinite(got).all()
        assert float((got - ref).abs().max() / ref.abs().max()) < 1e-3, key


@mps
def test_vo_umeyama_survives_large_magnitude_svd():
    """Pi3XVO's chunk alignment on MPS: a large-magnitude covariance (entries ~1e9) whose SVD MPS failed to converge."""
    from pi3.pipe.pi3x_vo_v2 import Pi3XVO
    torch.manual_seed(7)
    src = torch.randn(1, 6, 64, 64, 3) * torch.tensor([300., 120., 600.])
    rot = torch.linalg.qr(torch.randn(3, 3))[0]
    rot = rot * torch.sign(torch.det(rot))
    tgt = 1.3 * src @ rot.T + torch.tensor([4., -2., 9.])
    mask = torch.ones(1, 6, 64, 64, dtype=torch.bool)
    out = Pi3XVO._compute_sim3_umeyama_masked(None, src.to("mps"), tgt.to("mps"), mask.to("mps"), mask.to("mps")).cpu()
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0, :3, :3], 1.3 * rot, atol=1e-3)
