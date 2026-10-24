from pathlib import Path

import pytest

from atas_geometry.service.resource_errors import ResourceUnavailableError, native_resource_code
from tests.mac.conftest import darwin_only

pytestmark = darwin_only


@pytest.mark.parametrize("message,code", [
    ("torch.OutOfMemoryError: CUDA out of memory. GPU 0 is full.", "cuda_out_of_memory"),
    ("RuntimeError: cusolver error: CUSOLVER_STATUS_INTERNAL_ERROR, when calling `cusolverDnCreate(handle)`",
     "cuda_linalg_initialization_failed"),
    ("RuntimeError: MPS backend out of memory (MPS allocated: 30.00 GB, other allocations: 1.00 GB, max allowed: 36.00 GB).",
     "cuda_out_of_memory"),
])
def test_vista_failure_is_structured_resource_on_mps(tmp_path, monkeypatch, message, code):
    """Counterpart of the upstream CUDA test: same structured retryable failure, MPS child environment."""
    from atas_geometry.reconstruct import vista4d as v
    monkeypatch.delenv("GEOMETRY_DEVICE", raising=False)
    captured = {}
    class Process:
        def __init__(self, command, **kwargs):
            captured.update(command=command, **kwargs)
            self.stdout = iter([message + "\n"])
        def wait(self): return 1
    monkeypatch.setattr(v.subprocess, "Popen", Process)
    with pytest.raises(ResourceUnavailableError) as error:
        v.run_recon_and_seg(python_bin=Path("python"), repo_dir=tmp_path, video=tmp_path / "video.mp4",
            output_dir=tmp_path / "l0", keywords=["cat"], gpu_index=0, log_path=tmp_path / "recon.log",
            pi3_model_id="pi3x", sam3_checkpoint=tmp_path / "sam3-f16.ggml", width=1280, height=720,
            num_frames=150, pi3_pixel_limit=255000, pi3_vo_chunk_size=24, pi3_vo_overlap=6, low_vram=True)
    assert error.value.fields()["retryable"] and error.value.error_code == code
    env = captured["env"]
    assert "CUDA_VISIBLE_DEVICES" not in env and env["GEOMETRY_DEVICE"] == "mps"
    assert env["PYTORCH_ENABLE_MPS_FALLBACK"] == "1" and env["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "1.0"
    command = captured["command"]
    assert command[command.index("--num_frames") + 1] == "150"
    assert command[command.index("--sam3_implementation") + 1] == "cpp"
    assert command[command.index("--sam3_cpp_model") + 1] == str(tmp_path / "sam3-f16.ggml")


def test_mps_oom_is_not_confused_with_geometry_rejection():
    assert native_resource_code("MPS backend out of memory") == "cuda_out_of_memory"
    assert native_resource_code("registration rejected: too few SIFT matches") is None
