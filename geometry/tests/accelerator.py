"""Device for upstream GPU tests: CUDA exactly as before when present, otherwise Apple MPS."""
import os

import torch


def render_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def explicit_test_gpu():
    """Device for tests gated on ATAS_TEST_GPU: cuda:<id> on NVIDIA hosts, mps on Apple Silicon."""
    if torch.cuda.is_available():
        return torch.device("cuda:" + os.environ["ATAS_TEST_GPU"])
    return torch.device("mps")


def available(name):
    return {"cpu": True, "cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}[name]
