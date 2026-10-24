"""Run a module under common random numbers (evaluation harness only; production code is unchanged).

    GEOMETRY_EVAL_CRN_SEED=<seed> GEOMETRY_EVAL_CRN_LOG=<file.json> python tools/crn_launch.py <module> [args...]

Upstream Pi3X draws a depth-scale augmentation with torch.rand at inference time, so two CPU runs of
the same clip already differ. Here every torch random draw (rand, rand_like, randn, randn_like,
randperm, Tensor.uniform_) comes from one seeded CPU generator and is then moved to the requested
device, so CPU and MPS runs receive identical numbers in the same order. Each process logs that CRN was
installed and a hash of its draw sequence (function, shape, dtype); the evaluation requires the CPU and
MPS sequences to match, which also proves control flow did not diverge between devices. A Pi3 model
run in training mode aborts. GEOMETRY_EVAL_NATIVE_RNG=1 disables the patch.
"""
import atexit
import hashlib
import json
import os
import runpy
import sys

SEED = 20261019
_STATE = {"native": None, "log": [], "installed": False}


def _record(name, values):
    _STATE["log"].append(f"{name}:{tuple(values.shape)}:{values.dtype}")
    return values


def install(seed: int | None = None):
    """(Re)start the shared stream at `seed`; repeated installs never wrap an earlier patch."""
    import torch
    seed = int(os.environ.get("GEOMETRY_EVAL_CRN_SEED", SEED)) if seed is None else seed
    shared = torch.Generator().manual_seed(seed)
    if _STATE["native"] is None:
        _STATE["native"] = {name: getattr(torch, name) for name in ("rand", "randn", "randperm")}
        _STATE["native"]["uniform_"] = torch.Tensor.uniform_
    native = _STATE["native"]
    _STATE["log"].clear()

    def placed(values, device, requires_grad=False):
        values = values.to(device) if device is not None else values
        return values.requires_grad_(requires_grad) if requires_grad else values

    def rand(*size, device=None, dtype=None, requires_grad=False, generator=None, **kw):  # noqa: ARG001
        return placed(_record("rand", native["rand"](*size, generator=shared, dtype=dtype or torch.float32, **kw)), device, requires_grad)

    def randn(*size, device=None, dtype=None, requires_grad=False, generator=None, **kw):  # noqa: ARG001
        return placed(_record("randn", native["randn"](*size, generator=shared, dtype=dtype or torch.float32, **kw)), device, requires_grad)

    def rand_like(t, *, dtype=None, device=None, **kw):  # noqa: ARG001
        return placed(_record("rand_like", native["rand"](tuple(t.shape), generator=shared, dtype=dtype or t.dtype)), device or t.device)

    def randn_like(t, *, dtype=None, device=None, **kw):  # noqa: ARG001
        return placed(_record("randn_like", native["randn"](tuple(t.shape), generator=shared, dtype=dtype or t.dtype)), device or t.device)

    def randperm(n, *, device=None, dtype=None, generator=None, **kw):  # noqa: ARG001
        return placed(_record("randperm", native["randperm"](n, generator=shared, dtype=dtype or torch.int64)), device)

    def uniform_(self, a=0.0, b=1.0, *, generator=None):  # noqa: ARG001
        dtype = self.dtype if self.dtype.is_floating_point and self.dtype != torch.float64 else torch.float32
        values = _record("uniform_", native["rand"](tuple(self.shape), generator=shared, dtype=dtype))
        return self.copy_((values * (b - a) + a).to(self.device))

    torch.rand, torch.randn, torch.rand_like, torch.randn_like, torch.randperm = rand, randn, rand_like, randn_like, randperm
    torch.Tensor.uniform_ = uniform_
    _STATE["installed"] = True
    _guard_training_mode()


def _guard_training_mode():
    """Abort if Pi3X is ever run in training mode (dropout or stochastic depth would draw unseen randomness)."""
    try:
        from utils.recon_and_seg import recon_pi3
    except ImportError:
        return
    original = getattr(recon_pi3.init_pi3, "_crn_original", recon_pi3.init_pi3)

    def init_pi3(*args, **kwargs):
        model = original(*args, **kwargs)

        def check(module, inputs):
            if module.training:
                raise RuntimeError("Pi3X forward in training mode under evaluation")
        model.register_forward_pre_hook(check)
        return model
    init_pi3._crn_original = original
    recon_pi3.init_pi3 = init_pi3


def draw_summary() -> dict:
    sequence = "\n".join(_STATE["log"])
    return {"crn_installed": _STATE["installed"], "draws": len(_STATE["log"]),
            "draw_sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "seed": int(os.environ.get("GEOMETRY_EVAL_CRN_SEED", SEED))}


def _write_log():
    path = os.environ.get("GEOMETRY_EVAL_CRN_LOG")
    if path:
        with open(path, "w") as handle:
            json.dump(draw_summary(), handle, indent=2)


if __name__ == "__main__":
    if os.environ.get("GEOMETRY_EVAL_NATIVE_RNG") != "1":
        sys.path.insert(0, os.getcwd())  # the recon script's cwd holds utils/ (training-mode guard)
        install()
        atexit.register(_write_log)
    module, sys.argv = sys.argv[1], [sys.argv[1], *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
