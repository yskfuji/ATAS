"""A12: the AC12 rotation metric is numerically stable; the original arccos form is kept only for side-by-side reporting."""
import json
import sys

import numpy as np
import pytest

from tests.mac.conftest import GEOMETRY

REPO = GEOMETRY.parent
sys.path[:0] = [str(REPO / "tools"), str(GEOMETRY / "backbones"), str(GEOMETRY)]
import geometry_eval as G  # noqa: E402

EVAL = REPO.parent / "cache/eval/mac-20261019"


def rotation(axis, degrees):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    t = np.radians(degrees)
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def pose(R):
    P = np.eye(4)
    P[:3, :3] = R
    return P.astype(np.float32)


@pytest.mark.parametrize("degrees", [1e-4, 1e-3, 0.01, 0.05, 0.3, 1.0, 10.0, 90.0, 179.0, 180.0])
def test_stable_metric_recovers_known_angles_from_float32_matrices(degrees):
    rng = np.random.default_rng(int(degrees * 1e4) + 1)
    worst = 0.0
    for _ in range(50):
        base = rotation(rng.normal(size=3), rng.uniform(0, 180))
        delta = rotation(rng.normal(size=3), degrees)
        a, b = pose(base), pose(base @ delta)          # both rounded to float32 independently
        worst = max(worst, abs(G.rotation_deg(a, b) - degrees))
    assert worst < 1e-5, worst


def test_original_formula_is_ill_conditioned_on_the_same_input():
    rng = np.random.default_rng(3)
    errors = []
    for _ in range(100):
        base = rotation(rng.normal(size=3), rng.uniform(0, 180))
        a, b = pose(base), pose(base @ rotation(rng.normal(size=3), 0.01))
        errors.append(abs(G.rotation_deg_pre_A12(a, b) - 0.01))
    assert max(errors) > 1e-3 > 1e-5   # the artefact the amendment removes: ~0.02 deg of rounding noise


def test_two_float32_roundings_of_one_rotation_give_zero():
    rng = np.random.default_rng(4)
    for _ in range(50):
        R = rotation(rng.normal(size=3), rng.uniform(0, 180))
        a = pose(R)
        b = pose(R + 1e-8 * rng.normal(size=(3, 3)))   # a different rounding of the same rotation
        assert G.rotation_deg(a, b) < 1e-4


@pytest.mark.parametrize("degrees", [30.0, 90.0, 150.0])
def test_both_forms_agree_when_well_conditioned(degrees):
    R = rotation([1, 2, 3], degrees)
    a, b = pose(np.eye(3)), pose(R)
    assert abs(G.rotation_deg(a, b) - G.rotation_deg_pre_A12(a, b)) < 1e-3


# needs the saved raw runs of the evaluation (not published); skipped when only their JSON summaries are present
@pytest.mark.skipif(not (EVAL / "sintel_temple_2/compare_float32.pre_A12.json").is_file()
                    or not (EVAL / "sintel_temple_2/cpu_float32_crn0/l0/cameras.npz").is_file(), reason="saved evaluation runs absent")
def test_stored_pre_A12_values_are_reproduced_by_the_original_formula_and_corrected_values_by_the_new_one():
    for clip, expected_old in (("sintel_bandage_1", 0.0809), ("sintel_temple_2", 0.3339)):
        base = EVAL / clip
        stored = json.loads((base / "compare_float32.pre_A12.json").read_text())["parity"]["rot_deg"]
        assert abs(stored - expected_old) < 5e-4
        old, new = [], []
        for k in range(3):
            _, ref, _ = G.load_l0(base / G.run_name("cpu", "float32", "crn", k) / "l0")
            _, got, _ = G.load_l0(base / G.run_name("mps", "float32", "crn", k) / "l0")
            old.append(max(G.rotation_deg_pre_A12(a, b) for a, b in zip(ref, got)))
            new.append(max(G.rotation_deg(a, b) for a, b in zip(ref, got)))
        assert abs(max(old) - stored) < 1e-9
        assert max(new) <= max(old) + 1e-9
