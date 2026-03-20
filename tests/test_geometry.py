from __future__ import annotations

import numpy as np

from recon.geometry import (
    apply_transform,
    invert_transform,
    make_transform,
    random_pose,
    relative_transform,
    rotation_error_deg,
    seed_everything,
    transform_delta_deg,
    transform_delta_mm,
    translation_error_mm,
)


def test_transform_inverse_roundtrip() -> None:
    transform = make_transform([12.0, -8.0, 21.0], [0.01, -0.02, 0.03])
    inverse = invert_transform(transform)
    points = np.asarray([[0.0, 0.0, 0.0], [0.01, 0.02, -0.03], [-0.02, 0.01, 0.02]])
    recovered = apply_transform(apply_transform(points, transform), inverse)
    assert np.allclose(points, recovered, atol=1e-8)


def test_pose_error_is_zero_for_identical_transforms() -> None:
    transform = make_transform([5.0, 7.0, -3.0], [0.02, 0.01, -0.04])
    assert rotation_error_deg(transform, transform) == 0.0
    assert translation_error_mm(transform, transform) == 0.0


def test_random_pose_is_seed_deterministic() -> None:
    pose_a = random_pose(seed_everything(7), max_rotation_deg=20.0, max_translation_m=0.05)
    pose_b = random_pose(seed_everything(7), max_rotation_deg=20.0, max_translation_m=0.05)
    assert np.allclose(pose_a, pose_b)


def test_relative_transform_maps_reference_pose_to_candidate_pose() -> None:
    reference = make_transform([4.0, -2.0, 8.0], [0.01, -0.02, 0.005])
    candidate = make_transform([7.0, 3.0, -5.0], [-0.015, 0.01, 0.02])
    delta = relative_transform(reference, candidate)
    points = np.asarray([[0.01, 0.0, 0.0], [0.0, 0.02, -0.01], [-0.01, 0.01, 0.03]])
    via_reference = apply_transform(apply_transform(points, reference), delta)
    direct = apply_transform(points, candidate)
    assert np.allclose(via_reference, direct, atol=1e-8)


def test_transform_delta_helpers_match_pose_error_helpers() -> None:
    reference = make_transform([2.0, -1.0, 5.0], [0.0, 0.01, -0.015])
    candidate = make_transform([5.0, 3.0, -4.0], [0.004, -0.006, 0.012])
    assert transform_delta_deg(reference, candidate) == rotation_error_deg(candidate, reference)
    assert transform_delta_mm(reference, candidate) == translation_error_mm(candidate, reference)
