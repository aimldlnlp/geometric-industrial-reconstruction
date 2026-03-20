from __future__ import annotations

import numpy as np

from recon.geometry import make_transform, transform_delta_deg, transform_delta_mm
from recon.pipeline import (
    RegistrationCandidate,
    _assess_refinement_candidate,
    _pipe_elbow_angle_candidates,
    _plate_snap_candidates,
)
from recon.types import ScanRecord, ScanReport


PIPELINE_CFG = {
    "coverage_tol_mm": 2.5,
    "success_rot_deg": 2.0,
    "success_trans_mm": 3.0,
    "refine_min_rel_improvement": 0.02,
    "refine_min_abs_improvement_mm": 0.10,
    "refine_max_coverage_drop": 0.01,
    "refine_max_delta_deg": 12.0,
    "refine_max_delta_mm": 8.0,
    "refine_strong_gain_rel": 0.10,
    "plate_snap_max_candidates": 16,
    "pipe_elbow_angle_search_deg": 3.0,
    "pipe_elbow_angle_step_deg": 1.0,
}


def _record_for_family(family: str) -> ScanRecord:
    primitive_gt: list[dict] = []
    if family == "flange":
        primitive_gt = [
            {
                "primitive_id": "cyl_hub",
                "type": "cylinder",
                "center": [0.0, 0.0, 0.0],
                "axis": [0.0, 0.0, 1.0],
                "radius": 0.02,
                "height": 0.06,
                "dimensions": {"diameter_m": 0.04, "height_m": 0.06},
            },
            {
                "primitive_id": "plane_top",
                "type": "plane",
                "center": [0.0, 0.0, 0.008],
                "normal": [0.0, 0.0, 1.0],
                "offset": -0.008,
                "dimensions": {"size_u_m": 0.10, "size_v_m": 0.10},
            },
            {
                "primitive_id": "plane_bottom",
                "type": "plane",
                "center": [0.0, 0.0, -0.008],
                "normal": [0.0, 0.0, 1.0],
                "offset": 0.008,
                "dimensions": {"size_u_m": 0.10, "size_v_m": 0.10},
            },
        ]
    elif family == "pipe_elbow":
        primitive_gt = [
            {
                "primitive_id": "cyl_x",
                "type": "cylinder",
                "center": [0.025, 0.0, 0.0],
                "axis": [1.0, 0.0, 0.0],
                "radius": 0.015,
                "height": 0.10,
                "dimensions": {"diameter_m": 0.03, "height_m": 0.10},
            },
            {
                "primitive_id": "cyl_z",
                "type": "cylinder",
                "center": [0.0, 0.0, 0.025],
                "axis": [0.0, 0.0, 1.0],
                "radius": 0.015,
                "height": 0.10,
                "dimensions": {"diameter_m": 0.03, "height_m": 0.10},
            },
            {
                "primitive_id": "sphere_corner",
                "type": "sphere",
                "center": [0.0, 0.0, 0.0],
                "radius": 0.017,
                "dimensions": {"diameter_m": 0.034},
            },
        ]
    elif family == "plate_with_holes":
        primitive_gt = [
            {
                "primitive_id": "plane_top",
                "type": "plane",
                "center": [0.0, 0.0, 0.006],
                "normal": [0.0, 0.0, 1.0],
                "offset": -0.006,
                "dimensions": {"size_u_m": 0.10, "size_v_m": 0.07},
            },
            {
                "primitive_id": "plane_bottom",
                "type": "plane",
                "center": [0.0, 0.0, -0.006],
                "normal": [0.0, 0.0, 1.0],
                "offset": 0.006,
                "dimensions": {"size_u_m": 0.10, "size_v_m": 0.07},
            },
        ]
    return ScanRecord(
        part_id=f"{family}_part_000",
        family=family,
        scan_id=f"{family}_part_000_scan_000",
        split="test",
        cloud_path="scans/test/example.ply",
        mesh_path="meshes/example.ply",
        reference_cloud_path="reference_clouds/example.ply",
        gt_pose=np.eye(4).tolist(),
        primitive_gt=primitive_gt,
        noise_profile={},
        metadata={},
    )


def _candidate(
    label: str,
    transform: np.ndarray,
    chamfer_mm: float,
    coverage: float,
    reference_transform: np.ndarray | None = None,
) -> RegistrationCandidate:
    delta_deg = 0.0 if reference_transform is None else transform_delta_deg(reference_transform, transform)
    delta_mm = 0.0 if reference_transform is None else transform_delta_mm(reference_transform, transform)
    return RegistrationCandidate(
        label=label,
        stage="init" if reference_transform is None else "refine",
        transform=transform,
        aligned_points=np.zeros((0, 3), dtype=float),
        chamfer_mm=chamfer_mm,
        coverage=coverage,
        objective=0.0,
        transform_delta_deg=delta_deg,
        transform_delta_mm=delta_mm,
    )


def test_refinement_gate_rejects_large_pose_jump_with_tiny_gain() -> None:
    record = _record_for_family("bracket")
    init_candidate = _candidate("init_identity", np.eye(4), chamfer_mm=4.0, coverage=0.82)
    refined_transform = make_transform([15.0, 0.0, 12.0], [0.010, 0.0, 0.0])
    refined_candidate = _candidate(
        "refine_point_to_plane",
        refined_transform,
        chamfer_mm=3.8,
        coverage=0.82,
        reference_transform=init_candidate.transform,
    )

    accepted, reason = _assess_refinement_candidate(init_candidate, refined_candidate, record, PIPELINE_CFG)

    assert not accepted
    assert reason == "pose_jump"


def test_refinement_gate_accepts_material_improvement() -> None:
    record = _record_for_family("bracket")
    init_candidate = _candidate("init_identity", np.eye(4), chamfer_mm=4.0, coverage=0.78)
    refined_transform = make_transform([2.5, -1.0, 1.5], [0.001, -0.0005, 0.0005])
    refined_candidate = _candidate(
        "refine_point_to_plane",
        refined_transform,
        chamfer_mm=3.1,
        coverage=0.78,
        reference_transform=init_candidate.transform,
    )

    accepted, reason = _assess_refinement_candidate(init_candidate, refined_candidate, record, PIPELINE_CFG)

    assert accepted
    assert reason is None


def test_flange_flip_candidate_is_rejected() -> None:
    record = _record_for_family("flange")
    init_candidate = _candidate("init_identity", np.eye(4), chamfer_mm=4.1, coverage=0.80)
    refined_transform = make_transform([180.0, 0.0, 0.0], [0.0, 0.0, 0.02])
    refined_candidate = _candidate(
        "refine_point_to_plane",
        refined_transform,
        chamfer_mm=3.9,
        coverage=0.80,
        reference_transform=init_candidate.transform,
    )

    accepted, reason = _assess_refinement_candidate(init_candidate, refined_candidate, record, PIPELINE_CFG)

    assert not accepted
    assert reason == "dominant_axis_flip"


def test_pipe_elbow_swap_candidate_is_rejected() -> None:
    record = _record_for_family("pipe_elbow")
    init_candidate = _candidate("init_identity", np.eye(4), chamfer_mm=3.5, coverage=0.84)
    refined_transform = make_transform([0.0, 180.0, 0.0], [0.0, 0.0, 0.0])
    refined_candidate = _candidate(
        "refine_point_to_plane",
        refined_transform,
        chamfer_mm=3.2,
        coverage=0.84,
        reference_transform=init_candidate.transform,
    )

    accepted, reason = _assess_refinement_candidate(init_candidate, refined_candidate, record, PIPELINE_CFG)

    assert not accepted
    assert reason == "pipe_elbow_axis_swap"


def test_plate_snap_candidates_include_translation_correction() -> None:
    record = _record_for_family("plate_with_holes")
    x_values = np.linspace(-0.05, 0.05, 11)
    y_values = np.linspace(-0.035, 0.035, 9)
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    top = np.column_stack([x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, 0.006)])
    bottom = np.column_stack([x_grid.ravel(), y_grid.ravel(), np.full(x_grid.size, -0.006)])
    target_points = np.vstack([top, bottom])
    source_points = target_points + np.asarray([0.004, -0.003, 0.0])
    base_candidate = _candidate("init_global", np.eye(4), chamfer_mm=4.5, coverage=0.55)
    base_candidate.aligned_points = source_points

    candidates = _plate_snap_candidates(base_candidate, source_points, target_points, record, PIPELINE_CFG)
    translations = [candidate.transform[:3, 3] for candidate in candidates]

    assert candidates
    assert any(np.allclose(translation[:2], [-0.004, 0.003], atol=1.5e-3) for translation in translations)


def test_pipe_elbow_angle_candidates_generate_small_local_rotations() -> None:
    record = _record_for_family("pipe_elbow")
    samples = np.linspace(0.0, 0.04, 8)
    leg_x = np.column_stack([samples, np.zeros_like(samples), np.zeros_like(samples)])
    leg_z = np.column_stack([np.zeros_like(samples), np.zeros_like(samples), samples])
    target_points = np.vstack([leg_x, leg_z])
    base_transform = make_transform([0.0, 2.0, 0.0], [0.0, 0.0, 0.0])
    base_candidate = _candidate("init_global", base_transform, chamfer_mm=3.0, coverage=0.79)
    base_candidate.aligned_points = target_points

    candidates = _pipe_elbow_angle_candidates(base_candidate, target_points, target_points, record, PIPELINE_CFG)

    assert candidates
    assert all(candidate.stage == "family_refine" for candidate in candidates)
    assert any(candidate.transform_delta_deg <= PIPELINE_CFG["pipe_elbow_angle_search_deg"] + 1e-6 for candidate in candidates)


def test_scan_report_serializes_registration_diagnostics() -> None:
    report = ScanReport(
        scan_id="scan_000",
        family="flange",
        mode="full",
        registration_success=True,
        rot_err_deg=0.5,
        trans_err_mm=0.4,
        chamfer_mm=2.3,
        pre_refine_chamfer_mm=2.8,
        chamfer_improvement_pct=17.8,
        coverage=0.83,
        primitive_f1=1.0,
        primitive_precision=1.0,
        primitive_recall=1.0,
        dimension_mae_mm=0.0,
        runtime_sec=0.42,
        predicted_primitives=[],
        reconstruction_mesh_path=None,
        stage_path="artifacts/stages.npz",
        raw_rot_err_deg=0.5,
        raw_trans_err_mm=0.4,
        symmetry_mode="continuous_axial_symmetry",
        selected_candidate="refine_coarse_to_fine",
        refine_accepted=True,
        refine_reject_reason=None,
        init_chamfer_mm=2.8,
        candidate_scores=[{"label": "init_global", "accepted": True}],
        transform_delta_deg=1.2,
        transform_delta_mm=0.7,
        notes=["Primitive recovery used reference-conditioned geometric priors."],
    )

    payload = report.to_dict()

    assert payload["selected_candidate"] == "refine_coarse_to_fine"
    assert payload["refine_accepted"] is True
    assert payload["init_chamfer_mm"] == 2.8
    assert payload["candidate_scores"] == [{"label": "init_global", "accepted": True}]
    assert payload["transform_delta_deg"] == 1.2
    assert payload["transform_delta_mm"] == 0.7
