from __future__ import annotations

import pytest

from recon.make_figures import (
    _aggregate_family_mode_metrics,
    _compute_difficulty_indices,
    _matched_primitive_rows,
    _select_family_representatives,
    _select_hero_report,
)


def test_select_hero_report_prefers_visual_family_within_quality_margin() -> None:
    reports = [
        {
            "scan_id": "shaft_scan",
            "family": "shaft",
            "registration_success": True,
            "primitive_f1": 1.0,
            "coverage": 0.95,
            "chamfer_mm": 1.3,
            "rot_err_deg": 1.0,
            "trans_err_mm": 0.5,
        },
        {
            "scan_id": "pipe_scan",
            "family": "pipe_elbow",
            "registration_success": True,
            "primitive_f1": 1.0,
            "coverage": 0.88,
            "chamfer_mm": 2.2,
            "rot_err_deg": 1.2,
            "trans_err_mm": 0.7,
        },
    ]

    selected = _select_hero_report(reports)

    assert selected["scan_id"] == "pipe_scan"


def test_select_hero_report_keeps_best_quality_when_gap_is_large() -> None:
    reports = [
        {
            "scan_id": "shaft_scan",
            "family": "shaft",
            "registration_success": True,
            "primitive_f1": 1.0,
            "coverage": 0.95,
            "chamfer_mm": 1.3,
            "rot_err_deg": 1.0,
            "trans_err_mm": 0.5,
        },
        {
            "scan_id": "pipe_scan",
            "family": "pipe_elbow",
            "registration_success": True,
            "primitive_f1": 0.96,
            "coverage": 0.61,
            "chamfer_mm": 5.4,
            "rot_err_deg": 3.9,
            "trans_err_mm": 2.7,
        },
    ]

    selected = _select_hero_report(reports)

    assert selected["scan_id"] == "shaft_scan"


def test_select_family_representatives_returns_one_per_family_on_split() -> None:
    records = [
        {"scan_id": "flange_train", "family": "flange", "split": "train"},
        {"scan_id": "flange_test", "family": "flange", "split": "test"},
        {"scan_id": "shaft_test", "family": "shaft", "split": "test"},
        {"scan_id": "bracket_test", "family": "bracket", "split": "test"},
        {"scan_id": "pipe_test", "family": "pipe_elbow", "split": "test"},
        {"scan_id": "plate_test", "family": "plate_with_holes", "split": "test"},
    ]

    selected = _select_family_representatives(records, split="test", max_items=5)

    assert [record["family"] for record in selected] == [
        "flange",
        "shaft",
        "bracket",
        "pipe_elbow",
        "plate_with_holes",
    ]
    assert all(record["split"] == "test" for record in selected)


def test_aggregate_family_mode_metrics_returns_success_and_medians() -> None:
    modes = {
        "full": {
            "reports": [
                {"family": "flange", "registration_success": True, "rot_err_deg": 1.0, "trans_err_mm": 0.5, "chamfer_mm": 3.0, "runtime_sec": 0.4},
                {"family": "flange", "registration_success": False, "rot_err_deg": 3.0, "trans_err_mm": 1.5, "chamfer_mm": 5.0, "runtime_sec": 0.6},
                {"family": "shaft", "registration_success": True, "rot_err_deg": 0.8, "trans_err_mm": 0.3, "chamfer_mm": 2.0, "runtime_sec": 0.5},
            ]
        },
        "no_refine": {
            "reports": [
                {"family": "flange", "registration_success": True, "rot_err_deg": 2.0, "trans_err_mm": 0.7, "chamfer_mm": 4.0, "runtime_sec": 0.7},
                {"family": "shaft", "registration_success": True, "rot_err_deg": 1.1, "trans_err_mm": 0.4, "chamfer_mm": 2.4, "runtime_sec": 0.55},
            ]
        },
    }

    aggregated = _aggregate_family_mode_metrics(modes)

    assert aggregated["full"]["flange"]["registration_success_rate"] == 0.5
    assert aggregated["full"]["flange"]["median_rot_err_deg"] == 2.0
    assert aggregated["full"]["flange"]["median_runtime_sec"] == 0.5
    assert aggregated["no_refine"]["shaft"]["registration_success_rate"] == 1.0


def test_compute_difficulty_indices_increases_with_harder_capture_profile() -> None:
    records = [
        {"scan_id": "easy", "noise_profile": {"noise_std_mm": 0.1, "outlier_ratio": 0.01, "occlusion_dropout": 0.05, "views_used": 5}},
        {"scan_id": "mid", "noise_profile": {"noise_std_mm": 0.3, "outlier_ratio": 0.02, "occlusion_dropout": 0.10, "views_used": 4}},
        {"scan_id": "hard", "noise_profile": {"noise_std_mm": 0.5, "outlier_ratio": 0.03, "occlusion_dropout": 0.18, "views_used": 2}},
    ]

    scores = _compute_difficulty_indices(records)

    assert scores["easy"] < scores["mid"] < scores["hard"]


def test_matched_primitive_rows_preserve_confidence_type_and_error() -> None:
    predicted = [
        {
            "primitive_id": "plane_top",
            "type": "plane",
            "center": [0.0, 0.0, 0.006],
            "axis": None,
            "normal": [0.0, 0.0, 1.0],
            "radius": None,
            "height": None,
            "offset": -0.006,
            "dimensions": {"size_u_m": 0.10, "size_v_m": 0.08},
            "role": "surface",
            "confidence": 0.92,
            "support_size": 1200,
        },
        {
            "primitive_id": "cyl_main",
            "type": "cylinder",
            "center": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "normal": None,
            "radius": 0.02,
            "height": 0.05,
            "offset": None,
            "dimensions": {"diameter_m": 0.04, "height_m": 0.05},
            "role": "shaft",
            "confidence": 0.81,
            "support_size": 900,
        },
    ]
    ground_truth = [
        {
            "primitive_id": "plane_gt",
            "type": "plane",
            "center": [0.0, 0.0, 0.006],
            "axis": None,
            "normal": [0.0, 0.0, 1.0],
            "radius": None,
            "height": None,
            "offset": -0.006,
            "dimensions": {"size_u_m": 0.10, "size_v_m": 0.08},
            "role": "surface",
            "confidence": None,
            "support_size": None,
        },
        {
            "primitive_id": "cyl_gt",
            "type": "cylinder",
            "center": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "normal": None,
            "radius": 0.021,
            "height": 0.05,
            "offset": None,
            "dimensions": {"diameter_m": 0.042, "height_m": 0.05},
            "role": "shaft",
            "confidence": None,
            "support_size": None,
        },
    ]

    rows = _matched_primitive_rows(predicted, ground_truth)

    assert len(rows) == 2
    plane_row = next(row for row in rows if row["primitive_type"] == "plane")
    cylinder_row = next(row for row in rows if row["primitive_type"] == "cylinder")

    assert plane_row["confidence"] == 0.92
    assert plane_row["ground_truth_id"] == "plane_gt"
    assert plane_row["dimension_error_mm"] == 0.0
    assert cylinder_row["confidence"] == 0.81
    assert cylinder_row["ground_truth_id"] == "cyl_gt"
    assert cylinder_row["dimension_error_mm"] == pytest.approx(1.0)
