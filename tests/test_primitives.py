from __future__ import annotations

from recon.primitives import evaluate_primitives


def test_perfect_primitive_match_scores_one() -> None:
    gt = [
        {
            "primitive_id": "cyl_0",
            "type": "cylinder",
            "center": [0.0, 0.0, 0.0],
            "axis": [0.0, 0.0, 1.0],
            "radius": 0.01,
            "height": 0.05,
            "dimensions": {"diameter_m": 0.02, "height_m": 0.05},
        },
        {
            "primitive_id": "plane_0",
            "type": "plane",
            "center": [0.0, 0.0, 0.025],
            "normal": [0.0, 0.0, 1.0],
            "offset": -0.025,
            "dimensions": {"size_u_m": 0.08, "size_v_m": 0.06},
        },
    ]
    predicted = [item.copy() for item in gt]
    metrics = evaluate_primitives(predicted, gt)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["dimension_mae_mm"] == 0.0


def test_missing_predictions_score_zero() -> None:
    gt = [
        {
            "primitive_id": "sphere_0",
            "type": "sphere",
            "center": [0.0, 0.0, 0.0],
            "radius": 0.02,
            "dimensions": {"diameter_m": 0.04},
        }
    ]
    metrics = evaluate_primitives([], gt)
    assert metrics["precision"] == 0.0
    assert metrics["recall"] == 0.0
    assert metrics["f1"] == 0.0
