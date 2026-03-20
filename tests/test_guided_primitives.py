from __future__ import annotations

import numpy as np

from recon.geometry import seed_everything
from recon.primitives import discover_primitives, evaluate_primitives


def test_reference_conditioned_cylinder_recovery_matches_template() -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    z_values = np.linspace(-0.03, 0.03, 32)
    angle_grid, z_grid = np.meshgrid(angles, z_values)
    radius = 0.012
    points = np.column_stack(
        [
            radius * np.cos(angle_grid).ravel(),
            radius * np.sin(angle_grid).ravel(),
            z_grid.ravel(),
        ]
    )

    config = {
        "primitive_threshold_m": 0.0025,
        "primitive_iterations": 48,
        "primitive_min_inliers": 120,
        "max_planes": 2,
        "max_cylinders": 2,
        "max_spheres": 1,
    }
    template = {
        "primitive_id": "cyl_stage_0",
        "type": "cylinder",
        "center": [0.0, 0.0, 0.0],
        "axis": [0.0, 0.0, 1.0],
        "radius": radius,
        "height": 0.06,
        "dimensions": {"diameter_m": radius * 2.0, "height_m": 0.06},
    }

    predicted = [primitive.to_dict() for primitive in discover_primitives(points, config, seed_everything(7), family="shaft", primitive_templates=[template])]
    metrics = evaluate_primitives(predicted, [template])

    assert len(predicted) == 1
    assert predicted[0]["primitive_id"] == "cyl_stage_0"
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
