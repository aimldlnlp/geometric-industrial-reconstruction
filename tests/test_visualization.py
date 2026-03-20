from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recon.visualization import draw_primitives


def test_draw_primitives_renders_sphere_without_dimension_errors() -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    draw_primitives(
        ax,
        [
            {
                "primitive_id": "sphere_corner",
                "type": "sphere",
                "center": [0.0, 0.0, 0.0],
                "axis": None,
                "normal": None,
                "radius": 0.02,
                "height": None,
                "offset": None,
                "dimensions": {"diameter_m": 0.04},
                "role": "blend",
                "confidence": 0.9,
                "support_size": 120,
            }
        ],
    )

    assert len(ax.lines) == 3
    plt.close(fig)
