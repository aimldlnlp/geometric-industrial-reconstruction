from __future__ import annotations

import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recon.styles import PALETTE, apply_paper_style
from recon.types import PrimitiveSpec


def setup_figure_style() -> None:
    apply_paper_style()


def set_equal_3d(ax, points: np.ndarray) -> None:
    if len(points) == 0:
        return
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * np.max(maxs - mins)
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect((1, 1, 1))


def style_3d_axis(ax, title: str | None = None) -> None:
    ax.set_facecolor("white")
    ax.xaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    if title:
        ax.set_title(title, pad=14)


def scatter_points(ax, points: np.ndarray, color: str, alpha: float = 0.9, size: float = 2.0, title: str | None = None) -> None:
    if len(points) > 0:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=size, c=color, alpha=alpha, linewidths=0)
        set_equal_3d(ax, points)
    style_3d_axis(ax, title=title)


def _orthonormal_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    direction = direction / max(np.linalg.norm(direction), 1e-9)
    candidate = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(candidate, direction)) > 0.9:
        candidate = np.array([0.0, 1.0, 0.0])
    tangent_1 = np.cross(direction, candidate)
    tangent_1 /= max(np.linalg.norm(tangent_1), 1e-9)
    tangent_2 = np.cross(direction, tangent_1)
    tangent_2 /= max(np.linalg.norm(tangent_2), 1e-9)
    return tangent_1, tangent_2


def draw_primitives(ax, primitives_payloads: list[dict], color: str = PALETTE["red"]) -> None:
    for payload in primitives_payloads:
        primitive = payload if isinstance(payload, PrimitiveSpec) else PrimitiveSpec.from_dict(payload)
        center = np.asarray(primitive.center, dtype=float)
        if primitive.type == "cylinder" and primitive.axis is not None and primitive.radius is not None:
            axis = np.asarray(primitive.axis, dtype=float)
            axis = axis / max(np.linalg.norm(axis), 1e-9)
            height = primitive.height or primitive.dimensions.get("height_m", 0.05)
            tangent_1, tangent_2 = _orthonormal_basis(axis)
            line = np.vstack([center - axis * height * 0.5, center + axis * height * 0.5])
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=2.0)
            angles = np.linspace(0.0, 2.0 * math.pi, 40)
            for sign in (-0.5, 0.5):
                circle_center = center + sign * axis * height
                circle = (
                    circle_center[None, :]
                    + primitive.radius * np.cos(angles)[:, None] * tangent_1[None, :]
                    + primitive.radius * np.sin(angles)[:, None] * tangent_2[None, :]
                )
                ax.plot(circle[:, 0], circle[:, 1], circle[:, 2], color=color, linewidth=1.0, alpha=0.7)
        elif primitive.type == "plane" and primitive.normal is not None:
            normal = np.asarray(primitive.normal, dtype=float)
            tangent_1, tangent_2 = _orthonormal_basis(normal)
            size_u = primitive.dimensions.get("size_u_m", 0.05) * 0.5
            size_v = primitive.dimensions.get("size_v_m", 0.05) * 0.5
            corners = np.asarray(
                [
                    center - tangent_1 * size_u - tangent_2 * size_v,
                    center + tangent_1 * size_u - tangent_2 * size_v,
                    center + tangent_1 * size_u + tangent_2 * size_v,
                    center - tangent_1 * size_u + tangent_2 * size_v,
                    center - tangent_1 * size_u - tangent_2 * size_v,
                ]
            )
            ax.plot(corners[:, 0], corners[:, 1], corners[:, 2], color=color, linewidth=1.4)
        elif primitive.type == "sphere" and primitive.radius is not None:
            radius = primitive.radius
            angles = np.linspace(0.0, 2.0 * math.pi, 60)
            ones = np.ones_like(angles)
            ax.plot(
                center[0] + radius * np.cos(angles),
                center[1] + radius * np.sin(angles),
                center[2] * ones,
                color=color,
                linewidth=1.2,
            )
            ax.plot(
                center[0] + radius * np.cos(angles),
                center[1] * ones,
                center[2] + radius * np.sin(angles),
                color=color,
                linewidth=1.2,
            )
            ax.plot(
                center[0] * ones,
                center[1] + radius * np.cos(angles),
                center[2] + radius * np.sin(angles),
                color=color,
                linewidth=1.2,
            )


def figure_to_array(fig) -> np.ndarray:
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    return buffer[:, :, :3].copy()


def save_frame_pairs(frames: list[np.ndarray], mp4_path: Path, gif_path: Path, fps: int) -> None:
    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(mp4_path, frames, fps=fps)
    imageio.mimsave(gif_path, frames, fps=max(6, min(fps, 15)))


def mode_metric(run_manifest: dict, mode: str, metric: str) -> list[float]:
    reports = run_manifest["modes"].get(mode, {}).get("reports", [])
    return [float(report[metric]) for report in reports if report.get(metric) is not None]
