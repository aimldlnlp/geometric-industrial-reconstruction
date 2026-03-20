from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from recon.config import load_config, project_root, resolve_from_root
from recon.io_utils import ensure_dir, read_json
from recon.visualization import (
    PALETTE,
    draw_primitives,
    figure_to_array,
    save_frame_pairs,
    scatter_points,
    setup_figure_style,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MP4 and GIF scene videos for a completed run.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory under artifacts/runs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional video output directory override.")
    return parser.parse_args()


def _load_stage(stage_path: str) -> dict[str, np.ndarray]:
    payload = np.load(resolve_from_root(stage_path))
    return {key: payload[key] for key in payload.files}


def _orbit_frames(points: np.ndarray, reference: np.ndarray | None, title: str, frames_per_scene: int, primitives: list[dict] | None = None, residual_mm: np.ndarray | None = None) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    primitives = primitives or []
    for frame_idx in range(frames_per_scene):
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        if reference is not None and len(reference) > 0:
            ax.scatter(reference[:, 0], reference[:, 1], reference[:, 2], s=0.8, c=PALETTE["gray"], alpha=0.12, linewidths=0)
        if residual_mm is None:
            scatter_points(ax, points, PALETTE["blue"], size=1.4, title=title)
        else:
            ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, c=residual_mm, cmap="viridis", alpha=0.95, linewidths=0)
            ax.set_title(title, pad=14)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        if primitives:
            draw_primitives(ax, primitives, color=PALETTE["red"])
        ax.view_init(elev=23, azim=(frame_idx / frames_per_scene) * 360.0)
        frames.append(figure_to_array(fig))
        plt.close(fig)
    return frames


def _alignment_refinement_frames(stage: dict[str, np.ndarray], frames_per_scene: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    thirds = frames_per_scene // 3
    for frame_idx in range(frames_per_scene):
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        reference = stage["reference_points"]
        ax.scatter(reference[:, 0], reference[:, 1], reference[:, 2], s=0.8, c=PALETTE["gray"], alpha=0.12, linewidths=0)
        if frame_idx < thirds:
            title = "Observed scan"
            points = stage["raw_points"]
            color = PALETTE["blue"]
        elif frame_idx < 2 * thirds:
            title = "After global registration"
            points = stage["global_points"]
            color = PALETTE["gold"]
        else:
            title = "After ICP refinement"
            points = stage["aligned_points"]
            color = PALETTE["green"]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1.5, c=color, alpha=0.9, linewidths=0)
        ax.set_title(title, pad=14)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.view_init(elev=22, azim=(frame_idx / frames_per_scene) * 240.0 + 40.0)
        frames.append(figure_to_array(fig))
        plt.close(fig)
    return frames


def _primitive_discovery_frames(stage: dict[str, np.ndarray], predicted_primitives: list[dict], frames_per_scene: int) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    primitive_count = max(1, len(predicted_primitives))
    for frame_idx in range(frames_per_scene):
        fig = plt.figure(figsize=(12.8, 7.2))
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        scatter_points(ax, stage["aligned_points"], PALETTE["blue"], size=1.4, title="Primitive discovery")
        visible = min(primitive_count, max(0, int(((frame_idx + 1) / frames_per_scene) * primitive_count)))
        draw_primitives(ax, predicted_primitives[:visible], color=PALETTE["red"])
        ax.view_init(elev=24, azim=(frame_idx / frames_per_scene) * 180.0 + 30.0)
        frames.append(figure_to_array(fig))
        plt.close(fig)
    return frames


def _select_reports(run_manifest: dict) -> tuple[dict, dict, dict, dict]:
    full_reports = run_manifest["modes"]["full"]["reports"]
    by_chamfer = sorted(full_reports, key=lambda item: float(item["chamfer_mm"]))
    by_primitive = sorted(full_reports, key=lambda item: float(item["primitive_f1"]))
    median_report = by_chamfer[len(by_chamfer) // 2]
    best_report = by_chamfer[0]
    worst_report = by_chamfer[-1]
    primitive_report = by_primitive[-1]
    return median_report, best_report, worst_report, primitive_report


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_figure_style()
    run_dir = resolve_from_root(args.run_dir)
    run_manifest = read_json(run_dir / "run_manifest.json")
    output_dir = ensure_dir(resolve_from_root(args.output_dir or config["videos"]["root"]) / run_dir.name)
    frames_per_scene = int(config["videos"]["frames_per_scene"])
    fps = int(config["videos"]["fps"])

    median_report, best_report, worst_report, primitive_report = _select_reports(run_manifest)
    median_stage = _load_stage(median_report["stage_path"])
    best_stage = _load_stage(best_report["stage_path"])
    worst_stage = _load_stage(worst_report["stage_path"])
    primitive_stage = _load_stage(primitive_report["stage_path"])

    scenes = [
        (
            "video_01_rotating_partial_scan",
            _orbit_frames(median_stage["raw_points"], None, "Rotating partial scan", frames_per_scene),
        ),
        (
            "video_02_alignment_refinement",
            _alignment_refinement_frames(median_stage, frames_per_scene),
        ),
        (
            "video_03_primitive_discovery",
            _primitive_discovery_frames(primitive_stage, primitive_report["predicted_primitives"], frames_per_scene),
        ),
        (
            "video_04_error_heatmap_orbit",
            _orbit_frames(
                worst_stage["aligned_points"],
                worst_stage["reference_points"],
                "Residual heatmap orbit",
                frames_per_scene,
                residual_mm=worst_stage["residual_mm"],
            ),
        ),
        (
            "video_05_final_reconstruction_overlay",
            _orbit_frames(
                best_stage["aligned_points"],
                best_stage["reference_points"],
                "Final reconstruction overlay",
                frames_per_scene,
                primitives=best_report["predicted_primitives"],
            ),
        ),
    ]

    for stem, frames in scenes:
        save_frame_pairs(
            frames,
            output_dir / f"{stem}.mp4",
            output_dir / f"{stem}.gif",
            fps=fps,
        )

    print(f"Exported videos to {output_dir.relative_to(project_root()).as_posix()}.")


if __name__ == "__main__":
    main()
