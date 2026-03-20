from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import linear_sum_assignment

from recon.config import load_config, project_root, resolve_from_root
from recon.io_utils import ensure_dir, read_json
from recon.types import PrimitiveSpec
from recon.visualization import PALETTE, draw_primitives, scatter_points, set_equal_3d, setup_figure_style


FAMILY_ORDER = {
    "flange": 0,
    "shaft": 1,
    "bracket": 2,
    "pipe_elbow": 3,
    "plate_with_holes": 4,
}

FAMILY_MARKERS = {
    "flange": "o",
    "shaft": "s",
    "bracket": "^",
    "pipe_elbow": "D",
    "plate_with_holes": "P",
}

HERO_VISUAL_PRIORITY = {
    "pipe_elbow": 0,
    "bracket": 1,
    "plate_with_holes": 2,
    "shaft": 3,
    "flange": 4,
}

MODE_ORDER = ["full", "no_global", "no_refine", "no_denoise"]
MODE_LABELS = {
    "full": "Full",
    "no_global": "No global",
    "no_refine": "No refine",
    "no_denoise": "No denoise",
}

PRIMITIVE_TYPE_ORDER = ["plane", "cylinder", "sphere"]
PRIMITIVE_TYPE_COLORS = {
    "plane": PALETTE["green"],
    "cylinder": PALETTE["blue"],
    "sphere": PALETTE["gold"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export paper-style PNG figures for a completed run.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory under artifacts/runs.")
    parser.add_argument("--output-dir", type=str, default=None, help="Optional figure output directory override.")
    return parser.parse_args()


def _family_label(family: str) -> str:
    return family.replace("_", " ").title()


def _load_points(path: Path) -> np.ndarray:
    import open3d as o3d

    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points, dtype=float)


def _load_stage(stage_path: str) -> dict[str, np.ndarray]:
    payload = np.load(resolve_from_root(stage_path))
    return {key: payload[key] for key in payload.files}


def _select_representative(reports: list[dict], key: str, strategy: str = "median") -> dict:
    ordered = sorted(reports, key=lambda item: float(item[key]))
    if not ordered:
        raise ValueError("No reports available.")
    if strategy == "best":
        return ordered[0]
    if strategy == "worst":
        return ordered[-1]
    return ordered[len(ordered) // 2]


def _metric_value(payload: dict, key: str, default: float = 0.0) -> float:
    value = payload.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _median_metric(items: list[dict], key: str) -> float:
    values = [float(item[key]) for item in items if item.get(key) is not None]
    if not values:
        return float("nan")
    return float(np.median(values))


def _primitive_specs(payloads: list[dict]) -> list[PrimitiveSpec]:
    return [item if isinstance(item, PrimitiveSpec) else PrimitiveSpec.from_dict(item) for item in payloads]


def _dimension_error_mm(predicted: PrimitiveSpec, ground_truth: PrimitiveSpec) -> float:
    shared = sorted(set(predicted.dimensions) & set(ground_truth.dimensions))
    if shared:
        return float(np.mean([abs(predicted.dimensions[key] - ground_truth.dimensions[key]) for key in shared]) * 1000.0)
    if predicted.radius is not None and ground_truth.radius is not None:
        return float(abs(predicted.radius - ground_truth.radius) * 1000.0)
    return 0.0


def _match_primitive_specs(predicted_payloads: list[dict], gt_payloads: list[dict]) -> list[tuple[PrimitiveSpec, PrimitiveSpec, float]]:
    predicted = _primitive_specs(predicted_payloads)
    ground_truth = _primitive_specs(gt_payloads)
    if not predicted or not ground_truth:
        return []
    cost_matrix = np.full((len(predicted), len(ground_truth)), fill_value=1e6, dtype=float)
    for row, pred in enumerate(predicted):
        for col, gt in enumerate(ground_truth):
            if pred.type != gt.type:
                continue
            cost_matrix[row, col] = _dimension_error_mm(pred, gt)
    rows, cols = linear_sum_assignment(cost_matrix)
    matches: list[tuple[PrimitiveSpec, PrimitiveSpec, float]] = []
    for row, col in zip(rows, cols):
        error = float(cost_matrix[row, col])
        if error < 1e5:
            matches.append((predicted[row], ground_truth[col], error))
    return matches


def _matched_primitive_rows(predicted_payloads: list[dict], gt_payloads: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for predicted, ground_truth, error in _match_primitive_specs(predicted_payloads, gt_payloads):
        rows.append(
            {
                "primitive_id": predicted.primitive_id,
                "primitive_type": predicted.type,
                "ground_truth_id": ground_truth.primitive_id,
                "confidence": None if predicted.confidence is None else float(predicted.confidence),
                "dimension_error_mm": float(error),
            }
        )
    return rows


def _hero_quality_score(report: dict) -> float:
    return (
        (2.0 if bool(report.get("registration_success", False)) else 0.0)
        + 1.4 * _metric_value(report, "primitive_f1")
        + 1.0 * _metric_value(report, "coverage")
        - 0.07 * _metric_value(report, "chamfer_mm", default=25.0)
        - 0.02 * _metric_value(report, "rot_err_deg", default=25.0)
        - 0.01 * _metric_value(report, "trans_err_mm", default=25.0)
    )


def _select_hero_report(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("No reports available.")
    successful = [report for report in reports if bool(report.get("registration_success", False))]
    pool = successful or reports
    strong = [
        report
        for report in pool
        if _metric_value(report, "primitive_f1") >= 0.95 and _metric_value(report, "coverage") >= 0.60
    ]
    pool = strong or pool
    best_score = max(_hero_quality_score(report) for report in pool)
    shortlist = [report for report in pool if best_score - _hero_quality_score(report) <= 0.30]
    return min(
        shortlist,
        key=lambda report: (
            HERO_VISUAL_PRIORITY.get(report.get("family", ""), 99),
            -_hero_quality_score(report),
            str(report.get("scan_id", "")),
        ),
    )


def _select_family_representatives(records: list[dict], split: str | None, max_items: int = 5) -> list[dict]:
    filtered = [record for record in records if split is None or record.get("split") == split]
    if not filtered:
        filtered = list(records)
    representatives: dict[str, dict] = {}
    ordered = sorted(
        filtered,
        key=lambda record: (
            FAMILY_ORDER.get(record.get("family", ""), 99),
            str(record.get("scan_id", "")),
        ),
    )
    for record in ordered:
        representatives.setdefault(str(record.get("family", "")), record)
    return list(representatives.values())[:max_items]


def _sample_indices(num_points: int, max_points: int) -> np.ndarray:
    if num_points <= 0:
        return np.zeros((0,), dtype=int)
    if num_points <= max_points:
        return np.arange(num_points, dtype=int)
    return np.linspace(0, num_points - 1, num=max_points, dtype=int)


def _sample_points(points: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    indices = _sample_indices(len(points), max_points)
    return points[indices], indices


def _style_hero_3d_axis(ax, title: str, elev: float = 22.0, azim: float = 35.0, title_size: float = 12.0) -> None:
    ax.set_facecolor("white")
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 0.0))
        axis.pane.set_edgecolor("#d9e2ec")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title, fontsize=title_size, pad=8)


def _scatter_panel(ax, points: np.ndarray, *, color: str, alpha: float, size: float, title: str, elev: float = 22.0, azim: float = 35.0, title_size: float = 12.0) -> None:
    sampled_points, _ = _sample_points(points, max_points=4500)
    if len(sampled_points) > 0:
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=size, c=color, alpha=alpha, linewidths=0)
        set_equal_3d(ax, sampled_points)
    _style_hero_3d_axis(ax, title=title, elev=elev, azim=azim, title_size=title_size)


def _overlay_panel(ax, aligned_points: np.ndarray, reference_points: np.ndarray, *, title: str, elev: float = 22.0, azim: float = 35.0, title_size: float = 12.0) -> None:
    sampled_ref, _ = _sample_points(reference_points, max_points=5000)
    sampled_aligned, _ = _sample_points(aligned_points, max_points=4500)
    if len(sampled_ref) > 0:
        ax.scatter(sampled_ref[:, 0], sampled_ref[:, 1], sampled_ref[:, 2], s=1.0, c=PALETTE["gray"], alpha=0.14, linewidths=0)
    if len(sampled_aligned) > 0:
        ax.scatter(sampled_aligned[:, 0], sampled_aligned[:, 1], sampled_aligned[:, 2], s=1.4, c=PALETTE["blue"], alpha=0.82, linewidths=0)
    combined = np.vstack([arr for arr in (sampled_ref, sampled_aligned) if len(arr) > 0]) if len(sampled_ref) or len(sampled_aligned) else np.zeros((0, 3))
    if len(combined) > 0:
        set_equal_3d(ax, combined)
    _style_hero_3d_axis(ax, title=title, elev=elev, azim=azim, title_size=title_size)


def _inspection_panel(ax, aligned_points: np.ndarray, primitives: list[dict], *, title: str, elev: float = 22.0, azim: float = 35.0, title_size: float = 12.0) -> None:
    sampled_points, _ = _sample_points(aligned_points, max_points=4500)
    if len(sampled_points) > 0:
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], sampled_points[:, 2], s=1.5, c=PALETTE["blue"], alpha=0.78, linewidths=0)
        set_equal_3d(ax, sampled_points)
    draw_primitives(ax, primitives, color=PALETTE["gold"])
    _style_hero_3d_axis(ax, title=title, elev=elev, azim=azim, title_size=title_size)


def _draw_metric_card(ax, summary: dict, split: str, num_scans: int) -> None:
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(0.0, 0.98, "Benchmark snapshot", fontsize=11, color=PALETTE["gray"], va="top")
    ax.text(0.0, 0.82, f"{100.0 * _metric_value(summary, 'registration_success_rate'):.1f}%", fontsize=28, fontweight="bold", va="top")
    ax.text(0.0, 0.68, "Registration success", fontsize=10, color=PALETTE["gray"], va="top")
    ax.axhline(0.60, color=PALETTE["light"], linewidth=1.2)
    rows = [
        ("Median rotation error", f"{_metric_value(summary, 'median_rot_err_deg'):.2f} deg"),
        ("Median translation error", f"{_metric_value(summary, 'median_trans_err_mm'):.2f} mm"),
        ("Median primitive F1", f"{_metric_value(summary, 'median_primitive_f1'):.2f}"),
    ]
    for index, (label, value) in enumerate(rows):
        y = 0.50 - index * 0.16
        ax.text(0.0, y, label, fontsize=9.5, color=PALETTE["gray"], va="top")
        ax.text(0.0, y - 0.075, value, fontsize=16, va="top")
    ax.text(0.0, 0.05, f"Held-out split: {split} ({num_scans} scans)", fontsize=9.5, color=PALETTE["gray"], va="bottom")


def _draw_workflow_note(ax) -> None:
    ax.axis("off")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.text(
        0.0,
        0.78,
        "Part family library  ->  partial scan  ->  registration  ->  geometric inspection",
        fontsize=9.3,
        va="top",
    )
    ax.text(
        0.0,
        0.28,
        "Python-only, CPU-only evaluation with geometric tolerance analysis on held-out industrial parts.",
        fontsize=9.1,
        color=PALETTE["gray"],
        va="top",
    )


def _match_dimension_errors(predicted_payloads: list[dict], gt_payloads: list[dict]) -> list[tuple[str, float]]:
    return [
        (f"{row['primitive_type']}:{row['primitive_id']}", float(row["dimension_error_mm"]))
        for row in _matched_primitive_rows(predicted_payloads, gt_payloads)
    ]


def _merge_reports_with_records(reports: list[dict], records: list[dict], split: str | None) -> list[dict]:
    filtered_records = [record for record in records if split is None or record.get("split") == split]
    record_lookup = {record["scan_id"]: record for record in filtered_records}
    merged: list[dict] = []
    for report in reports:
        record = record_lookup.get(report["scan_id"])
        if record is None:
            continue
        merged.append(
            {
                "scan_id": report["scan_id"],
                "family": report["family"],
                "registration_success": bool(report.get("registration_success", False)),
                "coverage": _metric_value(report, "coverage"),
                "chamfer_mm": _metric_value(report, "chamfer_mm"),
                "report": report,
                "record": record,
                "noise_profile": dict(record.get("noise_profile", {})),
            }
        )
    return merged


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values.copy()
    lower = float(np.min(values))
    upper = float(np.max(values))
    if upper - lower < 1e-12:
        return np.zeros_like(values, dtype=float)
    return (values - lower) / (upper - lower)


def _compute_difficulty_indices(records: list[dict]) -> dict[str, float]:
    if not records:
        return {}
    noise = np.asarray([float(record.get("noise_profile", {}).get("noise_std_mm", 0.0)) for record in records], dtype=float)
    outlier = np.asarray([float(record.get("noise_profile", {}).get("outlier_ratio", 0.0)) for record in records], dtype=float)
    occlusion = np.asarray([float(record.get("noise_profile", {}).get("occlusion_dropout", 0.0)) for record in records], dtype=float)
    view_penalty = np.asarray(
        [1.0 / max(float(record.get("noise_profile", {}).get("views_used", 1.0)), 1.0) for record in records],
        dtype=float,
    )
    difficulty = (
        0.32 * _normalize(noise)
        + 0.24 * _normalize(outlier)
        + 0.24 * _normalize(occlusion)
        + 0.20 * _normalize(view_penalty)
    )
    return {record["scan_id"]: float(score) for record, score in zip(records, difficulty)}


def _aggregate_family_mode_metrics(modes: dict[str, dict]) -> dict[str, dict[str, dict[str, float]]]:
    aggregated: dict[str, dict[str, dict[str, float]]] = {}
    for mode_name, mode_payload in modes.items():
        reports = mode_payload.get("reports", [])
        grouped: dict[str, list[dict]] = {}
        for report in reports:
            grouped.setdefault(report["family"], []).append(report)
        aggregated[mode_name] = {}
        for family, family_reports in grouped.items():
            aggregated[mode_name][family] = {
                "registration_success_rate": float(np.mean([1.0 if report.get("registration_success", False) else 0.0 for report in family_reports])),
                "median_rot_err_deg": _median_metric(family_reports, "rot_err_deg"),
                "median_trans_err_mm": _median_metric(family_reports, "trans_err_mm"),
                "median_chamfer_mm": _median_metric(family_reports, "chamfer_mm"),
                "median_runtime_sec": _median_metric(family_reports, "runtime_sec"),
            }
    return aggregated


def _normalize_matrix_by_column(matrix: np.ndarray, invert_columns: set[int] | None = None) -> np.ndarray:
    invert_columns = invert_columns or set()
    display = np.full(matrix.shape, np.nan, dtype=float)
    for column in range(matrix.shape[1]):
        values = matrix[:, column]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        finite_values = values[finite]
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
        if upper - lower < 1e-12:
            normalized = np.full(np.count_nonzero(finite), 0.5, dtype=float)
        else:
            normalized = (finite_values - lower) / (upper - lower)
        if column in invert_columns:
            normalized = 1.0 - normalized
        display[finite, column] = 0.18 + 0.72 * normalized
    return display


def _normalize_matrix_by_row(matrix: np.ndarray, invert_rows: set[int] | None = None) -> np.ndarray:
    invert_rows = invert_rows or set()
    display = np.full(matrix.shape, np.nan, dtype=float)
    for row in range(matrix.shape[0]):
        values = matrix[row]
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        finite_values = values[finite]
        lower = float(np.min(finite_values))
        upper = float(np.max(finite_values))
        if upper - lower < 1e-12:
            normalized = np.full(np.count_nonzero(finite), 0.5, dtype=float)
        else:
            normalized = (finite_values - lower) / (upper - lower)
        if row in invert_rows:
            normalized = 1.0 - normalized
        display[row, finite] = 0.18 + 0.72 * normalized
    return display


def _annotate_heatmap(ax, values: np.ndarray, formatter, missing_text: str = "--") -> None:
    rows, cols = values.shape
    for row in range(rows):
        for col in range(cols):
            value = values[row, col]
            label = formatter(float(value)) if np.isfinite(value) else missing_text
            ax.text(col, row, label, ha="center", va="center", fontsize=9)


def _style_heatmap_axis(ax, row_labels: list[str], col_labels: list[str]) -> None:
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xticks(np.arange(-0.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _empty_figure_panel(ax, title: str, message: str) -> None:
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=12)
    ax.text(0.5, 0.42, message, ha="center", va="center", fontsize=10, color=PALETTE["gray"])


def _render_pipeline_overview(output_path: Path, run_manifest: dict, dataset_manifest: dict, config: dict) -> None:
    del dataset_manifest
    hero_report = _select_hero_report(run_manifest["modes"]["full"]["reports"])
    hero_stage = _load_stage(hero_report["stage_path"])

    fig = plt.figure(figsize=(13, 4.8))
    grid = fig.add_gridspec(1, 3, left=0.035, right=0.99, bottom=0.08, top=0.82, wspace=0.06)

    fig.suptitle("Geometric Reconstruction from Partial Industrial Scans", y=0.92, fontsize=18)

    ax_observed = fig.add_subplot(grid[0, 0], projection="3d")
    ax_overlay = fig.add_subplot(grid[0, 1], projection="3d")
    ax_inspect = fig.add_subplot(grid[0, 2], projection="3d")

    _scatter_panel(ax_observed, hero_stage["raw_points"], color=PALETTE["blue"], alpha=0.72, size=1.2, title="Observed partial scan", title_size=11.5)
    _overlay_panel(
        ax_overlay,
        hero_stage["aligned_points"],
        hero_stage["reference_points"],
        title="Aligned reconstruction",
        title_size=11.5,
    )
    _inspection_panel(
        ax_inspect,
        hero_stage["aligned_points"],
        hero_report["predicted_primitives"],
        title="Geometric inspection",
        title_size=11.5,
    )

    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _render_dataset_diversity(output_path: Path, dataset_manifest: dict, config: dict) -> None:
    records = dataset_manifest["records"]
    family_choice = {}
    for record in records:
        family_choice.setdefault(record["family"], record)
    fig = plt.figure(figsize=(13, 8))
    for index, (family, record) in enumerate(family_choice.items(), start=1):
        ax = fig.add_subplot(2, 3, index, projection="3d")
        points = _load_points(resolve_from_root(record["cloud_path"]))
        scatter_points(ax, points, PALETTE["blue"], size=1.2, title=family.replace("_", " ").title())
        ax.view_init(elev=25, azim=42)
    fig.suptitle("Family Diversity in the Controlled Industrial Scan Benchmark", y=0.98, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _render_registration(output_path: Path, run_manifest: dict, config: dict) -> None:
    report = _select_representative(run_manifest["modes"]["full"]["reports"], "chamfer_mm", strategy="median")
    stage = _load_stage(report["stage_path"])
    fig = plt.figure(figsize=(14, 4.5))
    views = [
        ("Observed scan", stage["raw_points"], stage["reference_points"], 1),
        ("After global alignment", stage["global_points"], stage["reference_points"], 2),
        ("After ICP refinement", stage["aligned_points"], stage["reference_points"], 3),
    ]
    for title, source, target, index in views:
        ax = fig.add_subplot(1, 3, index, projection="3d")
        scatter_points(ax, target, PALETTE["gray"], alpha=0.16, size=1.3, title=title)
        ax.scatter(source[:, 0], source[:, 1], source[:, 2], s=1.5, c=PALETTE["blue"], alpha=0.85, linewidths=0)
        ax.view_init(elev=22, azim=35)
    fig.suptitle("Registration Progression: Raw, Global, and Refined Alignment", y=0.98, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _render_primitive_tolerance(output_path: Path, run_manifest: dict, dataset_manifest: dict, config: dict) -> None:
    report = max(run_manifest["modes"]["full"]["reports"], key=lambda item: float(item["primitive_f1"]))
    stage = _load_stage(report["stage_path"])
    record_lookup = {record["scan_id"]: record for record in dataset_manifest["records"]}
    gt_record = record_lookup[report["scan_id"]]
    matches = _match_dimension_errors(report["predicted_primitives"], gt_record["primitive_gt"])
    fig = plt.figure(figsize=(12.5, 5.2))
    ax_left = fig.add_subplot(1, 2, 1, projection="3d")
    scatter_points(ax_left, stage["aligned_points"], PALETTE["blue"], size=1.5, title="Recovered primitives on aligned scan")
    draw_primitives(ax_left, report["predicted_primitives"], color=PALETTE["red"])
    ax_left.view_init(elev=24, azim=35)
    ax_right = fig.add_subplot(1, 2, 2)
    if matches:
        labels, values = zip(*matches, strict=False)
        ax_right.bar(labels, values, color=PALETTE["gold"], edgecolor=PALETTE["ink"], linewidth=0.8)
        ax_right.axhline(3.0, color=PALETTE["red"], linewidth=1.4, linestyle="--", label="3 mm tolerance")
        ax_right.set_ylabel("Dimension error (mm)")
        ax_right.set_title("Tolerance inspection")
        ax_right.legend(loc="upper right")
        ax_right.tick_params(axis="x", rotation=25)
    else:
        ax_right.text(0.5, 0.5, "No matched primitives available.", ha="center", va="center")
        ax_right.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _render_error_ablation(output_path: Path, run_manifest: dict, config: dict) -> None:
    full_reports = run_manifest["modes"]["full"]["reports"]
    failure_reports = sorted(full_reports, key=lambda item: float(item["chamfer_mm"]), reverse=True)[:3]
    mode_names = list(run_manifest["modes"].keys())
    fig = plt.figure(figsize=(14, 8))
    top_metrics = [("rot_err_deg", "Rotation error (deg)"), ("trans_err_mm", "Translation error (mm)"), ("primitive_f1", "Primitive F1")]
    for index, (metric, label) in enumerate(top_metrics, start=1):
        ax = fig.add_subplot(2, 3, index)
        values = [[float(report[metric]) for report in run_manifest["modes"][mode]["reports"]] for mode in mode_names]
        ax.boxplot(values, patch_artist=True, boxprops={"facecolor": PALETTE["light"], "edgecolor": PALETTE["gray"]})
        ax.set_xticklabels([mode.replace("_", "\n") for mode in mode_names], rotation=0)
        ax.set_ylabel(label)
        ax.set_title(label)
    for idx, report in enumerate(failure_reports, start=4):
        ax = fig.add_subplot(2, 3, idx, projection="3d")
        stage = _load_stage(report["stage_path"])
        scatter_points(ax, stage["reference_points"], PALETTE["gray"], alpha=0.12, size=1.0, title=f"Failure atlas: {report['scan_id']}")
        ax.scatter(stage["aligned_points"][:, 0], stage["aligned_points"][:, 1], stage["aligned_points"][:, 2], s=1.3, c=PALETTE["red"], alpha=0.8, linewidths=0)
        ax.view_init(elev=23, azim=30)
    fig.suptitle("Error Summary, Ablation Study, and Failure Atlas", y=0.98, fontsize=16)
    fig.tight_layout()
    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _render_family_benchmark_matrix(output_path: Path, run_manifest: dict, config: dict) -> None:
    aggregated = _aggregate_family_mode_metrics(run_manifest["modes"])
    families = sorted(
        {family for mode_payload in aggregated.values() for family in mode_payload.keys()},
        key=lambda family: FAMILY_ORDER.get(family, 99),
    )
    modes = [mode for mode in MODE_ORDER if mode in aggregated]

    fig = plt.figure(figsize=(13, 5.2))
    grid = fig.add_gridspec(1, 2, left=0.06, right=0.98, bottom=0.12, top=0.88, wspace=0.18, width_ratios=[1.0, 1.15])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    fig.suptitle("Family-Level Benchmark Summary", y=0.96, fontsize=16)

    if not families or not modes:
        _empty_figure_panel(ax_left, "No family benchmark data", "Run reports are missing.")
        _empty_figure_panel(ax_right, "No family benchmark data", "Run reports are missing.")
        fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
        plt.close(fig)
        return

    success_matrix = np.full((len(families), len(modes)), np.nan, dtype=float)
    for row, family in enumerate(families):
        for col, mode in enumerate(modes):
            success_matrix[row, col] = aggregated.get(mode, {}).get(family, {}).get("registration_success_rate", np.nan)
    success_cmap = plt.get_cmap("Blues").copy()
    success_cmap.set_bad("white")
    ax_left.imshow(np.ma.masked_invalid(success_matrix), cmap=success_cmap, vmin=0.0, vmax=1.0, aspect="auto")
    _style_heatmap_axis(ax_left, [_family_label(family) for family in families], [MODE_LABELS.get(mode, mode) for mode in modes])
    ax_left.set_title("Registration success by family and mode")
    _annotate_heatmap(ax_left, success_matrix, formatter=lambda value: f"{100.0 * value:.0f}%")

    metric_specs = [
        ("median_rot_err_deg", "Rot\n(deg)"),
        ("median_trans_err_mm", "Trans\n(mm)"),
        ("median_chamfer_mm", "Chamfer\n(mm)"),
        ("median_runtime_sec", "Runtime\n(s)"),
    ]
    summary_matrix = np.full((len(families), len(metric_specs)), np.nan, dtype=float)
    full_summary = aggregated.get("full", {})
    for row, family in enumerate(families):
        family_summary = full_summary.get(family, {})
        for col, (metric, _) in enumerate(metric_specs):
            summary_matrix[row, col] = family_summary.get(metric, np.nan)
    summary_display = _normalize_matrix_by_column(summary_matrix, invert_columns={0, 1, 2, 3})
    summary_cmap = plt.get_cmap("Blues").copy()
    summary_cmap.set_bad("white")
    ax_right.imshow(np.ma.masked_invalid(summary_display), cmap=summary_cmap, vmin=0.0, vmax=1.0, aspect="auto")
    _style_heatmap_axis(ax_right, [_family_label(family) for family in families], [label for _, label in metric_specs])
    ax_right.set_title("Full-mode family summary")
    _annotate_heatmap(ax_right, summary_matrix, formatter=lambda value: f"{value:.2f}")

    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _render_capture_difficulty_vs_robustness(output_path: Path, run_manifest: dict, dataset_manifest: dict, config: dict) -> None:
    merged = _merge_reports_with_records(run_manifest["modes"]["full"]["reports"], dataset_manifest["records"], split=run_manifest.get("split"))
    difficulty_lookup = _compute_difficulty_indices([item["record"] for item in merged])

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharex=True)
    fig.suptitle("Capture Difficulty and Registration Robustness", y=0.97, fontsize=16)

    if not merged:
        _empty_figure_panel(axes[0], "No capture difficulty data", "Held-out reports could not be matched to dataset records.")
        _empty_figure_panel(axes[1], "No capture difficulty data", "Held-out reports could not be matched to dataset records.")
        fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
        plt.close(fig)
        return

    metric_specs = [
        ("coverage", "Coverage", "Coverage vs capture difficulty"),
        ("chamfer_mm", "Chamfer (mm)", "Chamfer vs capture difficulty"),
    ]
    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        for item in merged:
            family = item["family"]
            marker = FAMILY_MARKERS.get(family, "o")
            color = PALETTE["green"] if item["registration_success"] else PALETTE["red"]
            ax.scatter(
                difficulty_lookup.get(item["scan_id"], 0.0),
                float(item[metric]),
                marker=marker,
                s=64,
                c=color,
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
            )
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel("Capture difficulty index")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.5)

    family_handles = [
        Line2D([0], [0], marker=FAMILY_MARKERS[family], linestyle="None", markerfacecolor=PALETTE["blue"], markeredgecolor="white", markeredgewidth=0.5, markersize=8, label=_family_label(family))
        for family in sorted({item["family"] for item in merged}, key=lambda family: FAMILY_ORDER.get(family, 99))
    ]
    status_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=PALETTE["green"], markeredgecolor="white", markeredgewidth=0.5, markersize=8, label="Success"),
        Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=PALETTE["red"], markeredgecolor="white", markeredgewidth=0.5, markersize=8, label="Failure"),
    ]
    family_legend = axes[1].legend(handles=family_handles, loc="upper left", title="Family")
    axes[1].add_artist(family_legend)
    axes[1].legend(handles=status_handles, loc="lower right", title="Registration")

    fig.tight_layout()
    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def _render_primitive_confidence_tolerance(output_path: Path, run_manifest: dict, dataset_manifest: dict, config: dict) -> None:
    record_lookup = {record["scan_id"]: record for record in dataset_manifest["records"]}
    rows: list[dict] = []
    for report in run_manifest["modes"]["full"]["reports"]:
        ground_truth_record = record_lookup.get(report["scan_id"])
        if ground_truth_record is None:
            continue
        matched_rows = _matched_primitive_rows(report["predicted_primitives"], ground_truth_record["primitive_gt"])
        for row in matched_rows:
            row["scan_id"] = report["scan_id"]
            row["family"] = report["family"]
        rows.extend(matched_rows)

    fig = plt.figure(figsize=(13, 4.8))
    grid = fig.add_gridspec(1, 2, left=0.07, right=0.98, bottom=0.13, top=0.88, wspace=0.20, width_ratios=[1.25, 0.9])
    ax_left = fig.add_subplot(grid[0, 0])
    ax_right = fig.add_subplot(grid[0, 1])
    fig.suptitle("Primitive Confidence and Dimensional Tolerance", y=0.97, fontsize=16)

    rows_with_confidence = [row for row in rows if row.get("confidence") is not None]
    if not rows_with_confidence:
        _empty_figure_panel(ax_left, "No confidence-bearing primitives", "The current run did not produce confidence values.")
    else:
        for primitive_type in PRIMITIVE_TYPE_ORDER:
            subset = [row for row in rows_with_confidence if row["primitive_type"] == primitive_type]
            if not subset:
                continue
            ax_left.scatter(
                [float(row["confidence"]) for row in subset],
                [float(row["dimension_error_mm"]) for row in subset],
                s=50,
                c=PRIMITIVE_TYPE_COLORS[primitive_type],
                edgecolors="white",
                linewidths=0.5,
                alpha=0.9,
                label=_family_label(primitive_type),
            )
        ax_left.axhline(3.0, color=PALETTE["red"], linewidth=1.2, linestyle="--", label="3 mm tolerance")
        ax_left.set_xlim(-0.03, 1.03)
        ax_left.set_xlabel("Primitive confidence")
        ax_left.set_ylabel("Dimension error (mm)")
        ax_left.set_title("Confidence vs dimensional error")
        ax_left.legend(loc="upper right")

    summary_matrix = np.full((2, len(PRIMITIVE_TYPE_ORDER)), np.nan, dtype=float)
    for col, primitive_type in enumerate(PRIMITIVE_TYPE_ORDER):
        subset = [row for row in rows_with_confidence if row["primitive_type"] == primitive_type]
        if not subset:
            continue
        summary_matrix[0, col] = float(np.median([float(row["confidence"]) for row in subset]))
        summary_matrix[1, col] = float(np.median([float(row["dimension_error_mm"]) for row in subset]))
    summary_display = _normalize_matrix_by_row(summary_matrix, invert_rows={1})
    summary_cmap = plt.get_cmap("Blues").copy()
    summary_cmap.set_bad("white")
    ax_right.imshow(np.ma.masked_invalid(summary_display), cmap=summary_cmap, vmin=0.0, vmax=1.0, aspect="auto")
    _style_heatmap_axis(ax_right, ["Median confidence", "Median dim. error (mm)"], [_family_label(primitive_type) for primitive_type in PRIMITIVE_TYPE_ORDER])
    ax_right.set_title("Per-type summary")
    for row in range(summary_matrix.shape[0]):
        for col in range(summary_matrix.shape[1]):
            value = summary_matrix[row, col]
            label = "--" if not np.isfinite(value) else f"{float(value):.2f}"
            ax_right.text(col, row, label, ha="center", va="center", fontsize=9)

    fig.savefig(output_path, dpi=int(config["figures"]["dpi"]), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    setup_figure_style()
    run_dir = resolve_from_root(args.run_dir)
    run_manifest = read_json(run_dir / "run_manifest.json")
    dataset_manifest = read_json(resolve_from_root(run_manifest["dataset_root"]) / "manifest.json")
    output_dir = ensure_dir(resolve_from_root(args.output_dir or config["figures"]["root"]) / run_dir.name)

    _render_pipeline_overview(output_dir / "figure_01_pipeline_overview.png", run_manifest, dataset_manifest, config)
    _render_dataset_diversity(output_dir / "figure_02_dataset_diversity.png", dataset_manifest, config)
    _render_registration(output_dir / "figure_03_registration_before_after.png", run_manifest, config)
    _render_primitive_tolerance(output_dir / "figure_04_primitive_tolerance.png", run_manifest, dataset_manifest, config)
    _render_error_ablation(output_dir / "figure_05_error_ablation_failure_atlas.png", run_manifest, config)
    _render_family_benchmark_matrix(output_dir / "figure_06_family_benchmark_matrix.png", run_manifest, config)
    _render_capture_difficulty_vs_robustness(output_dir / "figure_07_capture_difficulty_vs_robustness.png", run_manifest, dataset_manifest, config)
    _render_primitive_confidence_tolerance(output_dir / "figure_08_primitive_confidence_tolerance.png", run_manifest, dataset_manifest, config)
    print(f"Exported figures to {output_dir.relative_to(project_root()).as_posix()}.")


if __name__ == "__main__":
    main()
