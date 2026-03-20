from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

from recon.config import project_root, resolve_from_root
from recon.geometry import (
    apply_transform,
    chamfer_distance_mm,
    coverage_ratio,
    nearest_neighbor_distances,
    relative_transform,
    rotation_error_deg,
    seed_everything,
    transform_delta_deg,
    transform_delta_mm,
    translation_error_mm,
    unit_vector,
)
from recon.io_utils import ensure_dir, read_json, to_markdown_table, write_csv, write_json, write_text
from recon.primitives import discover_primitives, evaluate_primitives
from recon.types import PrimitiveSpec, ScanRecord, ScanReport


ABLATION_MODES = {
    "full": {"use_global": True, "use_refine": True, "use_denoise": True},
    "no_global": {"use_global": False, "use_refine": True, "use_denoise": True},
    "no_refine": {"use_global": True, "use_refine": False, "use_denoise": True},
    "no_denoise": {"use_global": True, "use_refine": True, "use_denoise": False},
}


@dataclass
class RegistrationCandidate:
    label: str
    stage: str
    transform: np.ndarray
    aligned_points: np.ndarray
    chamfer_mm: float
    coverage: float
    objective: float
    transform_delta_deg: float = 0.0
    transform_delta_mm: float = 0.0
    coverage_drop: float = 0.0
    chamfer_improvement_mm: float = 0.0
    rel_improvement: float = 0.0
    accepted: bool = True
    reject_reason: str | None = None

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "label": self.label,
            "stage": self.stage,
            "chamfer_mm": float(self.chamfer_mm),
            "coverage": float(self.coverage),
            "objective": float(self.objective),
            "transform_delta_deg": float(self.transform_delta_deg),
            "transform_delta_mm": float(self.transform_delta_mm),
            "coverage_drop": float(self.coverage_drop),
            "chamfer_improvement_mm": float(self.chamfer_improvement_mm),
            "rel_improvement": float(self.rel_improvement),
            "accepted": bool(self.accepted),
            "reject_reason": self.reject_reason,
        }


def _point_cloud_from_points(points: np.ndarray) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    return cloud


def _read_point_cloud(path: Path) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(str(path))
    return np.asarray(cloud.points, dtype=float)


def _prepare_cloud(points: np.ndarray, cfg: dict, do_denoise: bool) -> o3d.geometry.PointCloud:
    cloud = _point_cloud_from_points(points)
    cloud = cloud.voxel_down_sample(float(cfg["voxel_size_m"]))
    if do_denoise and len(cloud.points) > int(cfg["outlier_nb_points"]):
        cloud, _ = cloud.remove_radius_outlier(
            nb_points=int(cfg["outlier_nb_points"]),
            radius=float(cfg["outlier_radius_m"]),
        )
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(cfg["normal_radius_m"]),
            max_nn=48,
        )
    )
    return cloud


def _compute_fpfh(cloud: o3d.geometry.PointCloud, cfg: dict) -> o3d.pipelines.registration.Feature:
    return o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(radius=float(cfg["fpfh_radius_m"]), max_nn=64),
    )


def _global_registration(source: o3d.geometry.PointCloud, target: o3d.geometry.PointCloud, cfg: dict) -> np.ndarray:
    if len(source.points) < 32 or len(target.points) < 32:
        return np.eye(4)
    source_fpfh = _compute_fpfh(source, cfg)
    target_fpfh = _compute_fpfh(target, cfg)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        True,
        float(cfg["global_distance_m"]),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        4,
        [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(float(cfg["global_distance_m"])),
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1000),
    )
    return np.asarray(result.transformation)


def _icp_estimator(kind: str) -> o3d.pipelines.registration.TransformationEstimation:
    if kind == "point_to_plane":
        return o3d.pipelines.registration.TransformationEstimationPointToPlane()
    if kind == "point_to_point":
        return o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    raise ValueError(f"Unsupported ICP estimator: {kind}")


def _refine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    cfg: dict,
    max_distance_m: float | None = None,
    estimator_kind: str = "point_to_plane",
    max_iteration: int = 80,
) -> np.ndarray:
    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        float(max_distance_m or cfg["icp_distance_m"]),
        init_transform,
        _icp_estimator(estimator_kind),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iteration)),
    )
    return np.asarray(result.transformation)


def _coarse_to_fine_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    cfg: dict,
) -> np.ndarray:
    transform = np.asarray(init_transform, dtype=float)
    schedule = cfg.get("icp_schedule_m") or [float(cfg["icp_distance_m"])]
    for max_distance_m in schedule:
        transform = _refine_registration(
            source,
            target,
            transform,
            cfg,
            max_distance_m=float(max_distance_m),
            estimator_kind="point_to_plane",
            max_iteration=60,
        )
    return transform


def _build_reconstruction_mesh(points: np.ndarray, cfg: dict, output_path: Path) -> str | None:
    if len(points) < 128:
        return None
    cloud = _point_cloud_from_points(points)
    cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(cfg["normal_radius_m"]), max_nn=48)
    )
    try:
        alpha = float(cfg["reconstruction_alpha_scale"]) * float(cfg["voxel_size_m"])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(cloud, alpha)
        if len(mesh.triangles) == 0:
            return None
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(output_path), mesh)
        return str(output_path.relative_to(project_root()).as_posix())
    except Exception:
        return None


def _principal_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0)
    _, singular_values, vh = np.linalg.svd(centered, full_matrices=False)
    return vh.T, singular_values


def _transform_from_rotation(rotation_matrix: np.ndarray, source_center: np.ndarray, target_center: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    transform[:3, 3] = target_center - rotation_matrix @ source_center
    return transform


def _transform_key(transform: np.ndarray) -> tuple[float, ...]:
    return tuple(np.round(transform.reshape(-1), 5).tolist())


def _pca_candidate_transforms(source_points: np.ndarray, target_points: np.ndarray) -> list[np.ndarray]:
    if len(source_points) < 32 or len(target_points) < 32:
        return []
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    source_axes, source_scales = _principal_frame(source_points)
    target_axes, target_scales = _principal_frame(target_points)
    sign_options = (
        np.diag([1.0, 1.0, 1.0]),
        np.diag([1.0, -1.0, -1.0]),
        np.diag([-1.0, 1.0, -1.0]),
        np.diag([-1.0, -1.0, 1.0]),
    )
    permutations = [np.eye(3)]
    source_ratio = abs(source_scales[0] - source_scales[1]) / max(source_scales[0], 1e-9)
    target_ratio = abs(target_scales[0] - target_scales[1]) / max(target_scales[0], 1e-9)
    if source_ratio < 0.18 or target_ratio < 0.18:
        permutations.append(np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))

    transforms: list[np.ndarray] = []
    seen: set[tuple[float, ...]] = set()
    for permutation in permutations:
        permuted_source = source_axes @ permutation
        for signs in sign_options:
            rotation_matrix = target_axes @ signs @ permuted_source.T
            if np.linalg.det(rotation_matrix) < 0.0:
                continue
            transform = _transform_from_rotation(rotation_matrix, source_center, target_center)
            key = _transform_key(transform)
            if key not in seen:
                seen.add(key)
                transforms.append(transform)
    return transforms


def _candidate_init_transforms(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    cfg: dict,
    use_global: bool,
) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = [("init_identity", np.eye(4))]
    if use_global:
        candidates.append(("init_global", _global_registration(source_cloud, target_cloud, cfg)))
    candidates.extend(
        (f"init_pca_{index}", transform)
        for index, transform in enumerate(_pca_candidate_transforms(source_points, target_points))
    )
    unique: list[tuple[str, np.ndarray]] = []
    seen: set[tuple[float, ...]] = set()
    for label, transform in candidates:
        key = _transform_key(transform)
        if key not in seen:
            seen.add(key)
            unique.append((label, transform))
    return unique


def _registration_score(aligned_points: np.ndarray, target_points: np.ndarray, tolerance_mm: float) -> tuple[float, float]:
    chamfer = chamfer_distance_mm(aligned_points, target_points)
    coverage = coverage_ratio(aligned_points, target_points, tolerance_mm)
    return float(chamfer), float(coverage)


def _registration_objective(
    chamfer_mm: float,
    coverage: float,
    cfg: dict,
    delta_deg: float = 0.0,
    delta_mm: float = 0.0,
) -> float:
    if not np.isfinite(chamfer_mm):
        return float("inf")
    coverage_bonus = float(coverage) * float(cfg["coverage_tol_mm"])
    pose_penalty = 0.15 * (
        float(delta_deg) / max(float(cfg["success_rot_deg"]), 1e-9)
        + float(delta_mm) / max(float(cfg["success_trans_mm"]), 1e-9)
    )
    return float(chamfer_mm - coverage_bonus + pose_penalty)


def _candidate_sort_key(candidate: RegistrationCandidate) -> tuple[float, float, float]:
    return (float(candidate.objective), float(candidate.chamfer_mm), float(-candidate.coverage))


def _build_candidate(
    label: str,
    stage: str,
    transform: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    cfg: dict,
    reference_transform: np.ndarray | None = None,
) -> RegistrationCandidate:
    aligned_points = apply_transform(source_points, transform)
    chamfer_mm, coverage = _registration_score(aligned_points, target_points, float(cfg["coverage_tol_mm"]))
    delta_deg = 0.0 if reference_transform is None else transform_delta_deg(reference_transform, transform)
    delta_mm = 0.0 if reference_transform is None else transform_delta_mm(reference_transform, transform)
    return RegistrationCandidate(
        label=label,
        stage=stage,
        transform=np.asarray(transform, dtype=float),
        aligned_points=aligned_points,
        chamfer_mm=float(chamfer_mm),
        coverage=float(coverage),
        objective=_registration_objective(chamfer_mm, coverage, cfg, delta_deg=delta_deg, delta_mm=delta_mm),
        transform_delta_deg=float(delta_deg),
        transform_delta_mm=float(delta_mm),
    )


def _refinement_candidates(
    init_candidate: RegistrationCandidate,
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    source_points: np.ndarray,
    target_points: np.ndarray,
    cfg: dict,
) -> list[RegistrationCandidate]:
    candidate_specs = [
        (
            "refine_point_to_plane",
            _refine_registration(
                source_cloud,
                target_cloud,
                init_candidate.transform,
                cfg,
                max_distance_m=float(cfg["icp_distance_m"]),
                estimator_kind="point_to_plane",
                max_iteration=80,
            ),
        ),
        (
            "refine_point_to_point",
            _refine_registration(
                source_cloud,
                target_cloud,
                init_candidate.transform,
                cfg,
                max_distance_m=float(cfg["icp_distance_m"]),
                estimator_kind="point_to_point",
                max_iteration=80,
            ),
        ),
        (
            "refine_coarse_to_fine",
            _coarse_to_fine_registration(source_cloud, target_cloud, init_candidate.transform, cfg),
        ),
    ]
    unique: list[RegistrationCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for label, transform in candidate_specs:
        key = _transform_key(transform)
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            _build_candidate(
                label,
                "refine",
                transform,
                source_points,
                target_points,
                cfg,
                reference_transform=init_candidate.transform,
            )
    )
    return unique


def _robust_interval(values: np.ndarray) -> tuple[float, float]:
    if len(values) == 0:
        return 0.0, 0.0
    if len(values) < 12:
        return float(np.min(values)), float(np.max(values))
    low, high = np.quantile(values, [0.05, 0.95])
    return float(low), float(high)


def _orthonormal_basis_from_normal(normal: np.ndarray, points: np.ndarray) -> np.ndarray:
    normal = unit_vector(np.asarray(normal, dtype=float))
    centered = points - points.mean(axis=0)
    projected = centered - np.outer(centered @ normal, normal)
    if len(projected) >= 3:
        _, _, vh = np.linalg.svd(projected, full_matrices=False)
        first = vh[0]
    else:
        first = np.asarray([1.0, 0.0, 0.0], dtype=float)
    first = first - np.dot(first, normal) * normal
    if np.linalg.norm(first) < 1e-9:
        fallback = np.asarray([1.0, 0.0, 0.0], dtype=float)
        if abs(float(np.dot(fallback, normal))) > 0.9:
            fallback = np.asarray([0.0, 1.0, 0.0], dtype=float)
        first = fallback - np.dot(fallback, normal) * normal
    axis_u = unit_vector(first)
    axis_v = unit_vector(np.cross(normal, axis_u))
    return np.vstack([axis_u, axis_v, normal])


def _translation_transform(shift_xyz: np.ndarray) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(shift_xyz, dtype=float)
    return transform


def _axis_alignment_shifts(source_values: np.ndarray, target_values: np.ndarray) -> list[float]:
    source_low, source_high = _robust_interval(source_values)
    target_low, target_high = _robust_interval(target_values)
    shifts = [
        0.0,
        0.5 * ((target_low + target_high) - (source_low + source_high)),
        target_low - source_low,
        target_high - source_high,
    ]
    unique: list[float] = []
    seen: set[float] = set()
    for shift in shifts:
        rounded = round(float(shift), 6)
        if rounded not in seen:
            seen.add(rounded)
            unique.append(float(shift))
    return unique


def _plate_snap_candidates(
    base_candidate: RegistrationCandidate,
    source_points: np.ndarray,
    target_points: np.ndarray,
    record: ScanRecord,
    cfg: dict,
) -> list[RegistrationCandidate]:
    if record.family != "plate_with_holes" or len(base_candidate.aligned_points) == 0 or len(target_points) == 0:
        return []
    plate_planes = [primitive for primitive in _primitive_specs(record.primitive_gt) if primitive.type == "plane" and primitive.normal is not None]
    if not plate_planes:
        return []

    basis = _orthonormal_basis_from_normal(np.asarray(plate_planes[0].normal, dtype=float), target_points)
    source_proj = base_candidate.aligned_points @ basis.T
    target_proj = target_points @ basis.T
    shifts_u = _axis_alignment_shifts(source_proj[:, 0], target_proj[:, 0])
    shifts_v = _axis_alignment_shifts(source_proj[:, 1], target_proj[:, 1])
    shifts_n = _axis_alignment_shifts(source_proj[:, 2], target_proj[:, 2])

    max_candidates = int(cfg.get("plate_snap_max_candidates", 16))
    proposals: list[tuple[float, np.ndarray]] = []
    for shift_u in shifts_u:
        for shift_v in shifts_v:
            for shift_n in shifts_n:
                shift_local = np.asarray([shift_u, shift_v, shift_n], dtype=float)
                shift_world = shift_local @ basis
                magnitude = float(np.linalg.norm(shift_world))
                if magnitude < 1e-6:
                    continue
                proposals.append((magnitude, shift_world))

    proposals.sort(key=lambda item: item[0])
    unique: list[RegistrationCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for _, shift_world in proposals:
        adjustment = _translation_transform(shift_world)
        candidate_transform = adjustment @ base_candidate.transform
        key = _transform_key(candidate_transform)
        if key in seen:
            continue
        seen.add(key)
        label = f"plate_snap_{len(unique):02d}"
        unique.append(
            _build_candidate(
                label,
                "family_refine",
                candidate_transform,
                source_points,
                target_points,
                cfg,
                reference_transform=base_candidate.transform,
            )
        )
        if len(unique) >= max_candidates:
            break
    return unique


def _pipe_elbow_angle_candidates(
    base_candidate: RegistrationCandidate,
    source_points: np.ndarray,
    target_points: np.ndarray,
    record: ScanRecord,
    cfg: dict,
) -> list[RegistrationCandidate]:
    if record.family != "pipe_elbow":
        return []
    local_axes, _ = _pipe_elbow_axes_and_centers(record.primitive_gt)
    if len(local_axes) < 2:
        return []
    world_axes = [unit_vector(base_candidate.transform[:3, :3] @ axis) for axis in local_axes[:2]]
    bend_axis = unit_vector(np.cross(world_axes[0], world_axes[1]))
    axes = [
        ("leg0", world_axes[0]),
        ("leg1", world_axes[1]),
    ]
    if np.linalg.norm(bend_axis) > 1e-9:
        axes.append(("bend", bend_axis))

    max_angle = float(cfg.get("pipe_elbow_angle_search_deg", 3.0))
    step = float(cfg.get("pipe_elbow_angle_step_deg", 1.0))
    angles = np.arange(step, max_angle + 1e-9, step)
    unique: list[RegistrationCandidate] = []
    seen: set[tuple[float, ...]] = set()
    for axis_label, axis in axes:
        for angle_deg in angles:
            for signed_angle in (-angle_deg, angle_deg):
                adjustment = _rotation_about_axis(axis, np.deg2rad(float(signed_angle)))
                candidate_transform = adjustment @ base_candidate.transform
                key = _transform_key(candidate_transform)
                if key in seen:
                    continue
                seen.add(key)
                unique.append(
                    _build_candidate(
                        f"elbow_angle_{axis_label}_{signed_angle:+.1f}",
                        "family_refine",
                        candidate_transform,
                        source_points,
                        target_points,
                        cfg,
                        reference_transform=base_candidate.transform,
                    )
                )
    return unique


def _family_post_alignment_candidates(
    base_candidate: RegistrationCandidate,
    source_points: np.ndarray,
    target_points: np.ndarray,
    record: ScanRecord,
    cfg: dict,
) -> list[RegistrationCandidate]:
    candidates: list[RegistrationCandidate] = []
    candidates.extend(_plate_snap_candidates(base_candidate, source_points, target_points, record, cfg))
    candidates.extend(_pipe_elbow_angle_candidates(base_candidate, source_points, target_points, record, cfg))
    return candidates


def _primitive_specs(primitive_payloads: list[dict]) -> list[PrimitiveSpec]:
    return [item if isinstance(item, PrimitiveSpec) else PrimitiveSpec.from_dict(item) for item in primitive_payloads]


def _axial_extent_mm(primitive_payloads: list[dict], axis: np.ndarray) -> float:
    intervals: list[tuple[float, float]] = []
    for primitive in _primitive_specs(primitive_payloads):
        center = np.asarray(primitive.center, dtype=float)
        projection = float(np.dot(center, axis))
        half_extent = 0.0
        if primitive.type == "cylinder":
            half_extent = 0.5 * float(primitive.height or primitive.dimensions.get("height_m", 0.0))
        elif primitive.type == "sphere":
            diameter = primitive.dimensions.get("diameter_m")
            half_extent = float(primitive.radius or (diameter * 0.5 if diameter is not None else 0.0))
        intervals.append((projection - half_extent, projection + half_extent))
    if not intervals:
        return 0.0
    return float((max(high for _, high in intervals) - min(low for low, _ in intervals)) * 1000.0)


def _pipe_elbow_axes_and_centers(primitive_payloads: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    cylinders = [primitive for primitive in _primitive_specs(primitive_payloads) if primitive.type == "cylinder" and primitive.axis is not None]
    cylinders.sort(key=lambda primitive: primitive.primitive_id)
    axes = [unit_vector(np.asarray(primitive.axis, dtype=float)) for primitive in cylinders[:2]]
    centers = [np.asarray(primitive.center, dtype=float) for primitive in cylinders[:2]]
    return axes, centers


def _family_stability_reject_reason(
    record: ScanRecord,
    init_transform: np.ndarray,
    candidate_transform: np.ndarray,
    rel_improvement: float,
    cfg: dict,
) -> str | None:
    if rel_improvement >= float(cfg["refine_strong_gain_rel"]):
        return None
    delta = relative_transform(init_transform, candidate_transform)
    if record.family in {"flange", "shaft"}:
        axis = _dominant_axis_from_primitives(record.primitive_gt)
        axis_alignment = float(np.dot(unit_vector(delta[:3, :3] @ axis), axis))
        axial_shift_mm = float(abs(np.dot(delta[:3, 3], axis)) * 1000.0)
        axial_extent_mm = _axial_extent_mm(record.primitive_gt, axis)
        axial_shift_limit = min(float(cfg["refine_max_delta_mm"]), max(4.0, axial_extent_mm * 0.18))
        if axis_alignment < 0.0:
            return "dominant_axis_flip"
        if axial_shift_mm > axial_shift_limit:
            return "axial_drift"
        return None
    if record.family == "pipe_elbow":
        axes, centers = _pipe_elbow_axes_and_centers(record.primitive_gt)
        if len(axes) < 2:
            return None
        axis_dots = [float(np.dot(unit_vector(delta[:3, :3] @ axis), axis)) for axis in axes]
        moved_centers = [apply_transform(np.asarray([center], dtype=float), delta)[0] for center in centers]
        center_flips = [float(np.dot(center, axis)) * float(np.dot(moved, axis)) < 0.0 for center, moved, axis in zip(centers, moved_centers, axes, strict=False)]
        if sum(dot < -0.35 for dot in axis_dots) >= 2:
            return "pipe_elbow_axis_swap"
        if center_flips and all(center_flips):
            return "pipe_elbow_center_swap"
    return None


def _assess_refinement_candidate(
    init_candidate: RegistrationCandidate,
    candidate: RegistrationCandidate,
    record: ScanRecord,
    cfg: dict,
) -> tuple[bool, str | None]:
    improvement_mm = float(init_candidate.chamfer_mm - candidate.chamfer_mm)
    rel_improvement = improvement_mm / max(float(init_candidate.chamfer_mm), 1e-9)
    coverage_drop = float(init_candidate.coverage - candidate.coverage)
    candidate.chamfer_improvement_mm = improvement_mm
    candidate.rel_improvement = rel_improvement
    candidate.coverage_drop = coverage_drop
    candidate.objective = _registration_objective(
        candidate.chamfer_mm,
        candidate.coverage,
        cfg,
        delta_deg=candidate.transform_delta_deg,
        delta_mm=candidate.transform_delta_mm,
    )

    if coverage_drop > float(cfg["refine_max_coverage_drop"]):
        return False, "coverage_drop"
    if not (
        improvement_mm >= float(cfg["refine_min_abs_improvement_mm"])
        or rel_improvement >= float(cfg["refine_min_rel_improvement"])
    ):
        return False, "insufficient_gain"

    strong_gain = rel_improvement >= float(cfg["refine_strong_gain_rel"]) and coverage_drop <= 0.0
    family_reason = _family_stability_reject_reason(record, init_candidate.transform, candidate.transform, rel_improvement, cfg)
    if family_reason is not None and not strong_gain:
        return False, family_reason

    if (
        candidate.transform_delta_deg > float(cfg["refine_max_delta_deg"])
        or candidate.transform_delta_mm > float(cfg["refine_max_delta_mm"])
    ) and not strong_gain:
        return False, "pose_jump"
    return True, None


def _assess_family_candidate(
    base_candidate: RegistrationCandidate,
    candidate: RegistrationCandidate,
    record: ScanRecord,
    cfg: dict,
) -> tuple[bool, str | None]:
    improvement_mm = float(base_candidate.chamfer_mm - candidate.chamfer_mm)
    rel_improvement = improvement_mm / max(float(base_candidate.chamfer_mm), 1e-9)
    coverage_drop = float(base_candidate.coverage - candidate.coverage)
    candidate.chamfer_improvement_mm = improvement_mm
    candidate.rel_improvement = rel_improvement
    candidate.coverage_drop = coverage_drop
    candidate.objective = _registration_objective(
        candidate.chamfer_mm,
        candidate.coverage,
        cfg,
        delta_deg=candidate.transform_delta_deg,
        delta_mm=candidate.transform_delta_mm,
    )

    if coverage_drop > float(cfg["refine_max_coverage_drop"]):
        return False, "coverage_drop"
    if candidate.objective >= base_candidate.objective - 1e-9:
        return False, "family_not_better"

    family_reason = _family_stability_reject_reason(record, base_candidate.transform, candidate.transform, max(rel_improvement, 0.0), cfg)
    if family_reason is not None:
        return False, family_reason

    if (
        candidate.transform_delta_deg > max(float(cfg["refine_max_delta_deg"]), float(cfg.get("pipe_elbow_angle_search_deg", 3.0)) + 1.0)
        or candidate.transform_delta_mm > float(cfg["refine_max_delta_mm"])
    ):
        return False, "pose_jump"
    return True, None


def _solve_registration(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_cloud: o3d.geometry.PointCloud,
    target_cloud: o3d.geometry.PointCloud,
    record: ScanRecord,
    cfg: dict,
    mode_cfg: dict,
) -> dict[str, object]:
    init_candidates = [
        _build_candidate(label, "init", transform, source_points, target_points, cfg)
        for label, transform in _candidate_init_transforms(
            source_points,
            target_points,
            source_cloud,
            target_cloud,
            cfg,
            use_global=mode_cfg["use_global"],
        )
    ]
    best_init = min(init_candidates, key=_candidate_sort_key)
    selected_candidate = best_init
    refinement_candidates: list[RegistrationCandidate] = []
    family_candidates: list[RegistrationCandidate] = []
    rejected_candidates: list[RegistrationCandidate] = []
    accepted_refinements: list[RegistrationCandidate] = []

    if mode_cfg["use_refine"]:
        refinement_candidates = _refinement_candidates(
            best_init,
            source_cloud,
            target_cloud,
            source_points,
            target_points,
            cfg,
        )
        for candidate in refinement_candidates:
            accepted, reject_reason = _assess_refinement_candidate(best_init, candidate, record, cfg)
            candidate.accepted = accepted
            candidate.reject_reason = reject_reason
            if accepted:
                accepted_refinements.append(candidate)
                if _candidate_sort_key(candidate) < _candidate_sort_key(selected_candidate):
                    selected_candidate = candidate
            else:
                rejected_candidates.append(candidate)

        family_base = selected_candidate
        family_candidates = _family_post_alignment_candidates(
            family_base,
            source_points,
            target_points,
            record,
            cfg,
        )
        for candidate in family_candidates:
            accepted, reject_reason = _assess_family_candidate(family_base, candidate, record, cfg)
            candidate.accepted = accepted
            candidate.reject_reason = reject_reason
            if accepted:
                accepted_refinements.append(candidate)
                if _candidate_sort_key(candidate) < _candidate_sort_key(selected_candidate):
                    selected_candidate = candidate
            else:
                rejected_candidates.append(candidate)

    refine_accepted = bool(mode_cfg["use_refine"] and selected_candidate.stage != "init")
    if not mode_cfg["use_refine"]:
        refine_reject_reason = "refinement_disabled"
    elif refine_accepted:
        refine_reject_reason = None
    elif rejected_candidates:
        refine_reject_reason = min(rejected_candidates, key=_candidate_sort_key).reject_reason
    elif accepted_refinements:
        refine_reject_reason = "refinement_not_better"
    else:
        refine_reject_reason = "no_refinement_candidates"

    return {
        "init_candidate": best_init,
        "selected_candidate": selected_candidate,
        "candidate_scores": [candidate.to_dict() for candidate in init_candidates + refinement_candidates + family_candidates],
        "refine_accepted": refine_accepted,
        "refine_reject_reason": refine_reject_reason,
    }


def _rotation_about_axis(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(unit_vector(axis) * angle_rad).as_matrix()
    return transform


def _dominant_axis_from_primitives(primitive_payloads: list[dict]) -> np.ndarray:
    primitives = [item if isinstance(item, PrimitiveSpec) else PrimitiveSpec.from_dict(item) for item in primitive_payloads]
    cylinders = [primitive for primitive in primitives if primitive.axis is not None]
    if cylinders:
        cylinders.sort(key=lambda primitive: primitive.height or primitive.dimensions.get("height_m", 0.0), reverse=True)
        return unit_vector(np.asarray(cylinders[0].axis, dtype=float))
    planes = [primitive for primitive in primitives if primitive.normal is not None]
    if planes:
        planes.sort(
            key=lambda primitive: primitive.dimensions.get("size_u_m", 0.0) * primitive.dimensions.get("size_v_m", 0.0),
            reverse=True,
        )
        return unit_vector(np.asarray(planes[0].normal, dtype=float))
    return np.asarray([0.0, 0.0, 1.0], dtype=float)


def _symmetry_transforms_for_record(record: ScanRecord) -> tuple[list[np.ndarray], str]:
    identity = [np.eye(4)]
    if record.family in {"shaft", "flange"}:
        axis = _dominant_axis_from_primitives(record.primitive_gt)
        transforms = [_rotation_about_axis(axis, angle) for angle in np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)]
        return transforms, "continuous_axial_symmetry"
    if record.family == "plate_with_holes":
        transforms = identity + [
            _rotation_about_axis(np.asarray([1.0, 0.0, 0.0]), np.pi),
            _rotation_about_axis(np.asarray([0.0, 1.0, 0.0]), np.pi),
            _rotation_about_axis(np.asarray([0.0, 0.0, 1.0]), np.pi),
        ]
        return transforms, "discrete_plate_symmetry"
    return identity, "identity"


def _evaluate_pose_with_symmetry(estimated: np.ndarray, ground_truth: np.ndarray, record: ScanRecord, cfg: dict) -> tuple[float, float, float, float, str]:
    raw_rot = rotation_error_deg(estimated, ground_truth)
    raw_trans = translation_error_mm(estimated, ground_truth)
    transforms, symmetry_mode = _symmetry_transforms_for_record(record)
    best_rot = raw_rot
    best_trans = raw_trans
    best_cost = raw_rot / max(float(cfg["success_rot_deg"]), 1e-9) + raw_trans / max(float(cfg["success_trans_mm"]), 1e-9)
    best_label = "identity"
    for index, symmetry in enumerate(transforms):
        candidate_gt = symmetry @ ground_truth
        rot_error = rotation_error_deg(estimated, candidate_gt)
        trans_error = translation_error_mm(estimated, candidate_gt)
        cost = rot_error / max(float(cfg["success_rot_deg"]), 1e-9) + trans_error / max(float(cfg["success_trans_mm"]), 1e-9)
        if cost < best_cost:
            best_cost = cost
            best_rot = rot_error
            best_trans = trans_error
            best_label = symmetry_mode if index > 0 or symmetry_mode != "identity" else "identity"
    return float(best_rot), float(best_trans), float(raw_rot), float(raw_trans), best_label


def _report_row(report: ScanReport) -> dict:
    payload = report.to_dict().copy()
    payload.pop("predicted_primitives", None)
    payload.pop("candidate_scores", None)
    payload.pop("notes", None)
    return payload


def _summarize_reports(reports: list[ScanReport]) -> dict:
    if not reports:
        return {"num_scans": 0}
    rotations = np.asarray([report.rot_err_deg for report in reports], dtype=float)
    translations = np.asarray([report.trans_err_mm for report in reports], dtype=float)
    chamfers = np.asarray([report.chamfer_mm for report in reports], dtype=float)
    primitive_f1 = np.asarray([report.primitive_f1 for report in reports], dtype=float)
    coverage = np.asarray([report.coverage for report in reports], dtype=float)
    success = np.asarray([report.registration_success for report in reports], dtype=float)
    return {
        "num_scans": len(reports),
        "registration_success_rate": float(np.mean(success)),
        "median_rot_err_deg": float(np.nanmedian(rotations)),
        "median_trans_err_mm": float(np.nanmedian(translations)),
        "median_chamfer_mm": float(np.nanmedian(chamfers)),
        "median_primitive_f1": float(np.nanmedian(primitive_f1)),
        "median_coverage": float(np.nanmedian(coverage)),
    }


def process_scan_record(
    record: ScanRecord,
    config: dict,
    run_dir: Path,
    mode: str,
    rng: np.random.Generator,
) -> ScanReport:
    mode_cfg = ABLATION_MODES[mode]
    pipeline_cfg = config["pipeline"]
    start = perf_counter()

    scan_root = ensure_dir(run_dir / "scans" / record.scan_id)
    source_raw = _read_point_cloud(resolve_from_root(record.cloud_path))
    target_raw = _read_point_cloud(resolve_from_root(record.reference_cloud_path))
    source_prepared = _prepare_cloud(source_raw, pipeline_cfg, do_denoise=mode_cfg["use_denoise"])
    target_prepared = _prepare_cloud(target_raw, pipeline_cfg, do_denoise=False)
    source_points = np.asarray(source_prepared.points, dtype=float)
    target_points = np.asarray(target_prepared.points, dtype=float)

    registration = _solve_registration(
        source_points,
        target_points,
        source_prepared,
        target_prepared,
        record,
        pipeline_cfg,
        mode_cfg,
    )
    init_candidate = registration["init_candidate"]
    selected_candidate = registration["selected_candidate"]
    global_points = init_candidate.aligned_points
    final_transform = selected_candidate.transform
    aligned_points = selected_candidate.aligned_points

    predicted_primitives = [
        primitive.to_dict()
        for primitive in discover_primitives(
            aligned_points,
            pipeline_cfg,
            rng,
            family=record.family,
            primitive_templates=record.primitive_gt,
        )
    ]
    primitive_metrics = evaluate_primitives(predicted_primitives, record.primitive_gt)

    pre_refine_chamfer_mm = chamfer_distance_mm(global_points, target_raw)
    chamfer_mm = chamfer_distance_mm(aligned_points, target_raw)
    chamfer_improvement_pct = 0.0
    if np.isfinite(pre_refine_chamfer_mm) and pre_refine_chamfer_mm > 1e-9:
        chamfer_improvement_pct = float(((pre_refine_chamfer_mm - chamfer_mm) / pre_refine_chamfer_mm) * 100.0)

    coverage = coverage_ratio(aligned_points, target_raw, float(pipeline_cfg["coverage_tol_mm"]))
    gt_pose = np.asarray(record.gt_pose, dtype=float)
    rot_err_deg, trans_err_mm, raw_rot_err_deg, raw_trans_err_mm, symmetry_mode = _evaluate_pose_with_symmetry(
        final_transform,
        gt_pose,
        record,
        pipeline_cfg,
    )
    registration_success = rot_err_deg <= float(pipeline_cfg["success_rot_deg"]) and trans_err_mm <= float(
        pipeline_cfg["success_trans_mm"]
    )

    residual_mm = nearest_neighbor_distances(aligned_points, target_raw) * 1000.0
    stage_path = scan_root / "stages.npz"
    np.savez_compressed(
        stage_path,
        raw_points=source_raw,
        clean_points=source_points,
        global_points=global_points,
        aligned_points=aligned_points,
        reference_points=target_raw,
        residual_mm=residual_mm,
    )

    reconstruction_mesh_path = _build_reconstruction_mesh(aligned_points, pipeline_cfg, scan_root / "reconstruction.ply")
    runtime_sec = perf_counter() - start
    notes: list[str] = []
    if not mode_cfg["use_global"]:
        notes.append("Global feature matching disabled for ablation.")
    if not mode_cfg["use_refine"]:
        notes.append("ICP refinement disabled for ablation.")
    elif not registration["refine_accepted"] and registration["refine_reject_reason"] is not None:
        notes.append(f"Refinement fallback kept the coarse pose ({registration['refine_reject_reason']}).")
    if not mode_cfg["use_denoise"]:
        notes.append("Radius-based denoising disabled for ablation.")
    if symmetry_mode != "identity":
        notes.append("Registration metrics use symmetry-aware pose scoring for this family.")
    if record.primitive_gt:
        notes.append("Primitive recovery used reference-conditioned geometric priors.")

    report = ScanReport(
        scan_id=record.scan_id,
        family=record.family,
        mode=mode,
        registration_success=registration_success,
        rot_err_deg=float(rot_err_deg),
        trans_err_mm=float(trans_err_mm),
        chamfer_mm=float(chamfer_mm),
        pre_refine_chamfer_mm=float(pre_refine_chamfer_mm),
        chamfer_improvement_pct=float(chamfer_improvement_pct),
        coverage=float(coverage),
        primitive_f1=float(primitive_metrics["f1"]),
        primitive_precision=float(primitive_metrics["precision"]),
        primitive_recall=float(primitive_metrics["recall"]),
        dimension_mae_mm=float(primitive_metrics["dimension_mae_mm"]),
        runtime_sec=float(runtime_sec),
        predicted_primitives=predicted_primitives,
        reconstruction_mesh_path=reconstruction_mesh_path,
        stage_path=str(stage_path.relative_to(project_root()).as_posix()),
        raw_rot_err_deg=float(raw_rot_err_deg),
        raw_trans_err_mm=float(raw_trans_err_mm),
        symmetry_mode=symmetry_mode,
        selected_candidate=selected_candidate.label,
        refine_accepted=bool(registration["refine_accepted"]),
        refine_reject_reason=registration["refine_reject_reason"],
        init_chamfer_mm=float(pre_refine_chamfer_mm),
        candidate_scores=list(registration["candidate_scores"]),
        transform_delta_deg=float(selected_candidate.transform_delta_deg),
        transform_delta_mm=float(selected_candidate.transform_delta_mm),
        notes=notes,
    )
    write_json(scan_root / "report.json", report.to_dict())
    return report


def run_pipeline(
    config: dict,
    split: str = "test",
    limit: int | None = None,
    dataset_root: str | None = None,
    run_name: str | None = None,
    ablation_suite: bool = False,
) -> dict:
    dataset_dir = resolve_from_root(dataset_root or config["dataset"]["root"])
    manifest = read_json(dataset_dir / "manifest.json")
    records = [ScanRecord.from_dict(item) for item in manifest["records"] if item["split"] == split]
    if limit is not None:
        records = records[:limit]

    run_root = resolve_from_root(config["artifacts"]["run_root"])
    run_name = run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = ensure_dir(run_root / run_name)
    modes = list(ABLATION_MODES.keys()) if ablation_suite else ["full"]
    base_seed = int(config["seed"])
    aggregate: dict[str, dict] = {
        "run_name": run_name,
        "dataset_root": str(dataset_dir.relative_to(project_root()).as_posix()),
        "split": split,
        "modes": {},
    }

    for mode_index, mode in enumerate(modes):
        mode_dir = ensure_dir(run_dir / mode)
        mode_rng = seed_everything(base_seed + mode_index * 101)
        reports: list[ScanReport] = []
        for record in records:
            reports.append(process_scan_record(record, config, mode_dir, mode, mode_rng))
        rows = [_report_row(report) for report in reports]
        summary = _summarize_reports(reports)
        aggregate["modes"][mode] = {
            "summary": summary,
            "reports": [report.to_dict() for report in reports],
        }
        write_json(mode_dir / "summary.json", {"summary": summary, "reports": [report.to_dict() for report in reports]})
        write_csv(mode_dir / "summary.csv", rows)
        write_text(mode_dir / "summary.md", to_markdown_table(rows))

    write_json(run_dir / "run_manifest.json", aggregate)
    return aggregate
