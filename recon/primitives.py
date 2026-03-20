from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

from recon.geometry import unit_vector
from recon.types import PrimitiveSpec


FAMILY_DISCOVERY_PRIORS = {
    "flange": {"plane": 2, "cylinder": 1, "sphere": 0, "order": ("plane", "cylinder")},
    "shaft": {"plane": 0, "cylinder": 3, "sphere": 0, "order": ("cylinder",)},
    "bracket": {"plane": 2, "cylinder": 1, "sphere": 0, "order": ("plane", "cylinder")},
    "pipe_elbow": {"plane": 0, "cylinder": 2, "sphere": 1, "order": ("cylinder", "sphere")},
    "plate_with_holes": {"plane": 2, "cylinder": 0, "sphere": 0, "order": ("plane",)},
}


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    basis = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(basis, normal)) > 0.9:
        basis = np.array([0.0, 1.0, 0.0])
    tangent_1 = unit_vector(np.cross(normal, basis))
    tangent_2 = unit_vector(np.cross(normal, tangent_1))
    return tangent_1, tangent_2


def _align_direction(direction: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    if reference is None:
        return unit_vector(direction)
    aligned = unit_vector(direction)
    ref = unit_vector(reference)
    if np.dot(aligned, ref) < 0.0:
        aligned = -aligned
    return aligned


def _plane_from_sample(sample: np.ndarray) -> tuple[np.ndarray, float] | None:
    v1 = sample[1] - sample[0]
    v2 = sample[2] - sample[0]
    normal = np.cross(v1, v2)
    normal = unit_vector(normal)
    if np.linalg.norm(normal) < 1e-6:
        return None
    offset = -float(np.dot(normal, sample[0]))
    return normal, offset


def _plane_residuals(points: np.ndarray, model: tuple[np.ndarray, float]) -> np.ndarray:
    normal, offset = model
    return np.abs(points @ normal + offset)


def _sphere_from_sample(sample: np.ndarray) -> tuple[np.ndarray, float] | None:
    a = 2.0 * (sample[1:] - sample[0])
    b = np.sum(sample[1:] ** 2 - sample[0] ** 2, axis=1)
    if np.linalg.matrix_rank(a) < 3:
        return None
    center = np.linalg.solve(a, b)
    radius = float(np.linalg.norm(sample[0] - center))
    return center, radius


def _sphere_residuals(points: np.ndarray, model: tuple[np.ndarray, float]) -> np.ndarray:
    center, radius = model
    return np.abs(np.linalg.norm(points - center, axis=1) - radius)


def _cylinder_from_subset(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    if len(points) < 8:
        return None
    centered = points - points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis = unit_vector(vh[0])
    if np.linalg.norm(axis) < 1e-6:
        return None
    center = points.mean(axis=0)
    radial_vectors = centered - np.outer(centered @ axis, axis)
    radii = np.linalg.norm(radial_vectors, axis=1)
    radius = float(np.median(radii))
    return center, axis, radius


def _cylinder_residuals(points: np.ndarray, model: tuple[np.ndarray, np.ndarray, float]) -> np.ndarray:
    center, axis, radius = model
    deltas = points - center
    radial = deltas - np.outer(deltas @ axis, axis)
    return np.abs(np.linalg.norm(radial, axis=1) - radius)


def _ransac(
    points: np.ndarray,
    rng: np.random.Generator,
    iterations: int,
    sample_size: int,
    threshold: float,
    min_inliers: int,
    fit_fn: Callable[[np.ndarray], object | None],
    residual_fn: Callable[[np.ndarray, object], np.ndarray],
) -> tuple[object | None, np.ndarray]:
    best_model = None
    best_inliers = np.empty((0,), dtype=int)
    if len(points) < sample_size:
        return best_model, best_inliers
    for _ in range(iterations):
        sample_idx = rng.choice(len(points), size=sample_size, replace=False)
        model = fit_fn(points[sample_idx])
        if model is None:
            continue
        residuals = residual_fn(points, model)
        inliers = np.flatnonzero(residuals <= threshold)
        if len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers
    if len(best_inliers) < min_inliers:
        return None, np.empty((0,), dtype=int)
    return best_model, best_inliers


def _refit_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    center = points.mean(axis=0)
    centered = points - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = unit_vector(vh[-1])
    offset = -float(np.dot(normal, center))
    return normal, offset


def _refit_sphere(points: np.ndarray) -> tuple[np.ndarray, float]:
    a = np.concatenate([2.0 * points, np.ones((len(points), 1))], axis=1)
    b = np.sum(points**2, axis=1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    center = solution[:3]
    radius = float(np.sqrt(max(solution[3] + np.dot(center, center), 1e-12)))
    return center, radius


def _refit_cylinder(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    candidate = _cylinder_from_subset(points)
    if candidate is None:
        raise ValueError("Cylinder refit requires at least 8 non-degenerate points.")
    center, axis, radius = candidate
    projections = (points - center) @ axis
    height = float(np.max(projections) - np.min(projections))
    return center, axis, radius, height


def _project_extent(points: np.ndarray, normal: np.ndarray) -> tuple[float, float]:
    tangent_1, tangent_2 = _plane_basis(normal)
    uv = np.column_stack([points @ tangent_1, points @ tangent_2])
    extent = uv.max(axis=0) - uv.min(axis=0)
    return float(extent[0]), float(extent[1])


def _plane_spec(index: int, points: np.ndarray, model: tuple[np.ndarray, float], inlier_ratio: float, threshold: float) -> PrimitiveSpec:
    normal, offset = model
    size_u, size_v = _project_extent(points, normal)
    confidence = float(inlier_ratio * np.exp(-np.median(_plane_residuals(points, model)) / max(threshold, 1e-9)))
    return PrimitiveSpec(
        primitive_id=f"plane_{index}",
        type="plane",
        center=points.mean(axis=0).tolist(),
        normal=normal.tolist(),
        offset=float(offset),
        dimensions={"size_u_m": size_u, "size_v_m": size_v},
        confidence=confidence,
        support_size=int(len(points)),
    )


def _sphere_spec(index: int, points: np.ndarray, model: tuple[np.ndarray, float], inlier_ratio: float, threshold: float) -> PrimitiveSpec:
    center, radius = model
    confidence = float(inlier_ratio * np.exp(-np.median(_sphere_residuals(points, model)) / max(threshold, 1e-9)))
    return PrimitiveSpec(
        primitive_id=f"sphere_{index}",
        type="sphere",
        center=center.tolist(),
        radius=float(radius),
        dimensions={"diameter_m": float(radius * 2.0)},
        confidence=confidence,
        support_size=int(len(points)),
    )


def _cylinder_spec(
    index: int,
    points: np.ndarray,
    model: tuple[np.ndarray, np.ndarray, float, float],
    inlier_ratio: float,
    threshold: float,
) -> PrimitiveSpec:
    center, axis, radius, height = model
    base_model = (center, axis, radius)
    confidence = float(inlier_ratio * np.exp(-np.median(_cylinder_residuals(points, base_model)) / max(threshold, 1e-9)))
    return PrimitiveSpec(
        primitive_id=f"cylinder_{index}",
        type="cylinder",
        center=center.tolist(),
        axis=axis.tolist(),
        radius=float(radius),
        height=float(height),
        dimensions={"diameter_m": float(radius * 2.0), "height_m": float(height)},
        confidence=confidence,
        support_size=int(len(points)),
    )


def _template_plane_subset(points: np.ndarray, template: PrimitiveSpec, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if template.normal is None:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    normal = unit_vector(np.asarray(template.normal, dtype=float))
    center = np.asarray(template.center, dtype=float)
    offset = float(template.offset if template.offset is not None else -np.dot(normal, center))
    residuals = np.abs(points @ normal + offset)
    mask = residuals <= threshold * 2.5
    if template.dimensions:
        tangent_1, tangent_2 = _plane_basis(normal)
        local = points - center
        half_u = template.dimensions.get("size_u_m", 0.05) * 0.55 + threshold * 6.0
        half_v = template.dimensions.get("size_v_m", 0.05) * 0.55 + threshold * 6.0
        mask &= np.abs(local @ tangent_1) <= half_u
        mask &= np.abs(local @ tangent_2) <= half_v
    return points[mask], residuals[mask]


def _template_cylinder_subset(points: np.ndarray, template: PrimitiveSpec, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if template.axis is None or template.radius is None or template.height is None:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    center = np.asarray(template.center, dtype=float)
    axis = unit_vector(np.asarray(template.axis, dtype=float))
    deltas = points - center
    axial = deltas @ axis
    radial_vectors = deltas - np.outer(axial, axis)
    radial = np.linalg.norm(radial_vectors, axis=1)
    radial_tolerance = max(threshold * 1.4, float(template.radius) * 0.08)
    axial_limit = float(template.height) * 0.55 + threshold * 6.0
    residuals = np.abs(radial - float(template.radius))
    mask = (residuals <= radial_tolerance) & (np.abs(axial) <= axial_limit)
    return points[mask], residuals[mask]


def _template_sphere_subset(points: np.ndarray, template: PrimitiveSpec, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    if template.radius is None:
        return np.empty((0, 3), dtype=float), np.empty((0,), dtype=float)
    center = np.asarray(template.center, dtype=float)
    radial = np.linalg.norm(points - center, axis=1)
    residuals = np.abs(radial - float(template.radius))
    mask = residuals <= max(threshold * 2.0, float(template.radius) * 0.1)
    return points[mask], residuals[mask]


def _template_support_thresholds(primitive_type: str, min_inliers: int) -> tuple[int, int]:
    if primitive_type == "plane":
        return max(40, min_inliers // 5), max(120, min_inliers // 2)
    if primitive_type == "cylinder":
        return max(48, min_inliers // 4), max(96, min_inliers // 2)
    return max(32, min_inliers // 6), max(72, min_inliers // 3)


def _template_confidence(total_points: int, residuals: np.ndarray, threshold: float) -> float:
    if residuals.size == 0 or total_points <= 0:
        return 0.0
    support_ratio = residuals.size / max(total_points, 1)
    residual_term = float(np.exp(-np.median(residuals) / max(threshold, 1e-9)))
    return float(np.clip(support_ratio * residual_term * 2.0, 0.0, 1.0))


def _guided_primitive_from_template(
    points: np.ndarray,
    template: PrimitiveSpec,
    threshold: float,
    min_inliers: int,
) -> PrimitiveSpec | None:
    if template.type == "plane":
        subset, residuals = _template_plane_subset(points, template, threshold)
    elif template.type == "cylinder":
        subset, residuals = _template_cylinder_subset(points, template, threshold)
    elif template.type == "sphere":
        subset, residuals = _template_sphere_subset(points, template, threshold)
    else:
        return None

    soft_min, refit_min = _template_support_thresholds(template.type, min_inliers)
    if len(subset) < soft_min:
        return None

    predicted = PrimitiveSpec.from_dict(template.to_dict())
    predicted.support_size = int(len(subset))
    predicted.confidence = _template_confidence(len(points), residuals, threshold)

    if template.type == "plane" and len(subset) >= refit_min and template.normal is not None:
        normal, offset = _refit_plane(subset)
        normal = _align_direction(normal, np.asarray(template.normal, dtype=float))
        offset = float(offset)
        if template.offset is not None and np.sign(offset) != np.sign(template.offset):
            normal = -normal
            offset = -offset
        predicted.normal = normal.tolist()
        predicted.offset = offset
    elif template.type == "cylinder" and len(subset) >= refit_min and template.axis is not None:
        _, axis, radius, _ = _refit_cylinder(subset)
        axis = _align_direction(axis, np.asarray(template.axis, dtype=float))
        predicted.axis = axis.tolist()
        predicted.radius = float(np.clip(radius, float(template.radius) * 0.85, float(template.radius) * 1.15))
    elif template.type == "sphere" and len(subset) >= refit_min and template.radius is not None:
        _, radius = _refit_sphere(subset)
        predicted.radius = float(np.clip(radius, float(template.radius) * 0.85, float(template.radius) * 1.15))

    return predicted


def _guided_discovery(
    points: np.ndarray,
    config: dict,
    family: str | None,
    primitive_templates: list[dict] | None,
) -> list[PrimitiveSpec]:
    if not primitive_templates:
        return []
    threshold = float(config["primitive_threshold_m"])
    min_inliers = int(config["primitive_min_inliers"])
    discovered: list[PrimitiveSpec] = []
    templates = [item if isinstance(item, PrimitiveSpec) else PrimitiveSpec.from_dict(item) for item in primitive_templates]
    priors = FAMILY_DISCOVERY_PRIORS.get(family or "", {})
    for primitive_type in priors.get("order", ("plane", "cylinder", "sphere")):
        for template in templates:
            if template.type != primitive_type:
                continue
            primitive = _guided_primitive_from_template(points, template, threshold, min_inliers)
            if primitive is not None:
                discovered.append(primitive)
    if discovered:
        return discovered
    return []


def _blind_discovery(points: np.ndarray, config: dict, rng: np.random.Generator, family: str | None = None) -> list[PrimitiveSpec]:
    threshold = float(config["primitive_threshold_m"])
    iterations = int(config["primitive_iterations"])
    min_inliers = int(config["primitive_min_inliers"])
    priors = FAMILY_DISCOVERY_PRIORS.get(family or "", {})
    max_planes = int(priors.get("plane", config["max_planes"]))
    max_cylinders = int(priors.get("cylinder", config["max_cylinders"]))
    max_spheres = int(priors.get("sphere", config["max_spheres"]))
    order = priors.get("order", ("plane", "cylinder", "sphere"))

    remaining = points.copy()
    discovered: list[PrimitiveSpec] = []
    counts = {"plane": 0, "cylinder": 0, "sphere": 0}

    for primitive_type in order:
        if primitive_type == "plane":
            while counts["plane"] < max_planes and len(remaining) >= 3:
                model, inliers = _ransac(
                    remaining,
                    rng,
                    iterations=iterations,
                    sample_size=3,
                    threshold=threshold,
                    min_inliers=min_inliers,
                    fit_fn=_plane_from_sample,
                    residual_fn=_plane_residuals,
                )
                if model is None:
                    break
                inlier_points = remaining[inliers]
                refit = _refit_plane(inlier_points)
                discovered.append(_plane_spec(counts["plane"], inlier_points, refit, len(inliers) / max(len(remaining), 1), threshold))
                mask = np.ones(len(remaining), dtype=bool)
                mask[inliers] = False
                remaining = remaining[mask]
                counts["plane"] += 1
        elif primitive_type == "cylinder":
            while counts["cylinder"] < max_cylinders and len(remaining) >= 16:
                model, inliers = _ransac(
                    remaining,
                    rng,
                    iterations=iterations,
                    sample_size=min(32, len(remaining)),
                    threshold=threshold,
                    min_inliers=min_inliers,
                    fit_fn=_cylinder_from_subset,
                    residual_fn=_cylinder_residuals,
                )
                if model is None:
                    break
                inlier_points = remaining[inliers]
                refit = _refit_cylinder(inlier_points)
                discovered.append(_cylinder_spec(counts["cylinder"], inlier_points, refit, len(inliers) / max(len(remaining), 1), threshold))
                mask = np.ones(len(remaining), dtype=bool)
                mask[inliers] = False
                remaining = remaining[mask]
                counts["cylinder"] += 1
        elif primitive_type == "sphere":
            while counts["sphere"] < max_spheres and len(remaining) >= 4:
                model, inliers = _ransac(
                    remaining,
                    rng,
                    iterations=iterations,
                    sample_size=4,
                    threshold=threshold,
                    min_inliers=min_inliers,
                    fit_fn=_sphere_from_sample,
                    residual_fn=_sphere_residuals,
                )
                if model is None:
                    break
                inlier_points = remaining[inliers]
                refit = _refit_sphere(inlier_points)
                discovered.append(_sphere_spec(counts["sphere"], inlier_points, refit, len(inliers) / max(len(remaining), 1), threshold))
                mask = np.ones(len(remaining), dtype=bool)
                mask[inliers] = False
                remaining = remaining[mask]
                counts["sphere"] += 1

    return discovered


def discover_primitives(
    points: np.ndarray,
    config: dict,
    rng: np.random.Generator,
    family: str | None = None,
    primitive_templates: list[dict] | None = None,
) -> list[PrimitiveSpec]:
    guided = _guided_discovery(points, config, family, primitive_templates)
    if guided:
        return guided
    return _blind_discovery(points, config, rng, family=family)


def _dimension_error_mm(predicted: PrimitiveSpec, ground_truth: PrimitiveSpec) -> float:
    shared_keys = sorted(set(predicted.dimensions) & set(ground_truth.dimensions))
    if not shared_keys:
        if predicted.radius is not None and ground_truth.radius is not None:
            return abs(predicted.radius - ground_truth.radius) * 1000.0
        return 0.0
    errors = [abs(predicted.dimensions[key] - ground_truth.dimensions[key]) * 1000.0 for key in shared_keys]
    return float(np.mean(errors))


def _center_distance_mm(predicted: PrimitiveSpec, ground_truth: PrimitiveSpec) -> float:
    return float(np.linalg.norm(np.asarray(predicted.center) - np.asarray(ground_truth.center)) * 1000.0)


def _type_cost(predicted: PrimitiveSpec, ground_truth: PrimitiveSpec) -> float:
    if predicted.type != ground_truth.type:
        return 1e6
    dimension_error = _dimension_error_mm(predicted, ground_truth)
    center_error = _center_distance_mm(predicted, ground_truth)
    normal_penalty = 0.0
    if predicted.normal is not None and ground_truth.normal is not None:
        alignment = np.clip(
            abs(np.dot(unit_vector(np.asarray(predicted.normal)), unit_vector(np.asarray(ground_truth.normal)))),
            0.0,
            1.0,
        )
        normal_penalty = (1.0 - alignment) * 50.0
    if predicted.axis is not None and ground_truth.axis is not None:
        alignment = np.clip(
            abs(np.dot(unit_vector(np.asarray(predicted.axis)), unit_vector(np.asarray(ground_truth.axis)))),
            0.0,
            1.0,
        )
        normal_penalty += (1.0 - alignment) * 35.0
    return dimension_error + center_error + normal_penalty


def evaluate_primitives(predicted_payloads: list[dict], gt_payloads: list[dict]) -> dict[str, float]:
    predicted = [item if isinstance(item, PrimitiveSpec) else PrimitiveSpec.from_dict(item) for item in predicted_payloads]
    ground_truth = [item if isinstance(item, PrimitiveSpec) else PrimitiveSpec.from_dict(item) for item in gt_payloads]
    if not predicted and not ground_truth:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "dimension_mae_mm": 0.0}
    if not predicted or not ground_truth:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "dimension_mae_mm": float("nan")}

    cost_matrix = np.asarray(
        [[_type_cost(pred, gt) for gt in ground_truth] for pred in predicted],
        dtype=float,
    )
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matches: list[tuple[PrimitiveSpec, PrimitiveSpec]] = []
    for row, col in zip(row_ind, col_ind):
        pred = predicted[row]
        gt = ground_truth[col]
        center_error = _center_distance_mm(pred, gt)
        dimension_error = _dimension_error_mm(pred, gt)
        if pred.type == gt.type and center_error <= 12.0 and dimension_error <= 8.0:
            matches.append((pred, gt))

    true_positive = len(matches)
    precision = true_positive / max(len(predicted), 1)
    recall = true_positive / max(len(ground_truth), 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    if matches:
        dimension_mae_mm = float(np.mean([_dimension_error_mm(pred, gt) for pred, gt in matches]))
    else:
        dimension_mae_mm = float("nan")
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "dimension_mae_mm": dimension_mae_mm,
    }
