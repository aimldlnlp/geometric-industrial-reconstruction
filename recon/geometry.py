from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def seed_everything(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def unit_vector(vector: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < eps:
        return vector.copy()
    return vector / norm


def make_transform(rotation_deg_xyz: Iterable[float], translation_xyz: Iterable[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_euler("xyz", rotation_deg_xyz, degrees=True).as_matrix()
    transform[:3, 3] = np.asarray(list(translation_xyz), dtype=float)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def relative_transform(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    return candidate @ invert_transform(reference)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    transformed = homogeneous @ transform.T
    return transformed[:, :3]


def random_pose(rng: np.random.Generator, max_rotation_deg: float, max_translation_m: float) -> np.ndarray:
    rotation = rng.uniform(-max_rotation_deg, max_rotation_deg, size=3)
    translation = rng.uniform(-max_translation_m, max_translation_m, size=3)
    return make_transform(rotation, translation)


def jitter_transform(rng: np.random.Generator, max_rotation_deg: float, max_translation_m: float) -> np.ndarray:
    return random_pose(rng, max_rotation_deg=max_rotation_deg, max_translation_m=max_translation_m)


def rotation_error_deg(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    delta = estimated[:3, :3] @ ground_truth[:3, :3].T
    trace_value = float((np.trace(delta) - 1.0) * 0.5)
    if trace_value >= 1.0 - 1e-12:
        return 0.0
    if trace_value <= -1.0 + 1e-12:
        return 180.0
    trace_value = float(np.clip(trace_value, -1.0, 1.0))
    return math.degrees(math.acos(trace_value))


def translation_error_mm(estimated: np.ndarray, ground_truth: np.ndarray) -> float:
    return float(np.linalg.norm(estimated[:3, 3] - ground_truth[:3, 3]) * 1000.0)


def transform_delta_deg(reference: np.ndarray, candidate: np.ndarray) -> float:
    return rotation_error_deg(candidate, reference)


def transform_delta_mm(reference: np.ndarray, candidate: np.ndarray) -> float:
    return translation_error_mm(candidate, reference)


def sample_mesh_surface(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    points, face_indices = trimesh.sample.sample_surface(mesh, count=count, seed=rng.integers(0, 2**31 - 1))
    normals = mesh.face_normals[face_indices]
    return points.astype(float), normals.astype(float)


def sample_reference_cloud(mesh: trimesh.Trimesh, count: int, rng: np.random.Generator) -> np.ndarray:
    points, _ = sample_mesh_surface(mesh, count, rng)
    return points


def visible_mask(normals: np.ndarray, view_direction: np.ndarray, visibility_cosine: float) -> np.ndarray:
    return normals @ unit_vector(view_direction) > visibility_cosine


def simulate_multiview_visibility(
    points: np.ndarray,
    normals: np.ndarray,
    rng: np.random.Generator,
    min_views: int,
    max_views: int,
    visibility_cosine: float,
    dropout_range: tuple[float, float],
) -> tuple[np.ndarray, dict[str, float]]:
    candidate_views = np.asarray(
        [
            [1.0, 0.0, 0.4],
            [-1.0, 0.2, 0.6],
            [0.2, 1.0, 0.6],
            [0.2, -1.0, 0.5],
            [0.4, 0.3, 1.0],
            [-0.5, -0.4, 1.0],
        ],
        dtype=float,
    )
    view_count = int(rng.integers(min_views, max_views + 1))
    indices = rng.choice(len(candidate_views), size=view_count, replace=False)
    keep_indices: list[np.ndarray] = []
    dropout = float(rng.uniform(*dropout_range))
    for idx in indices:
        mask = visible_mask(normals, candidate_views[idx], visibility_cosine)
        visible_idx = np.flatnonzero(mask)
        if len(visible_idx) == 0:
            continue
        retained = rng.random(len(visible_idx)) > dropout
        keep_indices.append(visible_idx[retained])
    if keep_indices:
        merged = np.unique(np.concatenate(keep_indices))
    else:
        merged = np.arange(len(points))
    return merged, {"views_used": float(view_count), "occlusion_dropout": dropout}


def add_noise_and_outliers(
    points: np.ndarray,
    rng: np.random.Generator,
    noise_std_m: float,
    outlier_ratio: float,
) -> np.ndarray:
    if points.size == 0:
        return points.copy()
    noisy = points + rng.normal(scale=noise_std_m, size=points.shape)
    outlier_count = int(len(noisy) * outlier_ratio)
    if outlier_count <= 0:
        return noisy
    minimum = noisy.min(axis=0)
    maximum = noisy.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-3)
    outliers = rng.uniform(minimum - 0.2 * span, maximum + 0.2 * span, size=(outlier_count, 3))
    return np.concatenate([noisy, outliers], axis=0)


def chamfer_distance_mm(source_points: np.ndarray, target_points: np.ndarray) -> float:
    if len(source_points) == 0 or len(target_points) == 0:
        return float("nan")
    source_tree = cKDTree(source_points)
    target_tree = cKDTree(target_points)
    source_distances, _ = source_tree.query(target_points, k=1)
    target_distances, _ = target_tree.query(source_points, k=1)
    chamfer = 0.5 * (np.mean(source_distances**2) + np.mean(target_distances**2))
    return float(np.sqrt(chamfer) * 1000.0)


def coverage_ratio(source_points: np.ndarray, target_points: np.ndarray, tolerance_mm: float) -> float:
    if len(source_points) == 0 or len(target_points) == 0:
        return 0.0
    tree = cKDTree(source_points)
    distances, _ = tree.query(target_points, k=1)
    return float(np.mean(distances <= tolerance_mm / 1000.0))


def nearest_neighbor_distances(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    if len(source_points) == 0 or len(target_points) == 0:
        return np.empty((0,), dtype=float)
    tree = cKDTree(target_points)
    distances, _ = tree.query(source_points, k=1)
    return distances


def bounds_extent(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.zeros(3, dtype=float)
    return points.max(axis=0) - points.min(axis=0)
