from __future__ import annotations

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from recon.config import project_root, resolve_from_root
from recon.geometry import (
    add_noise_and_outliers,
    apply_transform,
    invert_transform,
    jitter_transform,
    random_pose,
    sample_mesh_surface,
    sample_reference_cloud,
    seed_everything,
    simulate_multiview_visibility,
)
from recon.io_utils import ensure_dir, write_json
from recon.types import PrimitiveSpec, ScanRecord


FAMILIES = ("flange", "shaft", "bracket", "pipe_elbow", "plate_with_holes")


def _translated(mesh: trimesh.Trimesh, xyz: tuple[float, float, float]) -> trimesh.Trimesh:
    translated = mesh.copy()
    translated.apply_translation(np.asarray(xyz, dtype=float))
    return translated


def _concat(meshes: list[trimesh.Trimesh]) -> trimesh.Trimesh:
    mesh = trimesh.util.concatenate(meshes)
    mesh.remove_unreferenced_vertices()
    mesh.process(validate=True)
    return mesh


def _primitive_to_dicts(primitives: list[PrimitiveSpec]) -> list[dict]:
    return [primitive.to_dict() for primitive in primitives]


def build_flange(rng: np.random.Generator) -> tuple[trimesh.Trimesh, list[PrimitiveSpec], dict]:
    inner_r = float(rng.uniform(0.018, 0.024))
    outer_r = float(rng.uniform(0.048, 0.062))
    disk_h = float(rng.uniform(0.012, 0.018))
    hub_r = float(rng.uniform(0.017, 0.022))
    hub_h = float(rng.uniform(0.055, 0.075))
    annulus = trimesh.creation.annulus(r_min=inner_r, r_max=outer_r, height=disk_h, sections=64)
    hub = trimesh.creation.cylinder(radius=hub_r, height=hub_h, sections=64)
    mesh = _concat([annulus, hub])
    primitives = [
        PrimitiveSpec(
            primitive_id="cyl_hub",
            type="cylinder",
            center=[0.0, 0.0, 0.0],
            axis=[0.0, 0.0, 1.0],
            radius=hub_r,
            height=hub_h,
            dimensions={"diameter_m": hub_r * 2.0, "height_m": hub_h},
            role="main_shaft",
        ),
        PrimitiveSpec(
            primitive_id="plane_top",
            type="plane",
            center=[0.0, 0.0, disk_h * 0.5],
            normal=[0.0, 0.0, 1.0],
            offset=-disk_h * 0.5,
            dimensions={"size_u_m": outer_r * 2.0, "size_v_m": outer_r * 2.0},
            role="mount_face",
        ),
        PrimitiveSpec(
            primitive_id="plane_bottom",
            type="plane",
            center=[0.0, 0.0, -disk_h * 0.5],
            normal=[0.0, 0.0, 1.0],
            offset=disk_h * 0.5,
            dimensions={"size_u_m": outer_r * 2.0, "size_v_m": outer_r * 2.0},
            role="back_face",
        ),
    ]
    return mesh, primitives, {"outer_radius_m": outer_r, "inner_radius_m": inner_r, "disk_height_m": disk_h}


def build_shaft(rng: np.random.Generator) -> tuple[trimesh.Trimesh, list[PrimitiveSpec], dict]:
    radii = [
        float(rng.uniform(0.012, 0.018)),
        float(rng.uniform(0.018, 0.024)),
        float(rng.uniform(0.01, 0.016)),
    ]
    heights = [
        float(rng.uniform(0.028, 0.04)),
        float(rng.uniform(0.04, 0.055)),
        float(rng.uniform(0.022, 0.034)),
    ]
    z_centers = np.cumsum([heights[0] * 0.5, heights[0] * 0.5 + heights[1] * 0.5, heights[1] * 0.5 + heights[2] * 0.5])
    z_centers = z_centers - np.mean([0.0, z_centers[-1]])
    sections = 64
    cylinders = [
        _translated(trimesh.creation.cylinder(radius=radii[0], height=heights[0], sections=sections), (0.0, 0.0, z_centers[0])),
        _translated(trimesh.creation.cylinder(radius=radii[1], height=heights[1], sections=sections), (0.0, 0.0, z_centers[1])),
        _translated(trimesh.creation.cylinder(radius=radii[2], height=heights[2], sections=sections), (0.0, 0.0, z_centers[2])),
    ]
    mesh = _concat(cylinders)
    primitives = []
    for idx, (radius, height, center_z) in enumerate(zip(radii, heights, z_centers, strict=False)):
        primitives.append(
            PrimitiveSpec(
                primitive_id=f"cyl_stage_{idx}",
                type="cylinder",
                center=[0.0, 0.0, float(center_z)],
                axis=[0.0, 0.0, 1.0],
                radius=radius,
                height=height,
                dimensions={"diameter_m": radius * 2.0, "height_m": height},
                role="shaft_segment",
            )
        )
    return mesh, primitives, {"radii_m": radii, "heights_m": heights}


def build_bracket(rng: np.random.Generator) -> tuple[trimesh.Trimesh, list[PrimitiveSpec], dict]:
    base_size = np.asarray([rng.uniform(0.08, 0.11), rng.uniform(0.05, 0.07), rng.uniform(0.012, 0.018)])
    web_size = np.asarray([rng.uniform(0.016, 0.024), rng.uniform(0.05, 0.07), rng.uniform(0.06, 0.085)])
    boss_radius = float(rng.uniform(0.01, 0.015))
    boss_height = float(rng.uniform(0.02, 0.03))
    base = trimesh.creation.box(extents=base_size)
    web = _translated(
        trimesh.creation.box(extents=web_size),
        (base_size[0] * 0.35, 0.0, base_size[2] * 0.5 + web_size[2] * 0.5),
    )
    boss = _translated(
        trimesh.creation.cylinder(radius=boss_radius, height=boss_height, sections=48),
        (-base_size[0] * 0.2, 0.0, base_size[2] * 0.5 + boss_height * 0.5),
    )
    mesh = _concat([base, web, boss])
    primitives = [
        PrimitiveSpec(
            primitive_id="plane_base_top",
            type="plane",
            center=[0.0, 0.0, float(base_size[2] * 0.5)],
            normal=[0.0, 0.0, 1.0],
            offset=-float(base_size[2] * 0.5),
            dimensions={"size_u_m": float(base_size[0]), "size_v_m": float(base_size[1])},
            role="support_face",
        ),
        PrimitiveSpec(
            primitive_id="plane_web",
            type="plane",
            center=[float(base_size[0] * 0.35), 0.0, float(base_size[2] * 0.5 + web_size[2] * 0.5)],
            normal=[1.0, 0.0, 0.0],
            offset=-float(base_size[0] * 0.35),
            dimensions={"size_u_m": float(web_size[1]), "size_v_m": float(web_size[2])},
            role="mount_face",
        ),
        PrimitiveSpec(
            primitive_id="cyl_boss",
            type="cylinder",
            center=[-float(base_size[0] * 0.2), 0.0, float(base_size[2] * 0.5 + boss_height * 0.5)],
            axis=[0.0, 0.0, 1.0],
            radius=boss_radius,
            height=boss_height,
            dimensions={"diameter_m": boss_radius * 2.0, "height_m": boss_height},
            role="locating_boss",
        ),
    ]
    return mesh, primitives, {"base_size_m": base_size.tolist(), "web_size_m": web_size.tolist()}


def build_pipe_elbow(rng: np.random.Generator) -> tuple[trimesh.Trimesh, list[PrimitiveSpec], dict]:
    radius = float(rng.uniform(0.014, 0.02))
    leg_a = float(rng.uniform(0.08, 0.11))
    leg_b = float(rng.uniform(0.08, 0.11))
    cyl_x = trimesh.creation.cylinder(radius=radius, height=leg_a, sections=64)
    cyl_x.apply_transform(trimesh.transformations.rotation_matrix(np.pi * 0.5, [0, 1, 0]))
    cyl_x.apply_translation((leg_a * 0.25, 0.0, 0.0))
    cyl_z = trimesh.creation.cylinder(radius=radius, height=leg_b, sections=64)
    cyl_z.apply_translation((0.0, 0.0, leg_b * 0.25))
    corner = trimesh.creation.icosphere(subdivisions=2, radius=radius * 1.15)
    mesh = _concat([cyl_x, cyl_z, corner])
    primitives = [
        PrimitiveSpec(
            primitive_id="cyl_x",
            type="cylinder",
            center=[leg_a * 0.25, 0.0, 0.0],
            axis=[1.0, 0.0, 0.0],
            radius=radius,
            height=leg_a,
            dimensions={"diameter_m": radius * 2.0, "height_m": leg_a},
            role="pipe_leg",
        ),
        PrimitiveSpec(
            primitive_id="cyl_z",
            type="cylinder",
            center=[0.0, 0.0, leg_b * 0.25],
            axis=[0.0, 0.0, 1.0],
            radius=radius,
            height=leg_b,
            dimensions={"diameter_m": radius * 2.0, "height_m": leg_b},
            role="pipe_leg",
        ),
        PrimitiveSpec(
            primitive_id="sphere_corner",
            type="sphere",
            center=[0.0, 0.0, 0.0],
            radius=radius * 1.15,
            dimensions={"diameter_m": radius * 2.3},
            role="blend_region",
        ),
    ]
    return mesh, primitives, {"pipe_radius_m": radius, "leg_a_m": leg_a, "leg_b_m": leg_b}


def build_plate_with_holes(rng: np.random.Generator) -> tuple[trimesh.Trimesh, list[PrimitiveSpec], dict]:
    width = float(rng.uniform(0.09, 0.12))
    depth = float(rng.uniform(0.06, 0.08))
    thickness = float(rng.uniform(0.01, 0.015))
    hole_radius = float(rng.uniform(0.006, 0.01))
    offset_x = width * 0.28
    offset_y = depth * 0.28
    hole_centers = [
        (-offset_x, -offset_y),
        (-offset_x, offset_y),
        (offset_x, -offset_y),
        (offset_x, offset_y),
    ]
    plate = trimesh.creation.box(extents=(width, depth, thickness))
    mesh = _concat([plate])
    primitives = [
        PrimitiveSpec(
            primitive_id="plane_top",
            type="plane",
            center=[0.0, 0.0, thickness * 0.5],
            normal=[0.0, 0.0, 1.0],
            offset=-thickness * 0.5,
            dimensions={"size_u_m": width, "size_v_m": depth},
            role="inspection_face",
        ),
        PrimitiveSpec(
            primitive_id="plane_bottom",
            type="plane",
            center=[0.0, 0.0, -thickness * 0.5],
            normal=[0.0, 0.0, 1.0],
            offset=thickness * 0.5,
            dimensions={"size_u_m": width, "size_v_m": depth},
            role="inspection_face",
        ),
    ]
    metadata = {
        "plate_size_m": [width, depth, thickness],
        "hole_radius_m": hole_radius,
        "hole_centers_xy_m": hole_centers,
    }
    return mesh, primitives, metadata


BUILDERS = {
    "flange": build_flange,
    "shaft": build_shaft,
    "bracket": build_bracket,
    "pipe_elbow": build_pipe_elbow,
    "plate_with_holes": build_plate_with_holes,
}


def create_part(family: str, rng: np.random.Generator) -> tuple[trimesh.Trimesh, list[PrimitiveSpec], dict]:
    if family not in BUILDERS:
        raise KeyError(f"Unsupported family: {family}")
    mesh, primitives, metadata = BUILDERS[family](rng)
    center_shift = np.asarray(mesh.centroid, dtype=float)
    mesh.apply_translation(-center_shift)
    adjusted = []
    for primitive in primitives:
        primitive = PrimitiveSpec.from_dict(primitive.to_dict())
        primitive.center = (np.asarray(primitive.center) - center_shift).tolist()
        adjusted.append(primitive)
    return mesh, adjusted, metadata


def _apply_plate_hole_mask(points: np.ndarray, metadata: dict) -> np.ndarray:
    if "hole_centers_xy_m" not in metadata or len(points) == 0:
        return points
    thickness = float(metadata["plate_size_m"][2])
    hole_radius = float(metadata["hole_radius_m"])
    top_bottom = np.abs(np.abs(points[:, 2]) - thickness * 0.5) < thickness * 0.35
    keep = np.ones(len(points), dtype=bool)
    for center_x, center_y in metadata["hole_centers_xy_m"]:
        radial = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)
        keep &= ~(top_bottom & (radial < hole_radius))
    return points[keep]


def _apply_view_jitter(points: np.ndarray, rng: np.random.Generator, deg: float, mm: float) -> np.ndarray:
    if len(points) == 0:
        return points
    groups = np.array_split(rng.permutation(len(points)), 3)
    jittered = points.copy()
    for group in groups:
        if len(group) == 0:
            continue
        transform = jitter_transform(rng, max_rotation_deg=deg, max_translation_m=mm / 1000.0)
        jittered[group] = apply_transform(jittered[group], transform)
    return jittered


def simulate_scan(
    mesh: trimesh.Trimesh,
    family: str,
    metadata: dict,
    rng: np.random.Generator,
    dataset_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, dict]:
    raw_points, normals = sample_mesh_surface(mesh, int(dataset_cfg["scan_points"] * 3), rng)
    visibility_idx, visibility_stats = simulate_multiview_visibility(
        raw_points,
        normals,
        rng,
        min_views=int(dataset_cfg["views_per_scan"]["min"]),
        max_views=int(dataset_cfg["views_per_scan"]["max"]),
        visibility_cosine=float(dataset_cfg["visibility_cosine"]),
        dropout_range=(
            float(dataset_cfg["occlusion_dropout"]["min"]),
            float(dataset_cfg["occlusion_dropout"]["max"]),
        ),
    )
    points = raw_points[visibility_idx]
    if family == "plate_with_holes":
        points = _apply_plate_hole_mask(points, metadata)
    if len(points) > int(dataset_cfg["scan_points"]):
        sample_idx = rng.choice(len(points), size=int(dataset_cfg["scan_points"]), replace=False)
        points = points[sample_idx]
    points = _apply_view_jitter(points, rng, deg=float(dataset_cfg["view_jitter_deg"]), mm=float(dataset_cfg["view_jitter_mm"]))
    ref_to_scan = random_pose(
        rng,
        max_rotation_deg=float(dataset_cfg["pose_rotation_deg"]),
        max_translation_m=float(dataset_cfg["pose_translation_m"]),
    )
    gt_pose = invert_transform(ref_to_scan)
    points = apply_transform(points, ref_to_scan)
    noise_std_m = float(rng.uniform(dataset_cfg["noise_std_range_mm"]["min"], dataset_cfg["noise_std_range_mm"]["max"])) / 1000.0
    outlier_ratio = float(rng.uniform(dataset_cfg["outlier_ratio_range"]["min"], dataset_cfg["outlier_ratio_range"]["max"]))
    points = add_noise_and_outliers(points, rng, noise_std_m=noise_std_m, outlier_ratio=outlier_ratio)
    noise_profile = {
        "noise_std_mm": noise_std_m * 1000.0,
        "outlier_ratio": outlier_ratio,
        "views_used": int(visibility_stats["views_used"]),
        "occlusion_dropout": visibility_stats["occlusion_dropout"],
        "view_jitter_deg": float(dataset_cfg["view_jitter_deg"]),
        "view_jitter_mm": float(dataset_cfg["view_jitter_mm"]),
    }
    return points, gt_pose, noise_profile


def _write_point_cloud(path: Path, points: np.ndarray) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points.astype(float))
    o3d.io.write_point_cloud(str(path), cloud, write_ascii=False)


def _assign_splits(part_ids: list[str], split_ratios: dict, rng: np.random.Generator) -> dict[str, str]:
    shuffled = list(part_ids)
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_count = max(1, int(round(total * float(split_ratios["train"]))))
    val_count = max(1, int(round(total * float(split_ratios["val"])))) if total >= 3 else 0
    if train_count + val_count >= total:
        val_count = max(0, total - train_count - 1)
    assignments: dict[str, str] = {}
    for index, part_id in enumerate(shuffled):
        if index < train_count:
            assignments[part_id] = "train"
        elif index < train_count + val_count:
            assignments[part_id] = "val"
        else:
            assignments[part_id] = "test"
    return assignments


def generate_dataset(config: dict, output_root: str | None = None) -> dict:
    dataset_cfg = config["dataset"]
    dataset_root = resolve_from_root(output_root or dataset_cfg["root"])
    ensure_dir(dataset_root)
    meshes_dir = ensure_dir(dataset_root / "meshes")
    reference_dir = ensure_dir(dataset_root / "reference_clouds")
    scans_dir = ensure_dir(dataset_root / "scans")
    seed = int(config["seed"])
    rng = seed_everything(seed)

    scan_records: list[ScanRecord] = []
    part_catalog: dict[str, dict] = {}
    split_lookup: dict[str, str] = {}
    for family in dataset_cfg["families"]:
        part_ids = [f"{family}_part_{index:03d}" for index in range(int(dataset_cfg["parts_per_family"]))]
        split_lookup.update(_assign_splits(part_ids, dataset_cfg["split_ratios"], rng))

    for family in dataset_cfg["families"]:
        for part_index in range(int(dataset_cfg["parts_per_family"])):
            part_id = f"{family}_part_{part_index:03d}"
            split = split_lookup[part_id]
            mesh, primitive_gt, metadata = create_part(family, rng)
            mesh_path = meshes_dir / f"{part_id}.ply"
            mesh.export(mesh_path)
            reference_points = sample_reference_cloud(mesh, int(dataset_cfg["reference_points"]), rng)
            reference_cloud_path = reference_dir / f"{part_id}.ply"
            _write_point_cloud(reference_cloud_path, reference_points)
            part_catalog[part_id] = {
                "family": family,
                "mesh_path": str(mesh_path.relative_to(project_root()).as_posix()),
                "reference_cloud_path": str(reference_cloud_path.relative_to(project_root()).as_posix()),
                "primitive_gt": _primitive_to_dicts(primitive_gt),
                "metadata": metadata,
            }
            split_dir = ensure_dir(scans_dir / split)
            for scan_index in range(int(dataset_cfg["scans_per_part"])):
                scan_id = f"{part_id}_scan_{scan_index:03d}"
                scan_points, gt_pose, noise_profile = simulate_scan(mesh, family, metadata, rng, dataset_cfg)
                cloud_path = split_dir / f"{scan_id}.ply"
                _write_point_cloud(cloud_path, scan_points)
                scan_records.append(
                    ScanRecord(
                        part_id=part_id,
                        family=family,
                        scan_id=scan_id,
                        split=split,
                        cloud_path=str(cloud_path.relative_to(project_root()).as_posix()),
                        mesh_path=str(mesh_path.relative_to(project_root()).as_posix()),
                        reference_cloud_path=str(reference_cloud_path.relative_to(project_root()).as_posix()),
                        gt_pose=np.asarray(gt_pose).tolist(),
                        primitive_gt=_primitive_to_dicts(primitive_gt),
                        noise_profile=noise_profile,
                        metadata=metadata,
                    )
                )

    manifest = {
        "dataset_name": "industrial_scan_benchmark",
        "seed": seed,
        "root": str(dataset_root.relative_to(project_root()).as_posix()),
        "records": [record.to_dict() for record in scan_records],
        "part_catalog": part_catalog,
    }
    write_json(dataset_root / "manifest.json", manifest)
    write_json(
        dataset_root / "summary.json",
        {
            "total_scans": len(scan_records),
            "families": list(dataset_cfg["families"]),
            "parts_per_family": int(dataset_cfg["parts_per_family"]),
            "scans_per_part": int(dataset_cfg["scans_per_part"]),
            "splits": {split: sum(record.split == split for record in scan_records) for split in ("train", "val", "test")},
        },
    )
    return manifest
