from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _to_list(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_to_list(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_list(item) for key, item in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


@dataclass
class PrimitiveSpec:
    primitive_id: str
    type: str
    center: list[float]
    axis: list[float] | None = None
    normal: list[float] | None = None
    radius: float | None = None
    height: float | None = None
    offset: float | None = None
    dimensions: dict[str, float] = field(default_factory=dict)
    role: str = ""
    confidence: float | None = None
    support_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive_id": self.primitive_id,
            "type": self.type,
            "center": _to_list(self.center),
            "axis": _to_list(self.axis),
            "normal": _to_list(self.normal),
            "radius": self.radius,
            "height": self.height,
            "offset": self.offset,
            "dimensions": _to_list(self.dimensions),
            "role": self.role,
            "confidence": self.confidence,
            "support_size": self.support_size,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PrimitiveSpec":
        return cls(
            primitive_id=payload["primitive_id"],
            type=payload["type"],
            center=list(payload.get("center", [])),
            axis=list(payload["axis"]) if payload.get("axis") is not None else None,
            normal=list(payload["normal"]) if payload.get("normal") is not None else None,
            radius=payload.get("radius"),
            height=payload.get("height"),
            offset=payload.get("offset"),
            dimensions=dict(payload.get("dimensions", {})),
            role=payload.get("role", ""),
            confidence=payload.get("confidence"),
            support_size=payload.get("support_size"),
        )


@dataclass
class ScanRecord:
    part_id: str
    family: str
    scan_id: str
    split: str
    cloud_path: str
    mesh_path: str
    reference_cloud_path: str
    gt_pose: list[list[float]]
    primitive_gt: list[dict[str, Any]]
    noise_profile: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "part_id": self.part_id,
            "family": self.family,
            "scan_id": self.scan_id,
            "split": self.split,
            "cloud_path": self.cloud_path,
            "mesh_path": self.mesh_path,
            "reference_cloud_path": self.reference_cloud_path,
            "gt_pose": _to_list(self.gt_pose),
            "primitive_gt": _to_list(self.primitive_gt),
            "noise_profile": _to_list(self.noise_profile),
            "metadata": _to_list(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScanRecord":
        return cls(
            part_id=payload["part_id"],
            family=payload["family"],
            scan_id=payload["scan_id"],
            split=payload["split"],
            cloud_path=payload["cloud_path"],
            mesh_path=payload["mesh_path"],
            reference_cloud_path=payload["reference_cloud_path"],
            gt_pose=payload["gt_pose"],
            primitive_gt=payload["primitive_gt"],
            noise_profile=payload["noise_profile"],
            metadata=payload.get("metadata", {}),
        )


@dataclass
class ScanReport:
    scan_id: str
    family: str
    mode: str
    registration_success: bool
    rot_err_deg: float
    trans_err_mm: float
    chamfer_mm: float
    pre_refine_chamfer_mm: float
    chamfer_improvement_pct: float
    coverage: float
    primitive_f1: float
    primitive_precision: float
    primitive_recall: float
    dimension_mae_mm: float
    runtime_sec: float
    predicted_primitives: list[dict[str, Any]]
    reconstruction_mesh_path: str | None
    stage_path: str
    raw_rot_err_deg: float | None = None
    raw_trans_err_mm: float | None = None
    symmetry_mode: str | None = None
    selected_candidate: str | None = None
    refine_accepted: bool | None = None
    refine_reject_reason: str | None = None
    init_chamfer_mm: float | None = None
    candidate_scores: list[dict[str, Any]] = field(default_factory=list)
    transform_delta_deg: float | None = None
    transform_delta_mm: float | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scan_id": self.scan_id,
            "family": self.family,
            "mode": self.mode,
            "registration_success": self.registration_success,
            "rot_err_deg": self.rot_err_deg,
            "trans_err_mm": self.trans_err_mm,
            "raw_rot_err_deg": self.raw_rot_err_deg,
            "raw_trans_err_mm": self.raw_trans_err_mm,
            "symmetry_mode": self.symmetry_mode,
            "selected_candidate": self.selected_candidate,
            "refine_accepted": self.refine_accepted,
            "refine_reject_reason": self.refine_reject_reason,
            "init_chamfer_mm": self.init_chamfer_mm,
            "candidate_scores": _to_list(self.candidate_scores),
            "transform_delta_deg": self.transform_delta_deg,
            "transform_delta_mm": self.transform_delta_mm,
            "chamfer_mm": self.chamfer_mm,
            "pre_refine_chamfer_mm": self.pre_refine_chamfer_mm,
            "chamfer_improvement_pct": self.chamfer_improvement_pct,
            "coverage": self.coverage,
            "primitive_f1": self.primitive_f1,
            "primitive_precision": self.primitive_precision,
            "primitive_recall": self.primitive_recall,
            "dimension_mae_mm": self.dimension_mae_mm,
            "runtime_sec": self.runtime_sec,
            "predicted_primitives": _to_list(self.predicted_primitives),
            "reconstruction_mesh_path": self.reconstruction_mesh_path,
            "stage_path": self.stage_path,
            "notes": list(self.notes),
        }
