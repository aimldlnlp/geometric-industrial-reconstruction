# Semi-Automatic 3D Reconstruction for Industrial Parts

This repository is a Python-only, CPU-only portfolio project for semi-automatic 3D reconstruction from partial point clouds. It is designed to look and feel like a compact internal benchmarking tool plus a paper-style experimental artifact: reproducible benchmark generation, measurable geometry recovery, and publication-oriented visual exports.

The project builds a `controlled industrial scan benchmark`, simulates multi-view partial observations, aligns each scan back to a clean reference, recovers simple primitives, and exports both scalar metrics and polished figures/videos.

## What the project demonstrates

- Industrial-style 3D data generation without external datasets
- Point-cloud preprocessing, registration, and refinement on CPU
- Primitive discovery with explicit geometric residuals
- Evaluation with reconstruction and inspection-oriented metrics
- Clean portfolio outputs: `5 PNG` figures and `5 MP4 + 5 GIF` scene videos

## Technical scope

- Language: Python
- Runtime target: Windows + VS Code + CPU only
- Core libraries: `numpy`, `scipy`, `open3d`, `trimesh`, `matplotlib`, `imageio`, `imageio-ffmpeg`, `PyYAML`
- Optional rendering stack: `pyvista` as an extra, not required for the default workflow

## Dataset design

The benchmark is framed as a controlled acquisition simulator rather than an ad-hoc toy dataset.

Supported families:

- `flange`
- `shaft`
- `bracket`
- `pipe_elbow`
- `plate_with_holes`

For each part family, the generator creates:

- A reference mesh
- A clean reference point cloud
- One or more partial scan point clouds
- Ground-truth scan-to-reference pose
- Primitive annotations for downstream evaluation
- Noise metadata including occlusion, outliers, and small view jitter

## Pipeline summary

1. Generate industrial reference geometries and controlled scan captures.
2. Preprocess point clouds with voxelization, outlier filtering, and normal estimation.
3. Run global alignment with FPFH + RANSAC.
4. Refine alignment with point-to-plane ICP and multi-start coarse initialization.
5. Score pose with symmetry-aware evaluation for rotationally ambiguous part families.
6. Recover planes, cylinders, and spheres with reference-conditioned geometric priors.
7. Export per-scan reports, aggregate summaries, publication-style figures, and paired MP4/GIF videos.

## Metrics

Per scan, the pipeline exports:

- `registration_success`
- `rot_err_deg`
- `trans_err_mm`
- `chamfer_mm`
- `pre_refine_chamfer_mm`
- `chamfer_improvement_pct`
- `coverage`
- `primitive_f1`
- `primitive_precision`
- `primitive_recall`
- `dimension_mae_mm`
- `runtime_sec`

## Repository layout

```text
configs/default.yaml
recon/generate_dataset.py
recon/run_pipeline.py
recon/make_figures.py
recon/make_videos.py
recon/benchmark_parts.py
recon/pipeline.py
recon/primitives.py
tests/
```

## Setup

Create and activate a virtual environment, then install the package in editable mode:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

If you want the optional PyVista-based stack available for future experiments:

```powershell
python -m pip install -e .[dev,render3d]
```

## Commands

Generate the benchmark dataset:

```powershell
python -m recon.generate_dataset
```

Run the full pipeline on the test split:

```powershell
python -m recon.run_pipeline --split test
```

Run the ablation suite:

```powershell
python -m recon.run_pipeline --split test --ablation-suite
```

Export the five paper-style figures:

```powershell
python -m recon.make_figures --run-dir artifacts/runs/<run_name>
```

Export the five MP4/GIF scene pairs:

```powershell
python -m recon.make_videos --run-dir artifacts/runs/<run_name>
```

## Expected outputs

Dataset artifacts:

- `data/industrial_scan_benchmark/manifest.json`
- `data/industrial_scan_benchmark/summary.json`
- per-part meshes and reference clouds
- per-scan partial point clouds

Run artifacts:

- `artifacts/runs/<run_name>/full/summary.json`
- `artifacts/runs/<run_name>/full/summary.csv`
- `artifacts/runs/<run_name>/full/scans/<scan_id>/report.json`
- `artifacts/runs/<run_name>/full/scans/<scan_id>/stages.npz`

Visual artifacts:

- `artifacts/figures/<run_name>/figure_01_pipeline_overview.png`
- `artifacts/figures/<run_name>/figure_02_dataset_diversity.png`
- `artifacts/figures/<run_name>/figure_03_registration_before_after.png`
- `artifacts/figures/<run_name>/figure_04_primitive_tolerance.png`
- `artifacts/figures/<run_name>/figure_05_error_ablation_failure_atlas.png`
- `artifacts/videos/<run_name>/video_01_*.mp4`
- `artifacts/videos/<run_name>/video_01_*.gif`
- ...

## Portfolio framing

Good positioning for a portfolio or application:

- Describe the data as a `controlled industrial scan benchmark with explicit acquisition assumptions`.
- Emphasize geometry, robustness, metrics, and inspection-style outputs.
- Show both the benchmark summary and the rendered media artifacts.
- Mention that the project is intentionally CPU-friendly and reproducible on a local Windows workstation.

## Notes

- The default renderer uses `matplotlib + imageio` to keep the workflow Python-only and reliable on CPU.
- The benchmark is procedurally generated, but it should be presented as a controlled evaluation environment with explicit acquisition assumptions.
- I did not run dataset generation, pipeline execution, or smoke tests in this environment, so the intended workflow is for you to run those locally from the terminal.
