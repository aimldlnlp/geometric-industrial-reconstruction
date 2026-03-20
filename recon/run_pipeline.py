from __future__ import annotations

import argparse

from recon.config import load_config
from recon.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the industrial 3D reconstruction pipeline.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to process.")
    parser.add_argument("--limit", type=int, default=None, help="Optional number of scans to process.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional dataset directory override.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional run directory name.")
    parser.add_argument("--ablation-suite", action="store_true", help="Run the full ablation suite.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    aggregate = run_pipeline(
        config,
        split=args.split,
        limit=args.limit,
        dataset_root=args.dataset_root,
        run_name=args.run_name,
        ablation_suite=args.ablation_suite,
    )
    print(f"Completed run {aggregate['run_name']} for split={aggregate['split']}.")


if __name__ == "__main__":
    main()
