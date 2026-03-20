from __future__ import annotations

import argparse
from pathlib import Path

from recon.config import load_config
from recon.benchmark_parts import generate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a controlled industrial scan benchmark.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--output-root", type=str, default=None, help="Optional dataset output directory override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {"seed": args.seed} if args.seed is not None else None
    config = load_config(args.config, overrides=overrides)
    manifest = generate_dataset(config, output_root=args.output_root)
    total_scans = len(manifest["records"])
    dataset_root = manifest["root"]
    print(f"Generated industrial scan benchmark at {dataset_root} with {total_scans} scans.")


if __name__ == "__main__":
    main()
