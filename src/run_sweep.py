"""run_sweep.py — run the five training runs declared in configs/sweep.yaml.

The sweep varies one axis (the model) and holds everything else fixed by reusing
the base config's data paths, seed, threshold, and MLflow target. Each run is
logged as its own MLflow run; compare_experiments.py ranks them afterward.

Run from repo root with the venv active:
    python src/run_sweep.py                        # uses configs/sweep.yaml
    python src/run_sweep.py --sweep configs/sweep.yaml
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from config import load_config  # noqa: E402
from train import train_and_log  # noqa: E402

DEFAULT_SWEEP = Path("configs/sweep.yaml")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the model sweep declared in a sweep YAML.")
    p.add_argument("--sweep", type=Path, default=DEFAULT_SWEEP)
    args = p.parse_args(argv)

    spec = yaml.safe_load(args.sweep.read_text())
    base = load_config(spec["base"])

    runs = spec["runs"]
    print(f"running {len(runs)} runs from {args.sweep} (base: {spec['base']})\n")
    for entry in runs:
        # Clone the base config, swapping in this run's model and hyperparameters.
        cfg = replace(base, model_type=entry["type"], model_params=dict(entry["params"]))
        train_and_log(cfg, run_name=entry["name"])

    print("\nsweep complete; rank with: python src/compare_experiments.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
