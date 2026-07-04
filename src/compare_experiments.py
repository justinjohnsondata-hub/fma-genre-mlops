"""compare_experiments.py — rank the logged runs by F1 and record the winner.

Ranking happens through MLflow's own `search_runs`, not a list we kept on the
side: the tracking store is the single source of truth for what ran and how it
scored. Runs are ordered by positive-class F1 (the primary metric), and the top
run is written to reports/winner.json so downstream steps can name it.

Run from repo root with the venv active (after run_sweep.py):
    python src/compare_experiments.py
    python src/compare_experiments.py --config configs/train.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, "src")
from config import load_config  # noqa: E402

DEFAULT_CONFIG = Path("configs/train.yaml")
WINNER_PATH = Path("reports/winner.json")

# Columns pulled for the ranked table, in display order.
SHOW = [
    ("run_name", "tags.mlflow.runName"),
    ("f1", "metrics.f1"),
    ("roc_auc", "metrics.roc_auc"),
    ("accuracy", "metrics.accuracy"),
    ("baseline_f1", "metrics.baseline_f1"),
    ("current_f1", "metrics.current_f1"),
]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Rank MLflow runs by F1 and save the winner.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    mlflow.set_tracking_uri(cfg.tracking_uri)

    runs = mlflow.search_runs(
        experiment_names=[cfg.experiment],
        order_by=["metrics.f1 DESC"],
    )
    if runs.empty:
        print("no runs found; run `python src/run_sweep.py` first", file=sys.stderr)
        return 1

    header = f"{'rank':<5}{'run':<24}" + "".join(f"{label:>12}" for label, _ in SHOW[1:])
    print(header)
    print("-" * len(header))
    for i, (_, row) in enumerate(runs.iterrows(), start=1):
        name = row.get("tags.mlflow.runName", "?")
        cells = "".join(f"{row.get(col, float('nan')):>12.3f}" for _, col in SHOW[1:])
        print(f"{i:<5}{name:<24}{cells}")

    top = runs.iloc[0]
    winner = {
        "run_id": top["run_id"],
        "run_name": top.get("tags.mlflow.runName", "?"),
        "model_uri": f"runs:/{top['run_id']}/model",
        "f1": float(top["metrics.f1"]),
        "roc_auc": float(top.get("metrics.roc_auc", float("nan"))),
        "current_f1": float(top.get("metrics.current_f1", float("nan"))),
    }
    WINNER_PATH.parent.mkdir(parents=True, exist_ok=True)
    WINNER_PATH.write_text(json.dumps(winner, indent=2) + "\n")
    print(f"\nwinner: {winner['run_name']} (F1 {winner['f1']:.3f}) -> {WINNER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
