"""monitor_drift.py — Evidently data-drift check across all 150 features.

Compares the two popularity tiers built in prep_data.py: reference = the
more-played half of each genre, current = the less-played half. Because the split
is taken within each genre, the genre mix (and the Hip-Hop base rate) is held
constant across the two windows, so any drift Evidently reports is a clean
covariate shift — popular vs obscure tracks differing in spectral profile — not a
manufactured change in class balance.

What it does:
  - asserts each window still carries its ~500 Hip-Hop positives (the floor the
    whole drift comparison rests on),
  - runs Evidently's DataDriftPreset over all 148 numeric + 2 categorical features,
  - prints the overall drift share and the list of drifted features,
  - saves the full HTML report to reports/,
  - exits non-zero when the drift share exceeds a configurable threshold, so this
    can run as a monitoring gate.

Run from repo root with the venv active:
    python src/monitor_drift.py
    python src/monitor_drift.py --drift-share-threshold 0.3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset

sys.path.insert(0, "src")
import preprocess as pp  # noqa: E402
from config import load_config  # noqa: E402

DEFAULT_CONFIG = Path("configs/train.yaml")
DEFAULT_REPORT = Path("reports/drift_report.html")
DEFAULT_DRIFT_SHARE = 0.5   # fraction of features that must drift to trip the gate
HIPHOP_FLOOR = 450          # each window must still carry ~500 Hip-Hop positives


def build_dataset(df: pd.DataFrame, numeric: list[str], categorical: list[str]) -> Dataset:
    """Wrap a frame with an explicit column-type definition.

    Telling Evidently which columns are numeric vs categorical picks the right
    per-column drift test (a distance test for numeric, a categorical test for the
    two engineered columns) instead of letting it guess from dtype.
    """
    dd = DataDefinition(numerical_columns=numeric, categorical_columns=categorical)
    return Dataset.from_pandas(df, data_definition=dd)


def extract_drift(snapshot) -> tuple[float, int, int, list[tuple[str, float, float]]]:
    """Pull the drift share, counts, and the drifted-feature list from a snapshot.

    Evidently's DriftedColumnsCount gives the authoritative share/count; the
    per-column ValueDrift metrics give each feature's score and threshold. A column
    counts as drifted when its distance score exceeds its threshold, and the total
    is reconciled against Evidently's own count so this parsing can't silently skew.
    """
    d = snapshot.dict()
    share = count = total = 0
    drifted: list[tuple[str, float, float]] = []
    for m in d["metrics"]:
        cfg = m.get("config", {})
        kind = cfg.get("type", "")
        if kind.endswith("DriftedColumnsCount"):
            share = float(m["value"]["share"])
            count = int(m["value"]["count"])
        elif kind.endswith("ValueDrift"):
            total += 1
            score, threshold = float(m["value"]), float(cfg["threshold"])
            if score > threshold:
                drifted.append((cfg["column"], score, threshold))

    if len(drifted) != count:
        raise RuntimeError(
            f"drift extraction mismatch: parsed {len(drifted)} but Evidently reports {count}"
        )
    drifted.sort(key=lambda t: -t[1])
    return share, count, total, drifted


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run the Evidently data-drift check.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument(
        "--drift-share-threshold",
        type=float,
        default=DEFAULT_DRIFT_SHARE,
        help="exit non-zero when the drifted-feature share exceeds this",
    )
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    reference = pd.read_parquet(cfg.reference_path)
    current = pd.read_parquet(cfg.current_path)

    # The floor the whole comparison rests on: each window must still be ~500 Hip-Hop.
    for name, df in (("reference", reference), ("current", current)):
        pos = int(df[pp.TARGET].sum())
        assert pos >= HIPHOP_FLOOR, f"{name} Hip-Hop {pos} < floor {HIPHOP_FLOOR}"

    features = [c for c in reference.columns if c != pp.TARGET]
    numeric = pp.numeric_columns(reference)
    categorical = list(pp.CATEGORICAL_COLS)

    report = Report([DataDriftPreset(drift_share=args.drift_share_threshold)])
    snapshot = report.run(
        current_data=build_dataset(current[features], numeric, categorical),
        reference_data=build_dataset(reference[features], numeric, categorical),
    )

    args.report.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(args.report))

    share, count, total, drifted = extract_drift(snapshot)
    print(f"drift share: {share:.3f}  ({count}/{total} features drifted)")
    if drifted:
        print("drifted features (score > threshold):")
        for col, score, threshold in drifted:
            print(f"  {col:<28} {score:.3f} > {threshold:.2f}")
    else:
        print("no features drifted")
    print(f"report saved to {args.report}")

    if share > args.drift_share_threshold:
        print(
            f"DRIFT: share {share:.3f} > threshold {args.drift_share_threshold:.3f}",
            file=sys.stderr,
        )
        return 1
    print(f"OK: share {share:.3f} <= threshold {args.drift_share_threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
