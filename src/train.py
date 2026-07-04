"""train.py — train one model, score it, log everything to MLflow.

One invocation = one config = one tracked run. The config (train.yaml) supplies
the model, its hyperparameters, the data paths, the seed, and the F1 threshold;
nothing that defines the run is hardcoded here. That is what makes a run
reproducible from an artifact you can commit and diff.

What each run records, and why:
  - params: model type + hyperparameters + seed, so a run is fully specified.
  - data version: the config's human label AND the DVC md5 of the prepared data,
    so a logged metric is always tied to the exact bytes it was computed on.
  - the majority baseline, logged first, as the floor the model must clear.
  - metrics: positive-class F1 (primary), ROC-AUC, accuracy, plus the transfer
    numbers (reference-test F1 vs held-out current F1).
  - the fitted pipeline as a model artifact.

The preprocessor is fit inside the pipeline on the training split only, so no
test or current-window row ever touches a fitted parameter.

Run from repo root with the venv active:
    python src/train.py                          # uses configs/train.yaml
    python src/train.py --config configs/train.yaml

Exit code: non-zero when reference-test F1 falls below the config threshold, so
CI can gate the train job on real model quality.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

sys.path.insert(0, "src")
import evaluate  # noqa: E402
import preprocess as pp  # noqa: E402
from config import Config, load_config  # noqa: E402

DEFAULT_CONFIG = Path("configs/train.yaml")

# Config's model.type string -> estimator class. Adding a model means adding a row
# here and a config; train.py itself does not change.
ESTIMATORS = {
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
    "gradient_boosting": GradientBoostingClassifier,
}


def build_estimator(model_type: str, params: dict[str, Any]):
    """Instantiate the configured estimator with its config hyperparameters."""
    try:
        cls = ESTIMATORS[model_type]
    except KeyError as e:
        raise ValueError(
            f"unknown model type {model_type!r}; expected one of {sorted(ESTIMATORS)}"
        ) from e
    return cls(**params)


def build_pipeline(numeric_cols: list[str], estimator) -> Pipeline:
    """Preprocessor + estimator as one fittable object.

    Bundling them means `.fit` fits the imputer/scaler/encoder on the training
    rows only, and every later `.predict` reuses those frozen parameters.
    """
    pre = pp.build_preprocessor(numeric_cols)
    return Pipeline([("preprocess", pre), ("model", estimator)])


def dvc_md5(pointer: Path = Path("data/prepared.dvc")) -> str:
    """Read the DVC content hash of the prepared data, or 'unknown' if absent.

    Logging this alongside the human label ties every metric to the exact data
    bytes; the label is for humans, the md5 is the machine-checkable version.
    """
    if not pointer.exists():
        return "unknown"
    doc = yaml.safe_load(pointer.read_text())
    return str(doc["outs"][0]["md5"])


def train_and_log(cfg: Config, run_name: str | None = None) -> dict[str, float]:
    """Run one experiment end to end and log it. Returns the run's key metrics.

    Kept separate from main() so the sweep script can call it in a loop without
    the process-exit gate. run_name labels the MLflow run (defaults to the model
    type) so a sweep of several RandomForest settings stays distinguishable.
    """
    reference = pd.read_parquet(cfg.reference_path)
    current = pd.read_parquet(cfg.current_path)

    X_train, X_test, y_train, y_test = pp.split_reference(reference, seed=cfg.seed)
    numeric_cols = pp.numeric_columns(reference)

    X_cur = current.drop(columns=[pp.TARGET])
    y_cur = current[pp.TARGET]

    estimator = build_estimator(cfg.model_type, cfg.model_params)
    pipeline = build_pipeline(numeric_cols, estimator)

    mlflow.set_tracking_uri(cfg.tracking_uri)
    mlflow.set_experiment(cfg.experiment)

    with mlflow.start_run(run_name=run_name or cfg.model_type) as run:
        mlflow.log_params(
            {
                "model_type": cfg.model_type,
                "seed": cfg.seed,
                "n_numeric_features": len(numeric_cols),
                "n_categorical_features": len(pp.CATEGORICAL_COLS),
                **{f"model__{k}": v for k, v in cfg.model_params.items()},
            }
        )
        mlflow.set_tags(
            {"data_version": cfg.data_version, "dvc_md5": dvc_md5()}
        )

        # Baseline first: the floor every real run is measured against.
        baseline = evaluate.majority_baseline_metrics(y_test)
        mlflow.log_metric("baseline_f1", baseline["f1"])
        mlflow.log_metric("baseline_accuracy", baseline["accuracy"])

        pipeline.fit(X_train, y_train)

        test_metrics = evaluate.evaluate_model(pipeline, X_test, y_test)
        transfer = evaluate.transfer_report(pipeline, X_test, y_test, X_cur, y_cur)

        mlflow.log_metric("f1", test_metrics["f1"])
        mlflow.log_metric("accuracy", test_metrics["accuracy"])
        if "roc_auc" in test_metrics:
            mlflow.log_metric("roc_auc", test_metrics["roc_auc"])
        for name in ("tp", "fp", "fn", "tn"):
            mlflow.log_metric(name, test_metrics[name])
        mlflow.log_metric("current_f1", transfer["current_f1"])
        mlflow.log_metric("transfer_f1_drop", transfer["transfer_f1_drop"])

        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        result = {
            "run_id": run.info.run_id,
            "f1": test_metrics["f1"],
            "accuracy": test_metrics["accuracy"],
            "roc_auc": test_metrics.get("roc_auc", float("nan")),
            "baseline_f1": baseline["f1"],
            "current_f1": transfer["current_f1"],
            "transfer_f1_drop": transfer["transfer_f1_drop"],
        }

    print(
        f"[{cfg.model_type}] run {result['run_id'][:8]} | "
        f"F1 {result['f1']:.3f} (baseline {result['baseline_f1']:.3f}) | "
        f"ROC-AUC {result['roc_auc']:.3f} | acc {result['accuracy']:.3f} | "
        f"current F1 {result['current_f1']:.3f} (drop {result['transfer_f1_drop']:+.3f})"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train the FMA Hip-Hop detector and log to MLflow.")
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    result = train_and_log(cfg)

    if result["f1"] < cfg.f1_threshold:
        print(
            f"FAIL: F1 {result['f1']:.3f} < threshold {cfg.f1_threshold:.3f}",
            file=sys.stderr,
        )
        return 1
    print(f"PASS: F1 {result['f1']:.3f} >= threshold {cfg.f1_threshold:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
