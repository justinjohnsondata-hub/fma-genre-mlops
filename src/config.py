"""config.py — read and validate the training config.

Why a config file at all: the moment a threshold, a file path, or a
hyperparameter is hardcoded in train.py, a "run" is no longer described by an
artifact you can commit and diff. The config IS the experiment's declaration:
change the YAML, get a different, reproducible run. So the config carries
everything that defines the run — model + hyperparameters, the feature set/seed,
the file paths, and the F1 performance threshold — not just the hyperparameters.

This module only reads and sanity-checks. It builds no estimators and touches no
data; train.py consumes the returned object. Keeping the reader dumb means a
malformed config fails here with a clear message, not three layers deep in sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Typed view of train.yaml. Flat-ish on purpose so train.py reads cfg.model_type
    rather than digging through nested dicts at every call site."""

    seed: int
    # data
    reference_path: Path
    current_path: Path
    data_version: str          # human label, e.g. fma_small_seed17_prep_v1
    # model
    model_type: str            # random_forest | logistic_regression | gradient_boosting
    model_params: dict[str, Any]
    # evaluation gate
    f1_threshold: float        # positive-class F1 floor; the CI train job gates on it
    # mlflow
    experiment: str
    tracking_uri: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)  # full dict, for logging


def load_config(path: str | Path) -> Config:
    """Load train.yaml into a validated Config.

    Fails loudly on a missing required key (a silent default here would let a run
    log against, say, the wrong data path and look fine). model_params is passed
    through verbatim — train.py hands it straight to the estimator constructor, so
    the config author controls hyperparameters without a code change.
    """
    path = Path(path)
    with path.open() as f:
        cfg = yaml.safe_load(f)

    try:
        data = cfg["data"]
        model = cfg["model"]
        return Config(
            seed=int(cfg["seed"]),
            reference_path=Path(data["reference"]),
            current_path=Path(data["current"]),
            data_version=str(data["version"]),
            model_type=str(model["type"]),
            model_params=dict(model.get("params", {})),
            f1_threshold=float(cfg["eval"]["f1_threshold"]),
            experiment=str(cfg["mlflow"]["experiment"]),
            tracking_uri=str(cfg["mlflow"]["tracking_uri"]),
            raw=cfg,
        )
    except KeyError as e:
        raise ValueError(f"config {path} is missing required key: {e}") from e
