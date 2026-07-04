"""Model-validation tests — the trained pipeline behaves like a usable classifier.

These train a small RandomForest on the reference train split inside the test
(fast, self-contained) rather than loading a logged run, so they hold even in CI
where the test job runs before any training job. They check the three things a
served model must do: beat the trivial baseline, return well-formed binary
predictions, and survive a row with a missing feature (the imputer's job).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import evaluate
import preprocess as pp
from train import build_estimator, build_pipeline

MIN_F1 = 0.40  # a real model must clear this; the majority baseline scores 0.0


@pytest.fixture(scope="module")
def fitted():
    """Train once per module: RF pipeline on the reference train split.

    Loads the reference frame directly (module-scoped, so it cannot reuse the
    function-scoped reference_df fixture). A trimmed forest (100 trees) keeps the
    test fast while still clearing the baseline. Returns the fitted pipeline plus
    the held-out test split.
    """
    path = Path(__file__).resolve().parents[2] / "data" / "prepared" / "reference.parquet"
    if not path.exists():
        pytest.skip("run `python src/prep_data.py` first to generate data/prepared/")
    reference_df = pd.read_parquet(path)

    X_train, X_test, y_train, y_test = pp.split_reference(reference_df, seed=17)
    estimator = build_estimator(
        "random_forest",
        {"n_estimators": 100, "class_weight": "balanced", "n_jobs": -1, "random_state": 17},
    )
    pipeline = build_pipeline(pp.numeric_columns(reference_df), estimator)
    pipeline.fit(X_train, y_train)
    return pipeline, X_test, y_test


def test_beats_majority_baseline(fitted):
    """The model's positive-class F1 beats the majority baseline (and clears a floor)."""
    model, X_test, y_test = fitted
    metrics = evaluate.evaluate_model(model, X_test, y_test)
    baseline = evaluate.majority_baseline_metrics(y_test)
    assert metrics["f1"] > baseline["f1"], "model does not beat the majority baseline"
    assert metrics["f1"] >= MIN_F1, f"F1 {metrics['f1']:.3f} below floor {MIN_F1}"


def test_predict_shape_and_classes(fitted):
    """predict returns one label per row, and every label is 0 or 1."""
    model, X_test, y_test = fitted
    preds = model.predict(X_test)
    assert preds.shape[0] == len(X_test)
    assert set(np.unique(preds)) <= {0, 1}


def test_predict_proba_in_unit_interval(fitted):
    """predict_proba returns a probability per class in [0, 1], two columns."""
    model, X_test, _ = fitted
    proba = model.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_nan_row_does_not_crash(fitted):
    """A row with a missing feature predicts cleanly — the imputer fills it, no crash."""
    model, X_test, _ = fitted
    row = X_test.iloc[[0]].copy()
    first_numeric = pp.numeric_columns(X_test)[0]
    row.loc[:, first_numeric] = np.nan  # blow a hole in the first numeric feature
    pred = model.predict(row)
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)
