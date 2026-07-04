"""evaluate.py — all model scoring in one place.

Every metric the project reports comes from here, so train.py, the comparison
script, and the model-validation tests all agree by construction. The primary
number is F1 on the Hip-Hop positive class: under a ~12.5% positive rate,
accuracy rewards a model that never predicts the minority class, and F1 does not.

Two ideas this module makes concrete:

  - The majority baseline. A model that always predicts the majority class scores
    ~88% accuracy and 0 recall on Hip-Hop. Reporting it next to the real model is
    the sanity check that shows why accuracy lies under imbalance.

  - Transfer. The detector trains only on the more-played (reference) half of each
    genre. `transfer_report` scores it on its own held-out reference test split and
    then, once, on the never-seen less-played (current) half, so the drop from
    reference-test F1 to current F1 is a measured answer to "does performance hold
    on the obscure tier," not an assertion.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

POSITIVE_LABEL = 1


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Score one set of predictions against the truth.

    F1 is pinned to the positive label so it measures Hip-Hop detection, not a
    class-averaged blend. ROC-AUC is only defined when probabilities are supplied
    and both classes are present; it is omitted otherwise rather than faked.
    The confusion counts (tn/fp/fn/tp) travel with the metrics so callers can see
    the error breakdown behind the F1.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: dict[str, float] = {
        "f1": float(f1_score(y_true, y_pred, pos_label=POSITIVE_LABEL, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics.update(tn=float(tn), fp=float(fp), fn=float(fn), tp=float(tp))

    if y_proba is not None and len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))

    return metrics


def majority_baseline_metrics(y_true: np.ndarray) -> dict[str, float]:
    """Score the 'always predict the majority class' strategy.

    This is the floor every real run must clear. On this data it lands near 88%
    accuracy with an F1 of 0 — the concrete demonstration that accuracy is the
    wrong headline metric here.
    """
    y_true = np.asarray(y_true)
    majority = int(np.bincount(y_true).argmax())
    y_pred = np.full_like(y_true, fill_value=majority)
    return compute_metrics(y_true, y_pred)


def evaluate_model(model: Any, X, y) -> dict[str, float]:
    """Predict with a fitted model and score it.

    Pulls positive-class probabilities when the estimator exposes them so ROC-AUC
    is available; falls back to labels-only scoring when it does not.
    """
    y_pred = model.predict(X)
    y_proba = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X)[:, POSITIVE_LABEL]
    return compute_metrics(y, y_pred, y_proba)


def transfer_report(model: Any, X_ref_test, y_ref_test, X_current, y_current) -> dict[str, float]:
    """Measure how F1 transfers from the reference tier to the held-out current tier.

    Returns reference-test F1, current F1, and their difference. A large positive
    gap means the detector leans on cues specific to popular tracks and degrades on
    obscure ones — the performance question drift monitoring exists to answer.
    """
    ref = evaluate_model(model, X_ref_test, y_ref_test)
    cur = evaluate_model(model, X_current, y_current)
    return {
        "reference_test_f1": ref["f1"],
        "current_f1": cur["f1"],
        "transfer_f1_drop": ref["f1"] - cur["f1"],
    }
