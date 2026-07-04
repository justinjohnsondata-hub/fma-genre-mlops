"""Unit tests — preprocessing behavior.

Four core behaviors are covered: handles missing values (test_imputes_missing),
encodes categoricals (test_encodes_categoricals), does not mutate its input
(test_no_mutation), and raises on invalid input (test_raises_on_missing_column,
test_raises_on_wrong_dtype). Plus split hygiene: no train/test overlap, a binary
target, and the train-only fit that keeps the preprocessor leak-safe.
"""
from __future__ import annotations

import numpy as np
import pytest

import preprocess as pp


# --------------------------------------------------------------------------- #
# 1. Split hygiene                                                             #
# --------------------------------------------------------------------------- #
def test_split_no_overlap(tiny_frame):
    """No row is in both train and test, and together they cover every input row."""
    X_train, X_test, y_train, y_test = pp.split_reference(tiny_frame, seed=17)

    train_ids = set(X_train.index)
    test_ids = set(X_test.index)
    assert train_ids.isdisjoint(test_ids), "Train and test sets overlap"
    assert train_ids.union(test_ids) == set(tiny_frame.index), "Split does not cover all rows"


def test_target_is_binary(tiny_frame):
    """The target is exactly {0, 1}."""
    _, _, y_train, y_test = pp.split_reference(tiny_frame, seed=17)

    values = set(y_train.unique()) | set(y_test.unique())
    assert values == {0, 1}, f"Target values are not binary: {values}"


# --------------------------------------------------------------------------- #
# 2. Transformer behavior                                                     #
# --------------------------------------------------------------------------- #
def test_encodes_categoricals(tiny_frame):
    """Output is all-numeric with the expected width (numeric + one-hot columns)."""
    num_cols = pp.numeric_columns(tiny_frame)
    pre = pp.build_preprocessor(num_cols)
    X = tiny_frame.drop(columns=["target"])
    out = pre.fit_transform(X)

    # tiny_frame: license_family has 4 categories, release_era has 4 -> 8 one-hot cols.
    expected_width = len(num_cols) + 4 + 4
    assert out.shape[1] == expected_width, f"Expected width {expected_width}, got {out.shape[1]}"


def test_imputes_missing(tiny_frame):
    """No NaN survives the transform — the imputer fills the hole before scaling."""
    assert tiny_frame["feat_a_mean_01"].isna().any()  # precondition: input really has a NaN

    num_cols = pp.numeric_columns(tiny_frame)
    pre = pp.build_preprocessor(num_cols)
    out = pre.fit_transform(tiny_frame.drop(columns=["target"]))

    assert not np.isnan(out).any(), "Transformed output contains NaN values"


def test_fit_is_train_only(tiny_frame):
    """The imputer's fill value is learned from TRAIN, not the whole frame.

    If fit leaked test rows, the learned median would match the full-frame median.
    Fit on train only -> it matches the TRAIN median instead.
    """
    X_train, X_test, y_train, y_test = pp.split_reference(tiny_frame, seed=17)
    num_cols = pp.numeric_columns(tiny_frame)
    pre = pp.build_preprocessor(num_cols)
    pre.fit(X_train)  # fit on TRAIN ONLY

    learned_median = pre.named_transformers_["num"].named_steps["impute"].statistics_
    train_median = X_train[num_cols].median().to_numpy()
    assert np.allclose(learned_median, train_median), "Learned median does not match TRAIN median"


# --------------------------------------------------------------------------- #
# 3. Contract: no mutation, raises on invalid                                 #
# --------------------------------------------------------------------------- #
def test_no_mutation(tiny_frame):
    """Splitting + transforming does not alter the input frame."""
    before = tiny_frame.copy(deep=True)

    X_train, X_test, _, _ = pp.split_reference(tiny_frame, seed=17)
    pp.build_preprocessor(pp.numeric_columns(tiny_frame)).fit_transform(X_train)

    assert tiny_frame.equals(before), "Input frame was mutated during processing"


def test_raises_on_missing_column(tiny_frame):
    """A frame missing a required column is rejected."""
    bad = tiny_frame.drop(columns=["license_family"])
    with pytest.raises(ValueError):
        pp.validate_frame(bad)


def test_raises_on_wrong_dtype(tiny_frame):
    """A numeric column holding strings is rejected."""
    bad = tiny_frame.copy()
    bad["feat_a_mean_01"] = "not_a_number"
    with pytest.raises(ValueError):
        pp.validate_frame(bad)
