"""Phase 2 unit tests — preprocessing behavior.

Setup (arrange/act) is scaffolded. YOU write the assertion in each TODO(Justin)
block — that is the retainable part of this phase. Each unwritten test calls
pytest.fail(), so the suite is honestly RED until you fill them in.

Four brief-named behaviors must be represented; they are tagged [BEHAVIOR] below:
  - handles missing values      -> test_imputes_missing
  - encodes categoricals        -> test_encodes_categoricals
  - no-mutation                 -> test_no_mutation
  - raises on invalid input     -> test_raises_on_missing_column, test_raises_on_wrong_dtype
Plus: no train/test overlap, target is binary, and train-only fit (the gate concept).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import preprocess as pp


# --------------------------------------------------------------------------- #
# 1. Split hygiene                                                             #
# --------------------------------------------------------------------------- #
def test_split_no_overlap(tiny_frame):
    """No row is in both train and test."""
    X_train, X_test, y_train, y_test = pp.split_reference(tiny_frame, seed=17)

    train_ids = set(X_train.index)
    test_ids = set(X_test.index)
    # TODO(Justin): assert the two id sets do not overlap (their intersection is empty),
    # and — if you want a second line — that together they account for every input row.

    assert train_ids.isdisjoint(test_ids), "Train and test sets overlap"

    assert train_ids.union(test_ids) == set(tiny_frame.index), "Train and test sets do not cover all rows"



def test_target_is_binary(tiny_frame):
    """The target is exactly {0, 1}."""
    _, _, y_train, y_test = pp.split_reference(tiny_frame, seed=17)

    values = set(y_train.unique()) | set(y_test.unique())
    # TODO(Justin): assert `values` contains only 0 and 1 (nothing else, no NaN).
    assert values == {0, 1}, f"Target values are not binary: {values}"

    

# --------------------------------------------------------------------------- #
# 2. Transformer behavior                                                     #
# --------------------------------------------------------------------------- #
def test_encodes_categoricals(tiny_frame):
    """[BEHAVIOR: encodes categoricals] Output is all-numeric with the expected width."""
    num_cols = pp.numeric_columns(tiny_frame)
    pre = pp.build_preprocessor(num_cols)
    X = tiny_frame.drop(columns=["target"])
    out = pre.fit_transform(X)

    # In tiny_frame: license_family has 4 categories, release_era has 4 -> 8 one-hot cols.
    expected_width = len(num_cols) + 4 + 4
    # TODO(Justin): assert out.shape[1] == expected_width (categoricals became one-hot columns,
    # so the width is #numeric + #onehot). Optionally assert the output carries no object dtype.

    assert out.shape[1] == expected_width, f"Expected width {expected_width}, got {out.shape[1]}"


def test_imputes_missing(tiny_frame):
    """[BEHAVIOR: handles missing values] No NaN survives the transform."""
    assert tiny_frame["feat_a_mean_01"].isna().any()  # precondition: input really has a NaN

    num_cols = pp.numeric_columns(tiny_frame)
    pre = pp.build_preprocessor(num_cols)
    out = pre.fit_transform(tiny_frame.drop(columns=["target"]))

    # TODO(Justin): assert the transformed output has zero NaNs (np.isnan(out).sum() == 0),
    # i.e. the imputer filled the hole before scaling.

    assert not np.isnan(out).any(), "Transformed output contains NaN values"


def test_fit_is_train_only(tiny_frame):
    """[GATE CONCEPT] The imputer's fill value is learned from TRAIN, not the whole frame.

    If fit leaked test rows, the learned median would match the full-frame median.
    Fit on train only -> it matches the TRAIN median instead.
    """
    X_train, X_test, y_train, y_test = pp.split_reference(tiny_frame, seed=17)
    num_cols = pp.numeric_columns(tiny_frame)
    pre = pp.build_preprocessor(num_cols)
    pre.fit(X_train)  # fit on TRAIN ONLY

    learned_median = pre.named_transformers_["num"].named_steps["impute"].statistics_
    train_median = X_train[num_cols].median().to_numpy()
    full_median = tiny_frame[num_cols].median().to_numpy()
    # TODO(Justin): assert learned_median matches the TRAIN medians
    # (np.allclose(learned_median, train_median)). This is the leakage guard made observable:
    # the fitted parameter came from train rows only.

    assert np.allclose(learned_median, train_median), "Learned median does not match TRAIN median"


# --------------------------------------------------------------------------- #
# 3. Contract: no mutation, raises on invalid                                 #
# --------------------------------------------------------------------------- #
def test_no_mutation(tiny_frame):
    """[BEHAVIOR: no-mutation] Splitting + transforming does not alter the input frame."""
    before = tiny_frame.copy(deep=True)

    X_train, X_test, _, _ = pp.split_reference(tiny_frame, seed=17)
    pp.build_preprocessor(pp.numeric_columns(tiny_frame)).fit_transform(X_train)

    # TODO(Justin): assert tiny_frame is unchanged vs `before` (tiny_frame.equals(before)).

    assert tiny_frame.equals(before), "Input frame was mutated during processing"


def test_raises_on_missing_column(tiny_frame):
    """[BEHAVIOR: raises on invalid] A frame missing a required column is rejected."""
    bad = tiny_frame.drop(columns=["license_family"])

    # TODO(Justin): assert this raises ValueError. Use:
    #   with pytest.raises(ValueError):
    #       pp.validate_frame(bad)

    with pytest.raises(ValueError):
        pp.validate_frame(bad)


def test_raises_on_wrong_dtype(tiny_frame):
    """[BEHAVIOR: raises on invalid] A numeric column holding strings is rejected."""
    bad = tiny_frame.copy()
    bad["feat_a_mean_01"] = "not_a_number"

    # TODO(Justin): assert this raises ValueError (validate_frame catches the bad dtype).

    with pytest.raises(ValueError):
        pp.validate_frame(bad)
