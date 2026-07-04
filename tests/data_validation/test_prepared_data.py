"""Data-validation tests — the prepared frames match the contract prep_data.py promises.

These run against the real reference.parquet and current.parquet (both windows,
parametrized) rather than a synthetic fixture: they are the guard that the data
feeding training is shaped, balanced, and clean the way every downstream metric
assumes. If prep_data.py drifts from its spec, these fail before a model is trained
on the bad frame.
"""
from __future__ import annotations

import pytest

import preprocess as pp

# The prepared-frame contract (locked in prep_data.py).
N_NUMERIC = 148                 # mean + std across the 74 librosa groups
N_TOTAL_COLS = N_NUMERIC + len(pp.CATEGORICAL_COLS) + 1  # + 2 categoricals + target
POSITIVE_RATE_BAND = (0.10, 0.15)   # ~12.5% Hip-Hop, held constant across windows
ROW_FLOOR = 3000                # each window is ~4000 rows; floor with wide margin
N_INJECTED_NAN_COLS = 2         # seeded NaNs go into exactly 2 numeric columns
MAX_NAN_RATE = 0.04             # injected at 0.02; bounded well under this

WINDOWS = ["reference_df", "current_df"]


@pytest.mark.parametrize("window", WINDOWS)
def test_schema(window, request):
    """Both categoricals, the target, and exactly 148 numeric columns are present."""
    df = request.getfixturevalue(window)
    for col in (*pp.CATEGORICAL_COLS, pp.TARGET):
        assert col in df.columns, f"{window}: missing required column {col!r}"
    assert len(pp.numeric_columns(df)) == N_NUMERIC
    assert df.shape[1] == N_TOTAL_COLS


@pytest.mark.parametrize("window", WINDOWS)
def test_dtypes(window, request):
    """Numeric features are numeric, categoricals are non-numeric, target is integer."""
    import pandas.api.types as pdt

    df = request.getfixturevalue(window)
    for col in pp.numeric_columns(df):
        assert pdt.is_numeric_dtype(df[col]), f"{window}: {col} is not numeric"
    for col in pp.CATEGORICAL_COLS:
        assert not pdt.is_numeric_dtype(df[col]), f"{window}: {col} should be categorical"
    assert pdt.is_integer_dtype(df[pp.TARGET]), f"{window}: target must be integer 0/1"


@pytest.mark.parametrize("window", WINDOWS)
def test_row_floor(window, request):
    """Each window carries enough rows to train and monitor on."""
    df = request.getfixturevalue(window)
    assert len(df) >= ROW_FLOOR, f"{window}: only {len(df)} rows"


@pytest.mark.parametrize("window", WINDOWS)
def test_class_balance_band(window, request):
    """The Hip-Hop rate stays in its band in BOTH windows (the split holds it constant)."""
    df = request.getfixturevalue(window)
    rate = df[pp.TARGET].mean()
    lo, hi = POSITIVE_RATE_BAND
    assert lo <= rate <= hi, f"{window}: positive rate {rate:.3f} outside {POSITIVE_RATE_BAND}"


@pytest.mark.parametrize("window", WINDOWS)
def test_missingness_bounded(window, request):
    """Missingness matches the seeded injection: exactly 2 numeric columns, bounded rate.

    Categoricals and the target must be complete; injected NaNs live only in the
    numeric block at the rate prep_data.py wrote, so imputation has a known target.
    """
    df = request.getfixturevalue(window)
    numeric = pp.numeric_columns(df)

    cols_with_nan = [c for c in numeric if df[c].isna().any()]
    assert len(cols_with_nan) == N_INJECTED_NAN_COLS, (
        f"{window}: {len(cols_with_nan)} numeric cols carry NaN, expected {N_INJECTED_NAN_COLS}"
    )
    for c in cols_with_nan:
        assert df[c].isna().mean() <= MAX_NAN_RATE, f"{window}: {c} NaN rate too high"
    for c in (*pp.CATEGORICAL_COLS, pp.TARGET):
        assert not df[c].isna().any(), f"{window}: {c} has unexpected NaN"


@pytest.mark.parametrize("window", WINDOWS)
def test_no_duplicate_ids(window, request):
    """Track ids are unique — deduped on load, so no track is double-counted."""
    df = request.getfixturevalue(window)
    assert not df.index.duplicated().any(), f"{window}: duplicate track ids present"


@pytest.mark.parametrize("window", WINDOWS)
def test_target_is_binary(window, request):
    """The target is exactly {0, 1} with no stray values or NaN."""
    df = request.getfixturevalue(window)
    assert set(df[pp.TARGET].unique()) <= {0, 1}
