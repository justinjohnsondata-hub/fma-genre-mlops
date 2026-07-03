"""preprocess.py — split the reference population and build the fit-on-train-only preprocessor.

Two jobs, both leak-safe by construction:

  1. split_reference(): ONE seeded, stratified train/test split taken WITHIN the
     reference window only. current.parquet is never touched here — it is the
     held-out drift tier (Phase 7), so letting it near the split would poison the
     "does F1 transfer to the obscure tier" question before we ever ask it.

  2. build_preprocessor(): a ColumnTransformer that imputes+scales the 148 numeric
     columns and one-hot-encodes the 2 categoricals. Every learned quantity
     (median fill, mean/std, the category vocabulary) is a FITTED parameter. The
     transformer is fit on X_train ONLY; X_test is only ever .transform()ed with
     those frozen train numbers. That is the leakage guard this phase exists to teach.

Nothing here mutates its input frame: sklearn copies internally, and split_reference
slices without writing back. Phase 2 tests assert that.

Locked knobs (Justin, 2026-07-03): median imputation, 80/20 split, stratify on y.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Schema of the prepared frame (prep_data.py output) ---
TARGET = "target"
CATEGORICAL_COLS = ["license_family", "release_era"]
# numeric = every other column; derived from the frame so we never hardcode 148 names.

# --- Locked knobs ---
DEFAULT_TEST_SIZE = 0.20
DEFAULT_SEED = 17
IMPUTE_STRATEGY = "median"


def numeric_columns(df: pd.DataFrame) -> list[str]:
    """The numeric feature columns = everything that isn't a categorical or the target."""
    return [c for c in df.columns if c not in CATEGORICAL_COLS and c != TARGET]


def validate_frame(df: pd.DataFrame) -> None:
    """Raise ValueError on invalid input BEFORE any sklearn call.

    A sklearn KeyError deep in a fit is a bad error surface for a caller; this is the
    explicit 'raises on invalid input' guard (brief behavior). Two checks:
      - every required column (the 2 categoricals + target) is present, and
      - the numeric columns are actually numeric dtype (an object column here means
        the frame was built wrong and StandardScaler would blow up cryptically).
    """
    required = [*CATEGORICAL_COLS, TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    num_cols = numeric_columns(df)
    if not num_cols:
        raise ValueError("no numeric feature columns found")
    non_numeric = [c for c in num_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"numeric columns have non-numeric dtype: {non_numeric[:5]}")


def split_reference(
    df: pd.DataFrame,
    *,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """One seeded, stratified train/test split within the reference frame.

    Returns (X_train, X_test, y_train, y_test). Stratifying on y holds the ~12.5%
    Hip-Hop rate on both sides so the test F1 estimates what we monitor. Does not
    mutate df (train_test_split returns row slices; df is untouched).
    """
    validate_frame(df)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)


def build_preprocessor(
    numeric_cols: list[str],
    categorical_cols: list[str] = CATEGORICAL_COLS,
    *,
    impute_strategy: str = IMPUTE_STRATEGY,
) -> ColumnTransformer:
    """Leak-safe preprocessing as a single fittable object.

    Numeric branch is itself a Pipeline: SimpleImputer -> StandardScaler. Order
    matters and is deliberate — impute the injected NaNs FIRST (fill value learned
    from train), THEN standardize. Both are train-only fitted parameters.

    Categorical branch: OneHotEncoder(handle_unknown="ignore") so a category seen
    only in test (or later in the current window) becomes an all-zeros row instead
    of crashing. sparse_output=False keeps the output a dense array for easy shape
    assertions in tests.
    """
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy=impute_strategy)),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [
            ("num", numeric, numeric_cols),
            ("cat", categorical, categorical_cols),
        ],
        remainder="drop",
    )


def load_reference(path: Path = Path("data/prepared/reference.parquet")) -> pd.DataFrame:
    """Convenience loader for scripts/tests. Reference window only."""
    return pd.read_parquet(path)
