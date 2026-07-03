"""Shared pytest fixtures + src path wiring for the test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make `import preprocess` (and other src modules) work without installing the package.
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    """A small, hand-checkable frame with the prepared-frame schema.

    3 numeric feature columns (one carries a NaN, standing in for the seeded
    missingness), the 2 real categoricals, and a binary target with both classes
    present (4 positives / 20 rows = 20%, enough to stratify-split). Deterministic.
    """
    rng = np.random.default_rng(0)
    n = 20
    df = pd.DataFrame(
        {
            "feat_a_mean_01": rng.normal(size=n),
            "feat_b_mean_02": rng.normal(size=n),
            "feat_c_std_01": rng.normal(size=n),
            "license_family": ["CC-BY", "CC-BY-NC", "other", "CC0/public-domain"] * 5,
            "release_era": ["pre_2000", "2000s", "2010s+", "unknown"] * 5,
            "target": [1, 0, 0, 0, 0] * 4,
        }
    )
    df.loc[0, "feat_a_mean_01"] = np.nan  # one injected-style NaN to exercise imputation
    return df


@pytest.fixture
def reference_df() -> pd.DataFrame:
    """The real reference window. Skips if prep_data.py hasn't been run yet."""
    path = Path(__file__).resolve().parent.parent / "data" / "prepared" / "reference.parquet"
    if not path.exists():
        pytest.skip("run `python src/prep_data.py` first to generate data/prepared/")
    return pd.read_parquet(path)
