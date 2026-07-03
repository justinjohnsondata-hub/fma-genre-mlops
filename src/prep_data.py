"""prep_data.py — build the modeling frame and the two drift windows.

Pipeline (top to bottom):
  1. load fma_small (tracks + features, aligned on track_id)          [plumbing]
  2. build the binary Hip-Hop target                                  TODO(Justin)
  3. select the 148 numeric (mean+std) features                       TODO(Justin)
  4. engineer the 2 categoricals (license family + release-era)       TODO(Justin)
  5. assemble X, dedupe on track_id, inject seeded NaNs               [plumbing]
  6. assign the within-genre listens split (reference/current)        TODO(Justin)  <- gate
  7. assert each window cleared its Hip-Hop floor, save artifacts     [plumbing]

The four TODO(Justin) blocks are the decisions/concepts to retain; they raise
NotImplementedError until filled, so a partial run tells you what's left. The
plumbing around them is done.

Run from repo root with the venv active:
    python src/prep_data.py                 # uses data/raw -> data/prepared
    python src/prep_data.py --seed 17
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import utils  # noqa: E402

# --- Locked knobs (PLAN.md) ---
DEFAULT_RAW = Path("data/raw/fma_metadata")
DEFAULT_OUT = Path("data/prepared")
DEFAULT_SEED = 17
HIPHOP_FLOOR = 450  # each drift window must carry ~500 Hip-Hop positives; floor with margin
NAN_FRACTION = 0.02  # seeded missingness injected into 2 numeric cols (Phase 5 tests this rate)

# Engagement columns kept OUT of X (they leak the drift signal); listens drives the split only.
ENGAGEMENT = [("track", "listens"), ("track", "favorites"), ("track", "interest")]


# --------------------------------------------------------------------------- #
# 1. Load fma_small                                                [plumbing]  #
# --------------------------------------------------------------------------- #
def load_small(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load tracks + features, filter to fma_small, align on track_id.

    Returns (tracks_small, features_small) sharing the same 8000-row index.
    """
    tracks = utils.load(str(raw_dir / "tracks.csv"))
    features = utils.load(str(raw_dir / "features.csv"))

    small_mask = tracks[("set", "subset")] <= "small"
    tracks_small = tracks[small_mask]

    # align features to the same track ids (intersection guards against any gaps)
    common = tracks_small.index.intersection(features.index)
    tracks_small = tracks_small.loc[common]
    features_small = features.loc[common]
    return tracks_small, features_small


# --------------------------------------------------------------------------- #
# 2. Target                                                    TODO(Justin)    #
# --------------------------------------------------------------------------- #
def build_target(tracks_small: pd.DataFrame) -> pd.Series:
    """Return the binary Hip-Hop target y as int 0/1, indexed by track_id.

    Spec: y = 1 where the track's top genre is exactly "Hip-Hop", else 0.
    The genre label lives at column ("track", "genre_top"). Confirmed label
    string is "Hip-Hop" (verify_data.py showed 1000/8000 = 12.5% positive).
    Return an integer Series (not bool) so downstream checks see 0/1.
    """

    y = tracks_small[("track", "genre_top")] == "Hip-Hop"
    return y.astype(int)



# --------------------------------------------------------------------------- #
# 3. Numeric features                                          TODO(Justin)    #
# --------------------------------------------------------------------------- #
def select_numeric_features(features_small: pd.DataFrame) -> pd.DataFrame:
    """Return the 148 numeric summary columns (mean + std across the 74 groups).

    Spec: features_small has a 3-level column index (feature, statistics, number).
    Keep only columns whose 'statistics' level (level 1) is "mean" or "std".
    verify_data.py confirmed that yields 148 columns. Flatten the column names
    to readable strings so the modeling frame is easy to inspect and one-hot
    later (e.g. "mfcc_mean_01"). Keep the track_id index intact.
    """

    stats = features_small.columns.get_level_values("statistics")

    mask = stats.isin(["mean", "std"])

    selected = features_small.loc[:, mask]

    selected.columns = ["_".join(map(str, col)) for col in selected.columns]
    return selected

   


# --------------------------------------------------------------------------- #
# 4. Categorical features                                      TODO(Justin)    #
# --------------------------------------------------------------------------- #
def engineer_categoricals(tracks_small: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame of the 2 engineered categoricals, indexed by track_id.

    Spec (two columns):
      - license_family: bucket ("track", "license") into a few CC families
        (e.g. CC-BY, CC-BY-NC, CC-BY-NC-ND, CC0/public-domain, other).
      - release_era:    bin ("album", "date_released") into eras. Do NOT use
        ("track", "date_created") — it clusters on the site's bulk-upload dates
        and collapses. Fallback if date_released NaN-rate is too high: bin
        ("track", "bit_rate") into bands instead. (Check the NaN rate and decide.)
    These give the pipeline its required mixed numeric/categorical types.
    """
    
    license_series = tracks_small[("track", "license")].copy()

    license_series = license_series.str.lower().str.strip().str.replace(" ", "")  # normalize case and whitespace

    def bucket_license(s: str) -> str:
        """Map one normalized license string to its CC family."""

        if pd.isna(s):
            return "other"
    
        # most-specific first, broadest last:
        if "cc0" in s or "publicdomain" in s:                     # cc0 / publicdomain
            return "CC0/public-domain"
    
        if "noncommercial" in s and "noderiv" in s:                    # NC + ND
            return "CC-BY-NC-ND"
    
        if "noncommercial" in s and "sharealike" in s:                    # NC + SA
            return "CC-BY-NC-SA"

        if "noncommercial" in s:                            # NC only
            return "CC-BY-NC"
    
        if "noderiv" in s:                            # ND only
            return "CC-BY-ND"
    
        if "sharealike" in s:                            # SA only
            return "CC-BY-SA"
    
        if "attribution" in s:                            # attribution only
            return "CC-BY"
        return "other"

 

    license_series = license_series.apply(bucket_license)



    release_series = tracks_small[("album", "date_released")].copy()

    release_series = pd.to_datetime(release_series, errors='coerce')

    release_era_bins = [pd.Timestamp('1900-01-01'), pd.Timestamp('2000-01-01'),
                    pd.Timestamp('2010-01-01'), pd.Timestamp('2100-01-01')]   # 4 edges -> 3 intervals
    release_era_labels = ["pre_2000", "2000s", "2010s+"]

    release_series = pd.cut(release_series, bins=release_era_bins, labels=release_era_labels, right=False)

    release_series = release_series.astype(object).fillna("unknown")

    categoricals_df = pd.DataFrame({
        'license_family': license_series,
        'release_era': release_series
    }, index=tracks_small.index) 

    return categoricals_df   

    

# --------------------------------------------------------------------------- #
# 5. Assemble, dedupe, inject missingness                          [plumbing]  #
# --------------------------------------------------------------------------- #
def inject_missing(X: pd.DataFrame, numeric_cols: list[str], seed: int,
                   fraction: float = NAN_FRACTION, n_cols: int = 2) -> pd.DataFrame:
    """Inject seeded NaNs into `n_cols` numeric columns at `fraction` rate.

    Mechanism only (plumbing). Deterministic via a seeded Generator so the
    missingness is reproducible and Phase 5 can assert the exact rate.
    """
    rng = np.random.default_rng(seed)
    X = X.copy()
    target_cols = numeric_cols[:n_cols]  # first n numeric cols; adjust if you want specific ones
    for col in target_cols:
        mask = rng.random(len(X)) < fraction
        X.loc[mask, col] = np.nan
    return X


# --------------------------------------------------------------------------- #
# 6. Within-genre listens split                               TODO(Justin)     #
#    THE GATE CONCEPT — save for last.                                          #
def assign_within_genre_split(tracks_small: pd.DataFrame) -> pd.Series:
    """Return a Series of "reference"/"current" per track_id.

    Spec (the crux): split ("track", "listens") at the median WITHIN EACH
    genre (group by ("track", "genre_top")). reference = the more-played half
    of every genre, current = the less-played half. A global split would let
    the genre mix shift between windows and starve Hip-Hop positives; splitting
    inside each genre holds the mix constant and guarantees ~500 Hip-Hop per
    window -> one honest covariate-shift question.
    """
    listens = tracks_small[("track", "listens")]
    genre = tracks_small[("track", "genre_top")]

    # each row's OWN genre median, broadcast back to every row:
    genre_median = listens.groupby(genre, observed=True).transform("median")

    # reference = the more-played half of each genre (listens >= its genre median), current = the rest
    window = pd.Series(np.where(listens >= genre_median, "reference", "current"), index=tracks_small.index)
    
    return window


# --------------------------------------------------------------------------- #
# 7. Save                                                          [plumbing]  #
# --------------------------------------------------------------------------- #
def save_artifacts(reference: pd.DataFrame, current: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(out_dir / "reference.parquet")
    current.to_parquet(out_dir / "current.parquet")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Prepare FMA modeling frame + drift windows.")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = p.parse_args(argv)

    # 1. load
    tracks_small, features_small = load_small(args.raw)
    print(f"fma_small: {len(tracks_small)} tracks")

    # 2-4. target + features (Justin's decisions)
    y = build_target(tracks_small)
    X_num = select_numeric_features(features_small)
    X_cat = engineer_categoricals(tracks_small)
    numeric_cols = list(X_num.columns)

    # 5. assemble X, attach target, dedupe on track_id, inject seeded NaNs
    df = pd.concat([X_num, X_cat], axis=1)
    df["target"] = y
    df = df[~df.index.duplicated(keep="first")]  # dedupe on track_id
    df = inject_missing(df, numeric_cols, seed=args.seed)

    # 6. within-genre split (Justin's decision; the gate)
    window = assign_within_genre_split(tracks_small).reindex(df.index)
    reference = df[window == "reference"]
    current = df[window == "current"]

    # 7. runtime floor check (PLAN: assert each window cleared ~500 Hip-Hop) + save
    ref_pos, cur_pos = int(reference["target"].sum()), int(current["target"].sum())
    print(f"reference: {len(reference)} rows, {ref_pos} Hip-Hop | "
          f"current: {len(current)} rows, {cur_pos} Hip-Hop")
    assert ref_pos >= HIPHOP_FLOOR, f"reference Hip-Hop {ref_pos} < floor {HIPHOP_FLOOR}"
    assert cur_pos >= HIPHOP_FLOOR, f"current Hip-Hop {cur_pos} < floor {HIPHOP_FLOOR}"

    save_artifacts(reference, current, args.out)
    print(f"saved reference/current to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
