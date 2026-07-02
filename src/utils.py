"""Vendored FMA data loader.

Adapted from the Free Music Archive utilities:
  https://github.com/mdeff/fma  (utils.py) — MIT License, (c) 2016 Michael Defferrard.
Dataset paper: Defferrard, Benzi, Vandergheynst, Bresson,
"FMA: A Dataset for Music Analysis", ISMIR 2017.

We vendor only ``load()`` instead of depending on the whole FMA repo. It exists
because the FMA CSVs are NOT flat:
  - ``tracks.csv``   has a 2-level column header  -> header=[0, 1]
  - ``features.csv`` has a 3-level column header  -> header=[0, 1, 2]
  - ``set/subset``   is an ORDERED categorical (small < medium < large), which is
    what lets us filter ``subset <= 'small'`` without hand-coding the ordering.
Modernized for pandas 2.x (uses ``pd.CategoricalDtype`` instead of the removed
``astype('category', categories=..., ordered=True)`` signature).
"""
from __future__ import annotations

import ast
import os

import pandas as pd


def load(filepath: str) -> pd.DataFrame:
    """Load an FMA metadata CSV with the correct multi-level header and dtypes."""
    filename = os.path.basename(filepath)

    if "features" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "echonest" in filename:
        return pd.read_csv(filepath, index_col=0, header=[0, 1, 2])

    if "genres" in filename:
        return pd.read_csv(filepath, index_col=0)

    if "tracks" in filename:
        tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

        # list-valued columns are stored as string reprs -> parse to real lists
        list_cols = [("track", "tags"), ("album", "tags"), ("artist", "tags"),
                     ("track", "genres"), ("track", "genres_all")]
        for col in list_cols:
            tracks[col] = tracks[col].map(ast.literal_eval)

        # date columns -> datetime; album/date_released feeds the release-era feature
        date_cols = [("track", "date_created"), ("track", "date_recorded"),
                     ("album", "date_created"), ("album", "date_released"),
                     ("artist", "date_created"), ("artist", "active_year_begin"),
                     ("artist", "active_year_end")]
        for col in date_cols:
            tracks[col] = pd.to_datetime(tracks[col], errors="coerce")

        # ordered subset categorical: enables `subset <= 'small'` comparisons
        subsets = ("small", "medium", "large")
        tracks["set", "subset"] = tracks["set", "subset"].astype(
            pd.CategoricalDtype(categories=subsets, ordered=True)
        )

        cat_cols = [("track", "genre_top"), ("track", "license"),
                    ("album", "type"), ("album", "information"), ("artist", "bio")]
        for col in cat_cols:
            tracks[col] = tracks[col].astype("category")

        return tracks

    raise ValueError(f"unrecognized FMA metadata file: {filename}")
