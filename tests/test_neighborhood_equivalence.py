import numpy as np
import pandas as pd
from pathlib import Path

from hist_via_cluster.load_dataset import (
    old_load_frame,
    load_pixel_dataframe_with_neighborhood,
)


# Project root = parent of tests/
ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"


def test_neighborhood_size_1_matches_old_load_frame():
    """
    Integration test on the real dataset.

    For neighborhood_size=1, the new neighborhood-based loader must
    reproduce the same number of rows and identical fluorescence values
    as the old implementation (old_load_frame), when both are run on
    the real data under ./data.
    """
    # Make sure the data directory exists (otherwise fail fast with a clear msg)
    assert DATA_ROOT.exists(), f"Data root does not exist: {DATA_ROOT}"

    # Use the repository's data/ directory as the root for load_* functions
    directory = str(DATA_ROOT)

    # Old implementation: one column per element, e.g. 'Ca', 'Cu', ...
    df_old = old_load_frame(directory)

    # New implementation with neighborhood_size=1: element columns are
    # named e.g. 'Ca_+0_+0', 'Cu_+0_+0', ...
    df_new = load_pixel_dataframe_with_neighborhood(
        directory,
        neighborhood_size=1,
    )

    # Check row count
    assert len(df_old) == len(df_new), (
        f"Row counts differ between old and new loaders: "
        f"{len(df_old)} vs {len(df_new)}"
    )

    # Core metadata columns must exist and coincide
    for col in ["diet", "mouse", "take", "row", "col", "label"]:
        assert col in df_old.columns, f"Missing column {col} in df_old"
        assert col in df_new.columns, f"Missing column {col} in df_new"
        # For categorical columns, compare underlying values as strings
        if str(df_old[col].dtype) == "category" or str(df_new[col].dtype) == "category":
            old_vals = df_old[col].astype("string").values
            new_vals = df_new[col].astype("string").values
            assert (old_vals == new_vals).all(), f"Mismatch in column {col}"
        else:
            assert (df_old[col].values == df_new[col].values).all(), f"Mismatch in column {col}"

    # For each element e.g. "Ca" in old -> must equal new[f"{Ca}_+0_+0"]
    for col in df_old.columns:
        if col in ["diet", "mouse", "take", "row", "col", "label"]:
            continue

        new_col = f"{col}_+0_+0"
        assert new_col in df_new.columns, f"Missing column {new_col}"

        # Compare numerically (allow small float differences)
        assert np.allclose(
            df_old[col].values,
            df_new[new_col].values,
            atol=1e-8,
        ), f"Mismatch in element {col} between old and new loaders"
