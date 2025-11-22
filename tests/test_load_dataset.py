import numpy as np
import pandas as pd
from pathlib import Path

from hist_via_cluster.load_dataset import (
    old_load_frame,
    load_pixel_dataframe_with_neighborhood,
    load_fluorescence,
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
    assert DATA_ROOT.exists(), f"Data root does not exist: {DATA_ROOT}"

    directory = str(DATA_ROOT)

    df_old = old_load_frame(directory)
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


def test_neighborhood_offsets_correct_indexing_real_data():
    """
    Integration test on the real dataset.

    For neighborhood_size=3 (k=1), check that the neighborhood columns
    elem_{row_offset}_{col_offset} correspond to the correct pixels in
    the underlying images loaded via load_fluorescence(as_dict=True).

    We:
    - take one row of the DataFrame,
    - locate the corresponding image via (diet, mouse, take),
    - check that a few offsets map to the expected pixels.
    """
    assert DATA_ROOT.exists(), f"Data root does not exist: {DATA_ROOT}"

    directory = str(DATA_ROOT)

    ds = load_fluorescence(directory, as_dict=True)
    df = load_pixel_dataframe_with_neighborhood(
        directory,
        neighborhood_size=3,  # k = 1
    )

    # Use the first row as a representative interior pixel
    row_rec = df.iloc[0]

    diet_val = int(row_rec["diet"])
    mouse_val = int(row_rec["mouse"])
    take_val = int(row_rec["take"])
    key = (diet_val, mouse_val, take_val)

    assert key in ds.images, f"Key {key} not found in dataset.images"

    image = ds.images[key]
    element_order = ds.element_order
    height, width, n_elems = image.shape

    r = int(row_rec["row"])
    c = int(row_rec["col"])

    # For neighborhood_size=3 (k=1), all valid central pixels must satisfy:
    # 1 <= r <= height - 2, 1 <= c <= width - 2
    assert 1 <= r <= height - 2
    assert 1 <= c <= width - 2

    # Test a few representative offsets
    offsets = [
        (0, 0),   # center
        (1, 0),   # up
        (0, 1),   # right
        (-1, 0),  # down
        (1, -1),  # up-left
    ]

    for row_offset, col_offset in offsets:
        # Spec: +row_offset -> up (decreasing row index)
        #       +col_offset -> right (increasing col index)
        neighbor_row = r - row_offset
        neighbor_col = c + col_offset

        assert 0 <= neighbor_row < height
        assert 0 <= neighbor_col < width

        for elem_idx, elem in enumerate(element_order):
            expected_value = image[neighbor_row, neighbor_col, elem_idx]
            col_name = f"{elem}_{row_offset:+d}_{col_offset:+d}"
            assert col_name in df.columns, f"Missing column {col_name}"

            observed_value = row_rec[col_name]
            assert np.isclose(
                observed_value,
                expected_value,
                atol=1e-8,
            ), (
                f"Value mismatch for {col_name} at central (row, col)=({r}, {c}) "
                f"-> neighbor (row, col)=({neighbor_row}, {neighbor_col})"
            )


def test_neighborhood_excludes_border_pixels_real_data():
    """
    Integration test on the real dataset.

    For neighborhood_size=3 (k=1), the central pixels stored in the
    DataFrame must all have a complete 3x3 neighborhood inside the
    image. That means, for each image:

        k <= row <= height - 1 - k
        k <= col <= width - 1 - k

    We verify this per (diet, mouse, take) group.
    """
    assert DATA_ROOT.exists(), f"Data root does not exist: {DATA_ROOT}"

    directory = str(DATA_ROOT)

    ds = load_fluorescence(directory, as_dict=True)
    k = 1
    df = load_pixel_dataframe_with_neighborhood(
        directory,
        neighborhood_size=2 * k + 1,  # 3x3 neighborhood
    )

    # Group by image identity
    grouped = df.groupby(["diet", "mouse", "take"], observed=True)

    for (_, group) in grouped:
        # Extract integer identifiers from the group
        diet_val = int(group["diet"].iloc[0])
        mouse_val = int(group["mouse"].iloc[0])
        take_val = int(group["take"].iloc[0])
        key = (diet_val, mouse_val, take_val)

        assert key in ds.images, f"Key {key} not found in dataset.images"
        image = ds.images[key]
        height, width, _ = image.shape

        min_row = group["row"].min()
        max_row = group["row"].max()
        min_col = group["col"].min()
        max_col = group["col"].max()

        assert min_row >= k, (
            f"Found central row {min_row} < {k} for key {key}; "
            "border pixels should be excluded."
        )
        assert max_row <= height - 1 - k, (
            f"Found central row {max_row} > {height - 1 - k} for key {key}; "
            "border pixels should be excluded."
        )
        assert min_col >= k, (
            f"Found central col {min_col} < {k} for key {key}; "
            "border pixels should be excluded."
        )
        assert max_col <= width - 1 - k, (
            f"Found central col {max_col} > {width - 1 - k} for key {key}; "
            "border pixels should be excluded."
        )


def test_all_generated_columns_present_real_data():
    """
    Integration test on the real dataset.

    For a given neighborhood_size (e.g. 5, k=2), verify that the
    DataFrame contains *all* expected fluorescence columns of the form:

        elem_{row_offset}_{col_offset}

    for every elem in element_order and row_offset, col_offset in
    {-k, ..., +k}.
    """
    assert DATA_ROOT.exists(), f"Data root does not exist: {DATA_ROOT}"

    directory = str(DATA_ROOT)

    ds = load_fluorescence(directory, as_dict=True)
    elements = ds.element_order

    k = 2
    neighborhood_size = 2 * k + 1
    offsets = [
        (row_offset, col_offset)
        for row_offset in range(-k, k + 1)
        for col_offset in range(-k, k + 1)
    ]

    df = load_pixel_dataframe_with_neighborhood(
        directory,
        neighborhood_size=neighborhood_size,
    )

    for elem in elements:
        for row_offset, col_offset in offsets:
            col_name = f"{elem}_{row_offset:+d}_{col_offset:+d}"
            assert col_name in df.columns, f"Missing column: {col_name}"
