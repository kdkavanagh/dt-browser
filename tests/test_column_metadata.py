import datetime

import polars as pl
from textual.pilot import Pilot

from dt_browser.browser import DtBrowserApp
from dt_browser.column_metadata import ColumnMetadata, compute_column_stats
from dt_browser.custom_table import CustomTable


# --- Unit tests for stats computation ---


def test_numeric_stats():
    series = pl.Series("val", [1.0, 2.0, 3.0, 4.0, 5.0])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert labels == ["Min", "Q1", "Median", "Q3", "Max"]
    assert stats[0][1] == "1.0"
    assert stats[4][1] == "5.0"


def test_numeric_stats_integers():
    series = pl.Series("val", [10, 20, 30, 40, 50])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert labels == ["Min", "Q1", "Median", "Q3", "Max"]
    assert stats[0][1] == "10"
    assert stats[4][1] == "50"


def test_numeric_stats_empty():
    series = pl.Series("val", [], dtype=pl.Float64)
    stats = compute_column_stats(series)
    assert stats == [("", "No data")]


def test_categorical_stats():
    series = pl.Series("cat", ["a", "b", "a", "c", "b", "a"]).cast(pl.Categorical)
    stats = compute_column_stats(series)
    assert stats[0] == ("Unique values", "3")
    # Top entries should be sorted by count descending
    assert stats[1] == ("  a", "3")
    assert stats[2] == ("  b", "2")
    assert stats[3] == ("  c", "1")


def test_categorical_stats_top_10():
    values = [f"cat_{i}" for i in range(20)] * 2
    series = pl.Series("cat", values).cast(pl.Categorical)
    stats = compute_column_stats(series)
    assert stats[0][0] == "Unique values"
    # Should have at most 10 variant rows + 1 header
    assert len(stats) <= 11


def test_temporal_stats():
    series = pl.Series("ts", [datetime.datetime(2024, 1, 1), datetime.datetime(2024, 6, 15), datetime.datetime(2024, 12, 31)])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert labels == ["Min", "Max"]
    assert "2024-01-01" in stats[0][1]
    assert "2024-12-31" in stats[1][1]


def test_temporal_stats_empty():
    series = pl.Series("ts", [], dtype=pl.Datetime)
    stats = compute_column_stats(series)
    assert stats == [("", "No data")]


def test_boolean_stats():
    series = pl.Series("flag", [True, True, False, True, False])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert "True" in labels
    assert "False" in labels
    true_val = next(s[1] for s in stats if s[0] == "True")
    false_val = next(s[1] for s in stats if s[0] == "False")
    assert true_val == "3"
    assert false_val == "2"


def test_boolean_stats_with_nulls():
    series = pl.Series("flag", [True, None, False, None])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert "Null" in labels
    null_val = next(s[1] for s in stats if s[0] == "Null")
    assert null_val == "2"


def test_numeric_stats_with_nulls():
    series = pl.Series("val", [1.0, None, 3.0, None, 5.0])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert "Null" in labels
    null_val = next(s[1] for s in stats if s[0] == "Null")
    assert null_val == "2"


def test_numeric_stats_with_nans():
    series = pl.Series("val", [1.0, float("nan"), 3.0, float("nan"), 5.0])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert "NaN" in labels
    nan_val = next(s[1] for s in stats if s[0] == "NaN")
    assert nan_val == "2"


def test_numeric_integer_no_nan_row():
    series = pl.Series("val", [1, 2, 3])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert "NaN" not in labels


def test_temporal_stats_with_nulls():
    series = pl.Series("ts", [datetime.datetime(2024, 1, 1), None, datetime.datetime(2024, 12, 31)])
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert "Null" in labels
    null_val = next(s[1] for s in stats if s[0] == "Null")
    assert null_val == "1"


def test_categorical_stats_with_nulls():
    series = pl.Series("cat", ["a", None, "b", None]).cast(pl.Categorical)
    stats = compute_column_stats(series)
    labels = [s[0] for s in stats]
    assert "Null" in labels
    null_val = next(s[1] for s in stats if s[0] == "Null")
    assert null_val == "2"


def test_unsupported_dtype_returns_empty():
    series = pl.Series("text", ["hello", "world"])
    stats = compute_column_stats(series)
    assert stats == []


# --- Integration tests ---


def _make_mixed_app(num_rows: int = 30) -> DtBrowserApp:
    df = pl.DataFrame(
        {
            "id": list(range(num_rows)),
            "score": [round(i * 1.5, 1) for i in range(num_rows)],
            "category": pl.Series([f"cat_{i % 5}" for i in range(num_rows)]).cast(pl.Categorical),
            "active": [i % 3 == 0 for i in range(num_rows)],
            "name": [f"item_{i}" for i in range(num_rows)],
        }
    )
    return DtBrowserApp("test", df)


async def test_column_metadata_visible_on_start():
    """Column metadata panel should be visible on app start and within viewport."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        metadata = app.query_one(ColumnMetadata)
        assert metadata is not None
        # Should show stats for the first column (Row # which is integer)
        assert metadata.column_info is not None
        # Must be within the visible area
        assert metadata.region.y + metadata.region.height <= 30, (
            f"ColumnMetadata at y={metadata.region.y} h={metadata.region.height} is off-screen"
        )
        assert metadata.region.height > 0


async def test_column_metadata_displays_numeric_stats():
    """Verify numeric column stats are actually rendered in the widget."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        metadata = app.query_one(ColumnMetadata)
        # First column is Row # (numeric) — stats should be cached
        col_name = metadata.column_info[0]
        assert col_name in metadata._cache
        stats = metadata._cache[col_name]
        labels = [s[0] for s in stats]
        assert "Min" in labels
        assert "Median" in labels
        assert "Max" in labels
        # Border title should show column name
        assert col_name in metadata.border_title


async def test_column_metadata_displays_categorical_stats():
    """Verify categorical column stats are rendered with value counts."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        metadata = app.query_one(ColumnMetadata)
        # Navigate to 'category' column (Row#, id, score, category)
        for _ in range(3):
            await pilot.press("right")
            await pilot.pause()
        assert metadata.column_info[0] == "category"
        stats = metadata._cache["category"]
        assert stats[0] == ("Unique values", "5")
        # Should have value count entries
        assert len(stats) == 6  # 1 header + 5 categories
        assert "category" in metadata.border_title


async def test_column_metadata_displays_boolean_stats():
    """Verify boolean column stats show True/False counts."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        metadata = app.query_one(ColumnMetadata)
        # Navigate to 'active' column (Row#, id, score, category, active)
        for _ in range(4):
            await pilot.press("right")
            await pilot.pause()
        assert metadata.column_info[0] == "active"
        stats = metadata._cache["active"]
        labels = [s[0] for s in stats]
        assert "True" in labels
        assert "False" in labels


async def test_column_metadata_updates_on_cursor_move():
    """Moving cursor to a different column updates column_info."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        metadata = app.query_one(ColumnMetadata)
        initial_info = metadata.column_info

        await pilot.press("right")
        await pilot.pause()

        assert metadata.column_info != initial_info


async def test_column_metadata_cache_works():
    """Moving to a column, away, and back should use cached stats."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        metadata = app.query_one(ColumnMetadata)

        # Move to column 1
        await pilot.press("right")
        await pilot.pause()
        col_name_1 = metadata.column_info[0]
        assert col_name_1 in metadata._cache

        # Move to column 2
        await pilot.press("right")
        await pilot.pause()

        # Move back to column 1
        await pilot.press("left")
        await pilot.pause()
        # Cache should still have the entry
        assert col_name_1 in metadata._cache


async def test_column_metadata_cache_invalidated_on_filter():
    """Applying a filter should clear the column metadata cache."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        metadata = app.query_one(ColumnMetadata)

        # Move to trigger cache population
        await pilot.press("right")
        await pilot.pause()
        assert len(metadata._cache) > 0

        # Apply a filter
        await pilot.press("f")
        await pilot.pause()
        await pilot.press(*list("id > 10"))
        await pilot.press("enter")
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

        assert len(metadata._cache) == 0


async def test_column_metadata_hidden_with_row_detail():
    """Toggling row detail off should also hide column metadata."""
    app = _make_mixed_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert len(app.query(ColumnMetadata)) == 1

        await pilot.press("r")
        await pilot.pause()

        # Both should be gone (inside DetailPanel)
        assert len(app.query(ColumnMetadata)) == 0


# --- Snapshot tests ---


def test_snap_column_metadata_numeric(snap_compare):
    """Snapshot showing numeric column stats (id column)."""
    assert snap_compare(
        _make_mixed_app(),
        press=["right"],
        terminal_size=(120, 30),
    )


def test_snap_column_metadata_categorical(snap_compare):
    """Snapshot showing categorical column stats."""

    async def run_before(pilot: Pilot) -> None:
        # Navigate to the 'category' column (index 3: Row#, id, score, category)
        for _ in range(3):
            await pilot.press("right")
            await pilot.pause()

    assert snap_compare(
        _make_mixed_app(),
        run_before=run_before,
        terminal_size=(120, 30),
    )


def test_snap_column_metadata_boolean(snap_compare):
    """Snapshot showing boolean column stats."""

    async def run_before(pilot: Pilot) -> None:
        # Navigate to the 'active' column (index 4: Row#, id, score, category, active)
        for _ in range(4):
            await pilot.press("right")
            await pilot.pause()

    assert snap_compare(
        _make_mixed_app(),
        run_before=run_before,
        terminal_size=(120, 30),
    )


def test_snap_detail_panel_layout(snap_compare):
    """Snapshot showing full detail panel with both row detail and column metadata."""
    assert snap_compare(
        _make_mixed_app(),
        press=["down", "down"],
        terminal_size=(120, 30),
    )
