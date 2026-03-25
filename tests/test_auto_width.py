import polars as pl
from textual.pilot import Pilot

from dt_browser.browser import DtBrowserApp
from dt_browser.custom_table import CustomTable


def _make_varying_width_app(num_rows: int = 50) -> DtBrowserApp:
    """Create an app where row data varies in width so auto-width has a visible effect.

    Early rows have short values, later rows have long values.
    """
    df = pl.DataFrame(
        {
            "short_col": [f"s{i}" for i in range(num_rows)],
            "growing_col": [f"{'x' * (i + 1)}" for i in range(num_rows)],
            "fixed_col": [f"fixed_{i:04d}" for i in range(num_rows)],
        }
    )
    return DtBrowserApp("test", df)


def _make_app_simple(num_rows: int = 20) -> DtBrowserApp:
    df = pl.DataFrame(
        {
            "name": [f"item_{i}" for i in range(num_rows)],
            "value": list(range(num_rows)),
            "score": [round(i * 1.5, 1) for i in range(num_rows)],
            "category": [f"cat_{i % 5}" for i in range(num_rows)],
        }
    )
    return DtBrowserApp("test", df)


async def test_auto_width_toggle():
    """Toggling auto_width on and off updates the reactive property."""
    app = _make_varying_width_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        assert table.auto_width is False

        await pilot.press("w")
        await pilot.pause()
        assert table.auto_width is True

        await pilot.press("w")
        await pilot.pause()
        assert table.auto_width is False


async def test_auto_width_narrows_columns():
    """When auto_width is on, columns are narrower if visible rows have shorter data."""
    app = _make_varying_width_app(num_rows=100)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        # Full widths computed from all 100 rows (growing_col goes up to 100 chars)
        full_widths = table._widths.copy()

        await pilot.press("w")
        await pilot.pause()

        # Auto widths should be narrower for growing_col since visible rows are near the top
        auto_widths = table._widths.copy()
        assert auto_widths["growing_col"] < full_widths["growing_col"], (
            f"Expected auto width ({auto_widths['growing_col']}) < full width ({full_widths['growing_col']}) "
            f"for growing_col when viewing top rows"
        )


async def test_auto_width_updates_on_scroll():
    """Scrolling to rows with wider data increases auto column widths."""
    app = _make_varying_width_app(num_rows=100)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        await pilot.press("w")
        await pilot.pause()
        widths_at_top = table._widths.copy()

        # Scroll to the bottom where growing_col values are much wider
        await pilot.press("G")
        await pilot.pause()

        widths_at_bottom = table._widths.copy()
        assert widths_at_bottom["growing_col"] > widths_at_top["growing_col"], (
            f"Expected wider growing_col at bottom ({widths_at_bottom['growing_col']}) "
            f"than at top ({widths_at_top['growing_col']})"
        )


async def test_auto_width_restores_full_widths_on_toggle_off():
    """Toggling auto_width off restores the original precomputed widths."""
    app = _make_varying_width_app(num_rows=100)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        full_widths = table._widths.copy()

        # Toggle on
        await pilot.press("w")
        await pilot.pause()
        assert table._widths != full_widths  # auto widths should differ

        # Toggle off
        await pilot.press("w")
        await pilot.pause()
        assert table._widths == full_widths, "Widths should be restored to full precomputed values"


async def test_auto_width_respects_column_name_min_width():
    """Auto width never makes a column narrower than its header name."""
    app = _make_varying_width_app(num_rows=100)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        await pilot.press("w")
        await pilot.pause()

        for col in table._dt.columns:
            assert table._widths[col] >= len(col), (
                f"Column '{col}' width ({table._widths[col]}) is less than header name length ({len(col)})"
            )


async def test_auto_width_skips_recompute_when_range_unchanged():
    """Moving cursor within visible area does not trigger width recomputation."""
    app = _make_varying_width_app(num_rows=100)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        await pilot.press("w")
        await pilot.pause()
        widths_before = table._widths.copy()
        range_before = table._auto_width_visible_range

        # Move cursor within visible area
        await pilot.press("down")
        await pilot.pause()

        assert table._auto_width_visible_range == range_before, "Visible range should not change for in-view cursor move"
        assert table._widths == widths_before, "Widths should not change when cursor moves within visible area"


async def test_auto_width_with_resize():
    """Auto width recalculates when the terminal is resized."""
    app = _make_varying_width_app(num_rows=100)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        await pilot.press("w")
        await pilot.pause()
        range_before = table._auto_width_visible_range

        # Resize changes dt_height, so auto widths should recompute
        await pilot.resize_terminal(120, 20)
        await pilot.pause()

        # Force a render by accessing the property
        _ = table.render_header_and_table

        assert table._auto_width_visible_range != range_before, (
            "Visible range should change after resize"
        )


# --- Snapshot tests ---


def test_snap_auto_width_off(snap_compare):
    """Snapshot with auto width OFF (default)."""
    assert snap_compare(_make_varying_width_app(num_rows=100), terminal_size=(120, 30))


def test_snap_auto_width_on(snap_compare):
    """Snapshot with auto width ON — columns should be narrower at top of data."""
    assert snap_compare(
        _make_varying_width_app(num_rows=100),
        press=["w"],
        terminal_size=(120, 30),
    )


def test_snap_auto_width_on_scrolled_bottom(snap_compare):
    """Snapshot with auto width ON after scrolling to bottom — columns should be wider."""

    async def run_before(pilot: Pilot) -> None:
        await pilot.press("w")
        await pilot.pause()
        await pilot.press("G")
        await pilot.pause()

    assert snap_compare(
        _make_varying_width_app(num_rows=100),
        run_before=run_before,
        terminal_size=(120, 30),
    )


def test_snap_auto_width_toggled_off_after_on(snap_compare):
    """Snapshot after toggling auto width ON then OFF — should match original layout."""
    assert snap_compare(
        _make_varying_width_app(num_rows=100),
        press=["w", "w"],
        terminal_size=(120, 30),
    )
