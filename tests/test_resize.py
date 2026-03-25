import polars as pl

from dt_browser.browser import DtBrowserApp
from dt_browser.custom_table import CustomTable


def _make_wide_app(num_rows: int = 30) -> DtBrowserApp:
    """Create an app with 10 columns of varied widths for resize testing."""
    df = pl.DataFrame(
        {
            "id": list(range(num_rows)),
            "name": [f"item_{i}" for i in range(num_rows)],
            "description": [f"a longer description for row {i}" for i in range(num_rows)],
            "category": [f"cat_{i % 5}" for i in range(num_rows)],
            "price": [round(i * 3.14, 2) for i in range(num_rows)],
            "quantity": [i * 10 for i in range(num_rows)],
            "warehouse": [f"warehouse_{i % 3}_location" for i in range(num_rows)],
            "supplier": [f"supplier_{i % 7}_name" for i in range(num_rows)],
            "rating": [round(1.0 + (i % 50) / 10.0, 1) for i in range(num_rows)],
            "notes": [f"some notes about item number {i}" for i in range(num_rows)],
        }
    )
    return DtBrowserApp("test", df)


def _visible_data_columns(table: CustomTable) -> list[str]:
    """Return the data column names currently visible in the viewport.

    Replicates the column-fitting logic from render_header_and_table to determine
    which of the table's data columns fit in the current effective width.
    """
    scroll_x = table.scroll_offset.x
    effective_width = table.scrollable_content_region.width
    if effective_width <= 2:
        return []
    cols = []
    for col_name in table._dt.columns:
        min_offset = table._cum_widths[col_name] - scroll_x
        max_offset = min_offset + table._widths[col_name]
        if min_offset < 0:
            continue
        max_available = effective_width - min_offset - 2  # 2 * COL_PADDING where COL_PADDING=1
        if max_offset >= effective_width and max_available < 4:
            break
        cols.append(col_name)
        if max_offset >= effective_width:
            break
    return cols


async def test_resize_changes_visible_columns():
    """Resizing to a narrow terminal reduces the number of visible columns."""
    app = _make_wide_app()
    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)
        wide_visible = _visible_data_columns(table)

        # Now resize to something narrow
        await pilot.resize_terminal(40, 30)
        await pilot.pause()

        narrow_visible = _visible_data_columns(table)
        assert len(narrow_visible) < len(wide_visible), (
            f"Expected fewer columns at width 40 ({len(narrow_visible)}) "
            f"than at width 160 ({len(wide_visible)})"
        )


async def test_incremental_shrink():
    """Shrinking the terminal in steps monotonically reduces visible columns."""
    app = _make_wide_app()
    async with app.run_test(size=(200, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        counts = []
        for width in [200, 160, 120, 80, 60]:
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            counts.append(len(_visible_data_columns(table)))

        # Monotonically non-increasing
        for i in range(len(counts) - 1):
            assert counts[i] >= counts[i + 1], (
                f"Column count increased from {counts[i]} to {counts[i + 1]} "
                f"when shrinking terminal"
            )

        # Widest has strictly more columns than narrowest
        assert counts[0] > counts[-1], (
            f"Expected more columns at width 200 ({counts[0]}) than at 60 ({counts[-1]})"
        )


async def test_incremental_expand():
    """Expanding the terminal in steps monotonically increases visible columns."""
    app = _make_wide_app()
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        counts = []
        for width in [60, 80, 120, 160, 200]:
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            counts.append(len(_visible_data_columns(table)))

        # Monotonically non-decreasing
        for i in range(len(counts) - 1):
            assert counts[i] <= counts[i + 1], (
                f"Column count decreased from {counts[i]} to {counts[i + 1]} "
                f"when expanding terminal"
            )

        # Narrowest has strictly fewer columns than widest
        assert counts[0] < counts[-1], (
            f"Expected fewer columns at width 60 ({counts[0]}) than at 200 ({counts[-1]})"
        )


async def test_shrink_then_expand_restores():
    """Shrinking then expanding back to original width restores the same columns."""
    app = _make_wide_app()
    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        cols_before = _visible_data_columns(table)

        # Shrink
        await pilot.resize_terminal(60, 30)
        await pilot.pause()

        # Expand back
        await pilot.resize_terminal(160, 30)
        await pilot.pause()

        cols_after = _visible_data_columns(table)

        assert cols_before == cols_after, (
            f"Columns differ after shrink/expand cycle: "
            f"before={cols_before}, after={cols_after}"
        )


async def test_cursor_stays_valid_after_shrink():
    """Cursor remains within valid bounds after shrinking hides columns."""
    app = _make_wide_app()
    async with app.run_test(size=(200, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        # Move cursor to a rightward column
        total_cols = len(table._dt.columns)
        target_col = total_cols - 1
        for _ in range(target_col):
            await pilot.press("right")
            await pilot.pause()
        assert table.cursor_coordinate.column == target_col

        # Shrink so some columns are hidden
        await pilot.resize_terminal(60, 30)
        await pilot.pause()

        # Cursor column must be within valid range
        cursor = table.cursor_coordinate
        assert 0 <= cursor.row < len(table._dt)
        assert 0 <= cursor.column < total_cols

        # The cursor column must be visible in the viewport
        visible_cols = _visible_data_columns(table)
        cursor_col_name = table._dt.columns[cursor.column]
        assert cursor_col_name in visible_cols, (
            f"Cursor on column '{cursor_col_name}' which is not in "
            f"visible columns {visible_cols}"
        )
