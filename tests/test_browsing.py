import polars as pl
from textual.coordinate import Coordinate

from dt_browser.browser import DtBrowserApp
from dt_browser.custom_table import CustomTable


def _make_app(num_rows: int = 50) -> DtBrowserApp:
    df = pl.DataFrame(
        {
            "name": [f"item_{i}" for i in range(num_rows)],
            "value": list(range(num_rows)),
            "score": [float(i) * 1.5 for i in range(num_rows)],
            "category": [f"cat_{i % 5}" for i in range(num_rows)],
            "extra": [f"extra_data_{i}" for i in range(num_rows)],
        }
    )
    return DtBrowserApp("test", df)


async def test_app_starts_with_data():
    """App mounts and table has the correct number of rows."""
    app = _make_app(num_rows=20)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)
        assert len(table._dt) == 20
        assert table.cursor_coordinate == Coordinate(0, 0)


async def test_cursor_movement_arrows():
    """Arrow keys move the cursor down and right."""
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        await pilot.press("down")
        await pilot.pause()
        assert table.cursor_coordinate.row == 1

        await pilot.press("down")
        await pilot.pause()
        assert table.cursor_coordinate.row == 2

        await pilot.press("right")
        await pilot.pause()
        assert table.cursor_coordinate.column == 1

        await pilot.press("up")
        await pilot.pause()
        assert table.cursor_coordinate.row == 1


async def test_cursor_jump_top_bottom():
    """G goes to last row, g goes to first row."""
    app = _make_app(num_rows=50)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)

        await pilot.press("G")
        await pilot.pause()
        assert table.cursor_coordinate.row == 49

        await pilot.press("g")
        await pilot.pause()
        assert table.cursor_coordinate.row == 0


async def test_resize_changes_visible_columns():
    """Resizing to a narrow terminal reduces the number of visible columns."""
    app = _make_app()
    async with app.run_test(size=(160, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)
        wide_widths = set(table._widths.keys())

        # Now resize to something narrow
        await pilot.resize_terminal(40, 30)
        await pilot.pause()

        # After resize, fewer columns should fit in the rendered output
        _, render_df = table.render_header_and_table
        rendered_cols = [c for c in render_df.columns if c in wide_widths]
        assert len(rendered_cols) < len(wide_widths)


async def test_page_down_up():
    """Pagedown moves cursor significantly, pageup brings it back."""
    app = _make_app(num_rows=100)
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        table = app.query_one("#main_table", CustomTable)
        assert table.cursor_coordinate.row == 0

        await pilot.press("pagedown")
        await pilot.pause()
        row_after_pgdn = table.cursor_coordinate.row
        assert row_after_pgdn > 5  # moved down significantly

        await pilot.press("pageup")
        await pilot.pause()
        assert table.cursor_coordinate.row < row_after_pgdn
