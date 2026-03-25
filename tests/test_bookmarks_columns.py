import polars as pl

from dt_browser.bookmarks import Bookmarks
from dt_browser.browser import DtBrowser, DtBrowserApp
from dt_browser.column_selector import ColumnSelector
from dt_browser.custom_table import CustomTable


def _make_app(num_rows: int = 3) -> DtBrowserApp:
    if num_rows <= 3:
        df = pl.DataFrame({"name": ["alice", "bob", "charlie"], "value": [10, 20, 30]})
    else:
        df = pl.DataFrame(
            {
                "name": [f"row_{i}" for i in range(num_rows)],
                "value": list(range(num_rows)),
            }
        )
    return DtBrowserApp("test", df)


async def test_toggle_bookmark():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)
        bookmarks = browser._bookmarks

        # Bookmark the current row
        await pilot.press("b")
        await pilot.pause()
        assert bookmarks.has_bookmarks

        # Unbookmark the same row
        await pilot.press("b")
        await pilot.pause()
        assert not bookmarks.has_bookmarks


async def test_show_bookmarks_panel():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        # Add a bookmark first
        await pilot.press("b")
        await pilot.pause()

        # Show bookmarks panel
        await pilot.press("B")
        await pilot.pause()

        # Bookmarks widget should be mounted in the app
        assert len(app.query(Bookmarks)) == 1


async def test_column_selector_opens():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        await pilot.press("c")
        await pilot.pause()

        assert len(app.query(ColumnSelector)) > 0


async def test_column_visibility():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)
        initial_cols = browser.visible_columns

        # Open column selector
        await pilot.press("c")
        await pilot.pause()

        # The column selector's SelectionList should be focused.
        # The first item (Row #) should be highlighted. Toggle it off by pressing space.
        from textual.widgets import SelectionList

        sel_list = app.query_one("#showColumns SelectionList", SelectionList)
        col_selector = app.query_one("#showColumns", ColumnSelector)
        assert set(col_selector.selected_columns) == set(initial_cols)

        # Deselect the first column programmatically via the SelectionList
        # (space key is consumed by the ColumnSelector's Input filter)
        sel_list.deselect(initial_cols[0])
        await pilot.pause()

        # Verify the ColumnSelector's selected_columns changed
        assert len(col_selector.selected_columns) == len(initial_cols) - 1

        # Apply changes
        await pilot.press("ctrl+a")
        await pilot.pause()

        new_cols = browser.visible_columns
        assert len(new_cols) == len(initial_cols) - 1
        assert initial_cols[0] not in new_cols


async def test_all_columns_present_initially():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)

        # The app adds "Row #" as the first column to the original dataframe columns
        expected = ("Row #", "name", "value")
        assert browser.visible_columns == expected
        assert browser.all_columns == expected


async def test_bookmark_navigation():
    """Bookmark a row, navigate away, then use the bookmarks panel to jump back."""
    app = _make_app(num_rows=50)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)
        main_table = app.query_one("#main_table", CustomTable)

        # Navigate to row 25 using "G" (go to last row) then move up
        # Instead, navigate down 25 times from row 0
        for _ in range(25):
            await pilot.press("down")
        await pilot.pause()
        assert main_table.cursor_coordinate.row == 25

        # Bookmark row 25
        await pilot.press("b")
        await pilot.pause()
        assert browser._bookmarks.has_bookmarks

        # Go back to row 0
        await pilot.press("g")
        await pilot.pause()
        assert main_table.cursor_coordinate.row == 0

        # Open bookmarks panel
        await pilot.press("B")
        await pilot.pause()
        assert len(app.query(Bookmarks)) == 1

        # The bookmark table should have focus; press Enter to select
        await pilot.press("enter")
        await pilot.pause()

        # Main table cursor should have moved to row 25
        assert main_table.cursor_coordinate.row == 25


async def test_multiple_bookmarks_navigation():
    """Bookmark multiple rows, then navigate to a specific one via the bookmarks panel."""
    app = _make_app(num_rows=50)
    async with app.run_test(size=(120, 40)) as pilot:
        await pilot.pause()
        main_table = app.query_one("#main_table", CustomTable)

        # Bookmark row 0
        await pilot.press("b")
        await pilot.pause()

        # Navigate to row 10 and bookmark
        for _ in range(10):
            await pilot.press("down")
        await pilot.pause()
        assert main_table.cursor_coordinate.row == 10
        await pilot.press("b")
        await pilot.pause()

        # Navigate to row 20 and bookmark
        for _ in range(10):
            await pilot.press("down")
        await pilot.pause()
        assert main_table.cursor_coordinate.row == 20
        await pilot.press("b")
        await pilot.pause()

        # Go back to row 0
        await pilot.press("g")
        await pilot.pause()
        assert main_table.cursor_coordinate.row == 0

        # Open bookmarks panel
        await pilot.press("B")
        await pilot.pause()
        assert len(app.query(Bookmarks)) == 1

        # The bookmark table cursor starts at row 0 (first bookmark = row 0).
        # Press down to move to the second bookmark (row 10), then Enter.
        await pilot.press("down")
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        # Main table should navigate to row 10
        assert main_table.cursor_coordinate.row == 10


async def test_bookmark_panel_closes_on_escape():
    """Opening the bookmarks panel with B and closing it with Escape."""
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        # Bookmark a row so we can open the panel
        await pilot.press("b")
        await pilot.pause()

        # Open bookmarks panel
        await pilot.press("B")
        await pilot.pause()
        assert len(app.query(Bookmarks)) == 1

        # Press escape to close
        await pilot.press("escape")
        await pilot.pause()

        # Bookmarks widget should be removed
        assert len(app.query(Bookmarks)) == 0
