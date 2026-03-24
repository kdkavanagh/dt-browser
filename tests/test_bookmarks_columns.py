import polars as pl

from dt_browser.bookmarks import Bookmarks
from dt_browser.browser import DtBrowser, DtBrowserApp
from dt_browser.column_selector import ColumnSelector


def _make_app() -> DtBrowserApp:
    df = pl.DataFrame({"name": ["alice", "bob", "charlie"], "value": [10, 20, 30]})
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
