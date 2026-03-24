import polars as pl

from dt_browser.browser import DtBrowserApp
from dt_browser.custom_table import CustomTable
from dt_browser.filter_box import FilterBox


def _make_app() -> DtBrowserApp:
    df = pl.DataFrame({"name": ["alice", "bob", "charlie", "dave"], "value": [1, 2, 3, 4]})
    return DtBrowserApp("test", df)


async def test_search_returns_focus_to_main_table():
    """After executing a search, focus should be on the main table, not the filter box."""
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        # Main table should have focus initially
        await pilot.pause()
        main_table = app.query_one("#main_table", CustomTable)
        assert main_table.has_focus

        # Open search with "/"
        await pilot.press("/")
        await pilot.pause()

        # FilterBox should now be mounted and its input focused
        filter_box = app.query_one(FilterBox)
        assert filter_box is not None
        assert not main_table.has_focus

        # Type a search query and submit
        await pilot.press(*list("value > 1"))
        await pilot.press("enter")
        await pilot.pause()

        # After search, focus must be back on the main table
        assert main_table.has_focus

        # FilterBox should be removed
        assert not app.query(FilterBox)
