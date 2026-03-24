import polars as pl

from dt_browser.browser import DtBrowser, DtBrowserApp
from dt_browser.filter_box import FilterBox


def _make_app() -> DtBrowserApp:
    df = pl.DataFrame({"name": ["alice", "bob", "charlie", "dave"], "value": [1, 2, 3, 4]})
    return DtBrowserApp("test", df)


async def test_filter_reduces_rows():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)
        original_rows = browser.cur_total_rows

        # Open filter with "f", type a WHERE clause, press enter
        await pilot.press("f")
        await pilot.pause()
        await pilot.press(*list("value > 2"))
        await pilot.press("enter")
        await pilot.pause()
        # Worker needs time to complete
        await pilot.pause()
        await pilot.pause()

        assert browser.is_filtered
        assert browser.cur_total_rows < original_rows
        assert browser.cur_total_rows == 2  # rows with value 3 and 4


async def test_filter_clear_restores_rows():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)
        original_rows = browser.cur_total_rows

        # Apply a filter first
        await pilot.press("f")
        await pilot.pause()
        await pilot.press(*list("value > 2"))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        assert browser.is_filtered
        assert browser.cur_total_rows < original_rows

        # Clear the filter: open filter box, clear the input, submit empty
        await pilot.press("f")  # re-open filter box
        await pilot.pause()
        # Select all text in the input and delete it
        filter_box = app.query_one(FilterBox)
        inp = filter_box.query_one("Input")
        # Clear the input by selecting all and deleting
        inp.value = ""
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        assert not browser.is_filtered
        assert browser.cur_total_rows == original_rows


async def test_search_highlights_results():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)

        # Open search with "/"
        await pilot.press("/")
        await pilot.pause()
        await pilot.press(*list("value > 1"))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        assert browser.active_search_queue is not None
        assert len(browser.active_search_queue) == 3  # rows with value 2, 3, 4


async def test_search_next_prev():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)

        # Open search, submit query
        await pilot.press("/")
        await pilot.pause()
        await pilot.press(*list("value > 1"))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()

        assert browser.active_search_queue is not None

        # After search, active_search_idx should be set (0 from the initial goto)
        initial_idx = browser.active_search_idx

        # Press "n" to go to next result
        await pilot.press("n")
        await pilot.pause()
        assert browser.active_search_idx == initial_idx + 1

        # Press "N" (shift+n) to go to previous result
        await pilot.press("N")
        await pilot.pause()
        assert browser.active_search_idx == initial_idx


async def test_filter_box_closes_on_escape():
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        # Open filter with "f"
        await pilot.press("f")
        await pilot.pause()
        assert app.query(FilterBox), "FilterBox should be mounted"

        # Press escape to close
        await pilot.press("escape")
        await pilot.pause()
        assert not app.query(FilterBox), "FilterBox should be removed after escape"
