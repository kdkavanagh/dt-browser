import polars as pl
from textual.pilot import Pilot

from dt_browser.browser import DtBrowserApp
from dt_browser.custom_table import CustomTable


def _make_app(num_rows: int = 20) -> DtBrowserApp:
    df = pl.DataFrame(
        {
            "name": [f"item_{i}" for i in range(num_rows)],
            "value": list(range(num_rows)),
            "score": [round(i * 1.5, 1) for i in range(num_rows)],
            "category": [f"cat_{i % 5}" for i in range(num_rows)],
        }
    )
    return DtBrowserApp("test", df)


def test_snap_initial_view(snap_compare):
    """Snapshot of the app on initial load."""
    assert snap_compare(_make_app(), terminal_size=(120, 30))


def test_snap_cursor_moved(snap_compare):
    """Snapshot after moving cursor down and right."""
    assert snap_compare(
        _make_app(),
        press=["down", "down", "down", "right", "right"],
        terminal_size=(120, 30),
    )


def test_snap_filter_applied(snap_compare):
    """Snapshot after applying a filter."""

    async def run_before(pilot: Pilot) -> None:
        await pilot.press("f")
        await pilot.pause()
        await pilot.press(*list("value > 10"))
        await pilot.press("enter")
        # Wait for the @work filter to complete
        await pilot.app.workers.wait_for_complete()
        await pilot.pause()

    assert snap_compare(_make_app(), run_before=run_before, terminal_size=(120, 30))


def test_snap_search_results(snap_compare):
    """Snapshot showing search result highlighting."""

    async def run_before(pilot: Pilot) -> None:
        await pilot.press("/")
        await pilot.pause()
        await pilot.press(*list("value > 5"))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

    assert snap_compare(_make_app(), run_before=run_before, terminal_size=(120, 30))


def test_snap_column_selector_open(snap_compare):
    """Snapshot with column selector panel open."""
    assert snap_compare(
        _make_app(),
        press=["c"],
        terminal_size=(120, 30),
    )


def test_snap_column_hidden(snap_compare):
    """Snapshot after hiding a column via the column selector."""

    async def run_before(pilot: Pilot) -> None:
        from textual.widgets import SelectionList

        await pilot.press("c")
        await pilot.pause()
        # Deselect the "score" column
        sel_list = pilot.app.query_one("#showColumns SelectionList", SelectionList)
        sel_list.deselect("score")
        await pilot.pause()
        # Apply changes
        await pilot.press("ctrl+a")
        await pilot.pause()

    assert snap_compare(_make_app(), run_before=run_before, terminal_size=(120, 30))


def test_snap_narrow_terminal(snap_compare):
    """Snapshot at a narrow terminal width showing fewer columns."""
    assert snap_compare(_make_app(), terminal_size=(60, 30))


def test_snap_row_detail(snap_compare):
    """Snapshot with row detail panel visible (default)."""
    assert snap_compare(
        _make_app(),
        press=["down", "down"],
        terminal_size=(120, 30),
    )
