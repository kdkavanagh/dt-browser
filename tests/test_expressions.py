import polars as pl

from dt_browser.browser import DtBrowser, DtBrowserApp
from dt_browser.custom_table import CustomTable
from dt_browser.expression_box import ExpressionBox
from dt_browser.filter_box import FilterBox


def _make_app() -> DtBrowserApp:
    df = pl.DataFrame({"name": ["alice", "bob", "charlie", "dave"], "value": [10, 20, 30, 40]})
    return DtBrowserApp("test", df)


async def _submit_expression(pilot, app, expr: str):
    """Type an expression in the already-open expression box and submit."""
    from textual.widgets import Input

    inp = app.query_one("ExpressionBox Input", Input)
    inp.value = ""
    await pilot.pause()
    await pilot.press(*list(expr))
    await pilot.press("enter")
    await pilot.pause()
    # Allow the @work to complete
    await pilot.pause()


async def test_compute_new_column():
    """Computing an expression adds a new column to the table."""
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)
        initial_cols = browser.visible_columns

        await pilot.press("x")
        await pilot.pause()
        await _submit_expression(pilot, app, "doubled = value * 2")

        assert "doubled" in browser.visible_columns
        assert "doubled" in browser.all_columns
        assert len(browser.visible_columns) == len(initial_cols) + 1

        # Verify the computed values are correct
        doubled_col = browser._original_dt["doubled"]
        assert doubled_col.to_list() == [20, 40, 60, 80]


async def test_computed_column_usable_in_filter():
    """A computed column can be used in a subsequent filter."""
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)

        # Add computed column
        await pilot.press("x")
        await pilot.pause()
        await _submit_expression(pilot, app, "doubled = value * 2")
        assert "doubled" in browser.all_columns

        # Close expression box, then filter using the computed column
        await pilot.press("escape")
        await pilot.pause()

        # Focus should be back on main table for "f" to work
        app.query_one("#main_table", CustomTable).focus()
        await pilot.pause()

        await pilot.press("f")
        await pilot.pause()
        await pilot.press(*list("doubled > 40"))
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert browser.is_filtered
        # doubled > 40 means value > 20, so charlie(60) and dave(80)
        assert browser.cur_total_rows == 2


async def test_computed_column_as_input_for_further_expression():
    """A computed column can be referenced in a subsequent expression."""
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)

        # First computed column
        await pilot.press("x")
        await pilot.pause()
        await _submit_expression(pilot, app, "doubled = value * 2")
        assert "doubled" in browser.all_columns

        # Second computed column referencing the first (expression box still open)
        await _submit_expression(pilot, app, "quadrupled = doubled * 2")
        assert "quadrupled" in browser.all_columns

        # Verify values
        assert browser._original_dt["quadrupled"].to_list() == [40, 80, 120, 160]


async def test_expression_box_closes_on_escape():
    """Expression box closes when escape is pressed."""
    app = _make_app()
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()

        await pilot.press("x")
        await pilot.pause()
        assert len(app.query(ExpressionBox)) == 1

        await pilot.press("escape")
        await pilot.pause()
        assert len(app.query(ExpressionBox)) == 0
