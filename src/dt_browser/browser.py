import asyncio
import pathlib
from typing import ClassVar

import click
import polars as pl
from rich.spinner import Spinner
from rich.style import Style
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.cache import LRUCache
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Footer, Label, Static
from textual_fastdatatable import DataTable

from dt_browser import (
    COLOR_COL,
    COLORS,
    COLORS_STYLES,
    INDEX_COL,
    ReactiveLabel,
    ReceivesTableSelect,
    SelectFromTable,
)
from dt_browser.bookmarks import Bookmarks
from dt_browser.column_selector import ColumnSelector
from dt_browser.filter_box import FilterBox
from dt_browser.polars_backend import PolarsBackend
from dt_browser.suggestor import ColumnNameSuggestor

_SHOW_COLUMNS_ID = "showColumns"
_COLOR_COLUMNS_ID = "colorColumns"


class ExtendedDataTable(DataTable):

    DEFAULT_CSS = (
        DataTable.DEFAULT_CSS
        + COLORS_STYLES
        + """

ExtendedDataTable > .datatable--row-bookmark {
    background: $error-lighten-3;
}
"""
    )

    COMPONENT_CLASSES: ClassVar[set[str]] = DataTable.COMPONENT_CLASSES.union(
        [f"datatable--row{i}" for i in range(len(COLORS.categories))]
    ).union(["datatable--row-bookmark"])

    def __init__(self, *args, metadata_dt: pl.DataFrame, bookmarks: Bookmarks, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_dt = metadata_dt
        self._bookmarks = bookmarks
        self.styles.height = "1fr"

    def _get_row_style(self, row_index: int, base_style: Style) -> Style:
        style = super()._get_row_style(row_index, base_style)
        styles = [style]
        if row_index >= 0 and COLOR_COL in self.meta_dt.columns:
            color_idx = self.meta_dt[COLOR_COL][row_index]
            styles.append(self.get_component_styles(f"datatable--row{color_idx}").rich_style)

        if not self._bookmarks._meta_dt.is_empty():
            idx = self.meta_dt[row_index][INDEX_COL][0]
            if idx in self._bookmarks._meta_dt[INDEX_COL]:
                styles.append(self.get_component_styles(f"datatable--row-bookmark").rich_style)
        return Style.combine(styles)


class SpinnerWidget(Static):
    def __init__(self, style: str):
        super().__init__("")
        self._spinner = Spinner(style)
        self.styles.width = 1

    def on_mount(self) -> None:
        self.update_render = self.set_interval(1 / 60, self.update_spinner)

    def update_spinner(self) -> None:
        self.update(self._spinner)


class TableFooter(Footer):
    DEFAULT_CSS = """
    TableFooter > .tablefooter--rowcount {
        background: $accent-darken-1;
        width: auto;
        padding: 0 2;
    }
    TableFooter > .tablefooter--search {
        background: $success-darken-1;
        width: auto;
        padding: 0 2;
    }
    TableFooter > .tablefooter--pending {
        background: $secondary-darken-1;
        width: auto;
        padding: 0 2;
    }

    FooterRowCount > .tablefooter--label {
        background: $secondary;
        text-style: bold;
    }

    """
    filter_pending = reactive(False, recompose=True)
    is_filtered = reactive(False)
    cur_row = reactive(1)
    cur_total_rows = reactive(0)
    total_rows = reactive(0)
    total_rows_display = reactive("", layout=True)

    pending_action = reactive("", recompose=True)

    search_pending: reactive[bool] = reactive(False, recompose=True)
    active_search_queue: reactive[list[int] | None] = reactive(None)
    active_search_idx: reactive[int | None] = reactive(None)
    active_search_idx_display: reactive[int | None] = reactive(None)
    active_search_len: reactive[int | None] = reactive(None, recompose=True)

    def compute_active_search_len(self):
        if self.active_search_queue is None:
            return None
        return len(self.active_search_queue)

    def compose(self):
        yield from super().compose()

        widths = ["auto"] * self.styles.grid_size_columns
        yield Label()
        widths.append("1fr")
        if self.pending_action:
            widths.append("auto")
            with Horizontal(classes="tablefooter--pending"):
                yield ReactiveLabel().data_bind(value=TableFooter.pending_action)
                yield Label(" ")
                yield SpinnerWidget("dots")
        if self.search_pending:
            widths.append("auto")
            with Horizontal(classes="tablefooter--search"):
                yield Label("Searching ")
                yield SpinnerWidget("dots")
        elif self.active_search_len is not None:
            widths.append("auto")
            with Horizontal(classes="tablefooter--search"):
                yield Label("Search: ")
                yield ReactiveLabel().data_bind(value=TableFooter.active_search_idx)
                yield Label(" / ")
                yield ReactiveLabel().data_bind(value=TableFooter.active_search_len)

        with Horizontal(classes="tablefooter--rowcount"):
            yield ReactiveLabel().data_bind(value=TableFooter.cur_row)
            yield Label(" / ")
            yield ReactiveLabel().data_bind(value=TableFooter.cur_total_rows)
            yield ReactiveLabel().data_bind(value=TableFooter.total_rows_display)
            if self.filter_pending:
                yield Label(" Filtering ")
                yield SpinnerWidget("dots")

        widths.append("auto")
        self.styles.grid_columns = " ".join(widths)
        self.styles.grid_size_columns = len(widths)

    def compute_total_rows_display(self):
        return f" (Filtered from {self.total_rows:,})" if self.cur_total_rows != self.total_rows else ""

    def compute_search_idx_display(self):
        return self.active_search_idx + 1


class RowDetail(Widget, can_focus=False, can_focus_children=False):

    DEFAULT_CSS = """
RowDetail {
    width: auto;
    padding: 0 1;
    border: tall $accent;
}
"""
    row_df = reactive(pl.DataFrame(), recompose=True, always_update=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "Row Detail"

    def compose(self):
        if self.row_df.is_empty():
            return
        dt = DataTable(
            backend=PolarsBackend.from_dataframe(
                self.row_df.with_columns(
                    [
                        pl.col(col).cast(pl.Utf8) if dtype == pl.Categorical else pl.col(col)
                        for col, dtype in zip(self.row_df.columns, self.row_df.dtypes)
                    ]
                ).transpose(include_header=True, header_name="Field", column_names=["Value"])
            ),
            show_cursor=False,
            show_header=False,
            show_row_labels=False,
        )
        dt.styles.width = "auto"
        yield dt


class DtBrowser(App):  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """A Textual app to manage stopwatches."""

    BINDINGS = [
        ("f", "show_filter", "Filter rows"),
        ("/", "show_search", "Search"),
        ("n", "iter_search(True)", "Next"),
        Binding("N", "iter_search(False)", "Prev", key_display="shift+N"),
        ("b", "toggle_bookmark", "Add/Del Bookmark"),
        Binding("B", "show_bookmarks", "Bookmarks", key_display="shift+B"),
        ("c", "column_selector", "Columns..."),
        ("r", "toggle_row_detail", "Toggle Row Detail"),
        Binding("C", "show_colors", "Colors...", key_display="shift+C"),
    ]

    color_by: reactive[tuple[str, ...]] = reactive(tuple(), init=False)
    visible_columns: reactive[tuple[str, ...]] = reactive(tuple())
    all_columns: reactive[tuple[str, ...]] = reactive(tuple())
    is_filtered = reactive(False)
    cur_row = reactive(1)
    cur_total_rows = reactive(0)
    total_rows = reactive(0)

    show_row_detail = reactive(True)

    active_search_queue: reactive[list[int] | None] = reactive(None)
    active_search_idx: reactive[int | None] = reactive(None)
    active_search: reactive[str | None] = reactive(None)
    # active_dt: reactive[pl.DataFrame] = reactive(pl.DataFrame(), init=False, always_update=True)

    def __init__(self, table_name: str, source_file: pathlib.Path):
        super().__init__()
        self._source = source_file

        self._display_dt = self._filtered_dt = self._original_dt = PolarsBackend.from_file_path(self._source).data
        self._meta_dt = self._original_meta = self._original_dt.with_row_index(name=INDEX_COL).select([INDEX_COL])
        self._table_name = table_name
        self._bookmarks = Bookmarks()
        self._suggestor = ColumnNameSuggestor()
        self._backend = PolarsBackend.from_dataframe(self._original_dt)
        self.visible_columns = tuple(self._backend.columns)
        self.all_columns = self.visible_columns
        self._filter_box = FilterBox(suggestor=self._suggestor, id="filter", classes="toolbox")
        self._select_interest: str | None = None
        self._column_selector = ColumnSelector(id=_SHOW_COLUMNS_ID, title="Show/Hide/Reorder Columns")
        self._color_selector = ColumnSelector(
            allow_reorder=False, id=_COLOR_COLUMNS_ID, title="Select columns to color by"
        )
        self._row_detail = RowDetail()

        # self.set_reactive(DtBrowser.color_by, self._backend.columns[0:1])

        self._color_by_cache: LRUCache[tuple[str, ...], pl.Series] = LRUCache(5)

    # def save_state(self):
    #     return {
    #         "version": 1,
    #         "name": self._table_name,
    #         "bookmarks": self._bookmarks.save_state(),
    #         "filters": self._filter_box.save_state(),
    #         "cols": self._column_selector.save_state(),
    #     }

    # def load_state(self, state: dict):
    #     self._table_name = state["name"]
    #     self._bookmarks.load_state(state["bookmarks"], self._table_name, self._backend.data)
    #     self._filter_box.load_state(state["filters"], self._table_name, self._backend.data)
    #     self._column_selector.load_state(state["cols"], self._table_name, self._backend.data)

    def watch_visible_columns(self):
        self._suggestor.columns = self.visible_columns

    @on(FilterBox.FilterSubmitted)
    @work(exclusive=True)
    async def apply_filter(self, event: FilterBox.FilterSubmitted):
        if not event.value:
            self.is_filtered = False
            idx = self.query_one(ExtendedDataTable).cursor_coordinate.row
            await self._set_filtered_dt(
                self._original_dt,
                self._original_meta,
                new_row=self._meta_dt[INDEX_COL][idx],
                focus=False,
            )
        else:
            (foot := self.query_one(TableFooter)).filter_pending = True
            ctx = pl.SQLContext(frames={"dt": pl.concat([self._original_dt, self._original_meta], how="horizontal")})
            try:
                dt = await ctx.execute(f"select * from dt where {event.value}").collect_async()
                meta = dt.select([x for x in dt.columns if x.startswith("__")])
                dt = dt.select([x for x in dt.columns if not x.startswith("__")])
                self.is_filtered = True
                if dt.is_empty():
                    self.notify(f"No results found for filter: {event.value}", severity="warn", timeout=5)
                else:
                    await self._set_filtered_dt(dt, meta, new_row=0, focus=None)
            except Exception as e:
                self.query_one(FilterBox).query_failed(event.value)
                self.notify(f"Failed to apply filter due to: {e}", severity="error", timeout=10)
            foot.filter_pending = False

    @on(FilterBox.GoToSubmitted)
    async def apply_search(self, event: FilterBox.GoToSubmitted):
        self.active_search = event.value

    @work(exclusive=True)
    async def watch_active_search(self):
        if not self.active_search:
            self.active_search_queue = None
            self.active_search_idx = 0
            return

        (foot := self.query_one(TableFooter)).search_pending = True
        try:
            ctx = pl.SQLContext(frames={"dt": (pl.concat([self._display_dt, self._meta_dt], how="horizontal"))})
            search_queue = list(
                (await ctx.execute(f"select {INDEX_COL} from dt where {self.active_search}").collect_async())[INDEX_COL]
            )

            foot.search_pending = False
            if not search_queue:
                self.notify("No results found for search", severity="warn", timeout=5)
            else:
                self.active_search_queue = search_queue
                self.active_search_idx = -1
                self.action_iter_search(True)
        except Exception as e:
            self.query_one(FilterBox).query_failed(self.active_search)
            self.notify(f"Failed to run search due to: {e}", severity="error", timeout=10)
            foot.search_pending = False

    def action_iter_search(self, forward: bool):
        table = self.query_one(ExtendedDataTable)
        coord = table.cursor_coordinate
        self.active_search_idx += 1 if forward else -1
        if self.active_search_idx >= 0 and self.active_search_idx < len(self.active_search_queue):
            next_idex = self.active_search_queue[self.active_search_idx]
            ys = next_idex
            xs = table.scroll_x
            table.scroll_to(xs, ys, animate=False, force=True)
            table.move_cursor(column=coord.column, row=next_idex)
        self.refresh_bindings()

    def action_toggle_bookmark(self):
        row_idx = self.query_one(ExtendedDataTable).cursor_coordinate.row
        did_add = self._bookmarks.toggle_bookmark(self._display_dt[row_idx], self._meta_dt[row_idx])
        (dt := self.query_one(ExtendedDataTable))._clear_caches()
        dt.refresh_row(row_idx)
        self.refresh_bindings()
        self.notify("Bookmark added!" if did_add else "Bookmark removed", severity="information", timeout=3)

    async def action_toggle_row_detail(self):
        self.show_row_detail = not self.show_row_detail

    async def watch_show_row_detail(self):
        if not self.show_row_detail:
            if existing := self.query(RowDetail):
                existing.remove()
        else:
            await self.query_one("#main_hori", Horizontal).mount(self._row_detail)

    async def action_show_bookmarks(self):
        await self.mount(self._bookmarks, before=self.query_one(TableFooter))

    async def action_column_selector(self):
        self._column_selector.data_bind(
            selected_columns=DtBrowser.visible_columns, available_columns=DtBrowser.all_columns
        )
        await self.query_one("#main_hori", Horizontal).mount(self._column_selector)

    async def action_show_colors(self):
        self._color_selector.data_bind(selected_columns=DtBrowser.color_by, available_columns=DtBrowser.all_columns)
        await self.query_one("#main_hori", Horizontal).mount(self._color_selector)

    async def _set_filtered_dt(self, filtered_dt: pl.DataFrame, filtered_meta: pl.DataFrame, **kwargs):
        self._filtered_dt = filtered_dt
        self._meta_dt = filtered_meta
        await self._set_active_dt(self._filtered_dt, **kwargs)

    async def _set_active_dt(self, active_dt: pl.DataFrame, **kwargs):
        self._display_dt = active_dt.select(self.visible_columns)
        self.cur_total_rows = len(self._display_dt)
        self.watch_active_search.__wrapped__(self)
        await self._redraw(**kwargs)

    @on(ColumnSelector.ColumnSelectionChanged, f"#{_SHOW_COLUMNS_ID}")
    async def reorder_columns(self, event: ColumnSelector.ColumnSelectionChanged):
        self.visible_columns = tuple(event.selected_columns)
        await self._set_active_dt(self._filtered_dt, focus=False)

    @on(ColumnSelector.ColumnSelectionChanged, f"#{_COLOR_COLUMNS_ID}")
    async def set_color_by(self, event: ColumnSelector.ColumnSelectionChanged):
        self.color_by = tuple(event.selected_columns)

    @on(SelectFromTable)
    def enable_select_from_table(self, event: SelectFromTable):
        self._select_interest = f"#{event.interested_widget.id}"
        self.query_one(ExtendedDataTable).focus()

    @on(ExtendedDataTable.CellHighlighted)
    async def handle_cell_highlight(self, event: ExtendedDataTable.CellHighlighted):
        self.cur_row = event.coordinate.row
        self._row_detail.row_df = self._display_dt[self.cur_row]

    @on(ExtendedDataTable.CellSelected)
    def handle_cell_select(self, event: ExtendedDataTable.CellSelected):
        if self._select_interest:
            self.query_one(self._select_interest, ReceivesTableSelect).on_table_select(event.value)
            self._select_interest = None

    @on(Bookmarks.BookmarkSelected)
    def handle_bookmark_select(self, event: Bookmarks.BookmarkSelected):
        dt = self.query_one(ExtendedDataTable)
        coord = dt.cursor_coordinate
        sel_idx = event.selected_index
        if self.is_filtered:
            filt = self._meta_dt.with_row_index("__displayIndex").filter(pl.col(INDEX_COL) == sel_idx)
            if filt.is_empty():
                self.notify(
                    "Bookmark not present in filtered view.  Remove filters to select this bookmark",
                    severity="error",
                    timeout=5,
                )
                return
            sel_idx = filt["__displayIndex"][0]
        ys = sel_idx
        xs = dt.scroll_x
        dt.scroll_to(xs, ys, animate=False, force=True)
        dt.move_cursor(column=coord.column, row=sel_idx)

    @on(Bookmarks.BookmarkRemoved)
    def handle_bookmark_removed(self, event: Bookmarks.BookmarkRemoved):
        self.query_one(ExtendedDataTable)._clear_caches()
        self.refresh_bindings()

    async def action_show_filter(self):
        if existing := self.query("#filter"):
            existing.remove()
            return
        self._filter_box.is_goto = False
        await self.mount(self._filter_box, before=self.query_one(TableFooter))

    async def action_show_search(self):
        if existing := self.query("#filter"):
            existing.remove()
            return
        self._filter_box.is_goto = True
        await self.mount(self._filter_box, before=self.query_one(TableFooter))

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if an action may run."""
        # self.query(".toolbox")
        if not (edtq := self.query(ExtendedDataTable)):
            return False

        if not edtq.only_one().has_focus and action in (
            x.action if isinstance(x, Binding) else x[1] for x in DtBrowser.BINDINGS
        ):
            return False

        if action == "iter_search":
            if not self.active_search_queue:
                return False
            if bool(parameters[0]) and self.active_search_idx == len(self.active_search_queue) - 1:
                return False
            if not bool(parameters[0]) and self.active_search_idx == 0:
                return False
        if action == "show_bookmarks":
            return self._bookmarks.has_bookmarks

        return True

    @work(exclusive=True)
    async def watch_color_by(self):
        if not self.color_by:
            self._meta_dt = self._meta_dt.drop(COLOR_COL, strict=False)
            self._original_meta = self._original_meta.drop(COLOR_COL, strict=False)
        else:
            (foot := self.query_one(TableFooter)).pending_action = "Recoloring"
            try:
                cols = tuple(sorted(self.color_by))
                if cols not in self._color_by_cache:
                    self._color_by_cache.set(
                        cols,
                        (
                            await self._original_dt.lazy()
                            .with_columns(
                                __color=(
                                    (pl.any_horizontal(*(pl.col(x) != pl.col(x).shift(1) for x in cols)))
                                    .cum_sum()
                                    .fill_null(0)
                                    % len(COLORS.categories)
                                )
                            )
                            .collect_async()
                        )[COLOR_COL],
                    )
                self._original_meta = self._original_meta.with_columns(__color=self._color_by_cache.get(cols))
                self._meta_dt = (
                    await self._meta_dt.lazy()
                    .drop(COLOR_COL, strict=False)
                    .join(self._original_meta.lazy().select([INDEX_COL, COLOR_COL]), how="left", on=INDEX_COL)
                    .collect_async()
                )
            except Exception as e:
                self.notify(f"Failed to apply coloring due to: {e}", severity="error", timeout=10)
            foot.pending_action = None

        await self._redraw(focus=None)

    async def _redraw(self, new_row: int | None = None, focus: bool | None = True):
        self._backend = PolarsBackend.from_dataframe(self._display_dt)
        existing_q = self.query(ExtendedDataTable)
        if not existing_q:
            return
        existing = existing_q.only_one()
        if focus is None:
            focus = existing.has_focus
        coord = existing.cursor_coordinate
        ys = new_row if new_row is not None else existing.scroll_y
        xs = existing.scroll_x
        await existing.remove()
        dt = ExtendedDataTable(backend=self._backend, id="table", metadata_dt=self._meta_dt, bookmarks=self._bookmarks)
        await self.query_one("#main_hori", Horizontal).mount(dt, before=0)

        if focus:
            dt.focus()
        with self.app.batch_update():
            dt.scroll_to(xs, ys, animate=False, force=True)
            dt.move_cursor(column=coord.column, row=new_row if new_row is not None else coord.row)

        if new_row is not None:
            self.cur_row = new_row

    def on_mount(self):
        self.app._driver._disable_mouse_support()
        # self.color_by = tuple(self._backend.columns[0:1])
        self.cur_total_rows = len(self._display_dt)
        self.total_rows = len(self._original_dt)
        self.cur_row = 1
        self._row_detail.row_df = self._display_dt[0]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        with Horizontal(id="main_hori"):
            yield ExtendedDataTable(
                backend=self._backend, id="table", metadata_dt=self._meta_dt, bookmarks=self._bookmarks
            )
        yield TableFooter().data_bind(
            DtBrowser.cur_row,
            DtBrowser.cur_total_rows,
            DtBrowser.is_filtered,
            DtBrowser.total_rows,
            DtBrowser.active_search_queue,
            DtBrowser.active_search_idx,
        )


@click.command()
@click.argument("source_file", nargs=1, required=True, type=pathlib.Path)
def run(source_file: pathlib.Path):
    app = DtBrowser(source_file, source_file)
    app.run()


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
