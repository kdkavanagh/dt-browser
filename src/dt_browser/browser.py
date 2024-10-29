import datetime
import gc
import pathlib
import time
from typing import ClassVar

import click
import polars as pl
import tzlocal
from rich.spinner import Spinner
from rich.style import Style
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.cache import LRUCache
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.timer import Timer
from textual.widget import Widget
from textual.widgets import Footer, Label, Static

from dt_browser import (
    COLOR_COL,
    COLORS,
    INDEX_COL,
    ReactiveLabel,
    ReceivesTableSelect,
    SelectFromTable,
)
from dt_browser.bookmarks import Bookmarks
from dt_browser.column_selector import ColumnSelector
from dt_browser.custom_table import CustomTable
from dt_browser.filter_box import FilterBox
from dt_browser.suggestor import ColumnNameSuggestor

_SHOW_COLUMNS_ID = "showColumns"
_COLOR_COLUMNS_ID = "colorColumns"
_TS_COLUMNS_ID = "tsColumns"

_TIMEZONE = str(tzlocal.get_localzone())


class TableWithBookmarks(CustomTable):

    DEFAULT_CSS = (
        CustomTable.DEFAULT_CSS
        + """
TableWithBookmarks > .datatable--row-bookmark {
    background: $error-lighten-3;
}

TableWithBookmarks > .datatable--row-search-result {
    background: $surface-lighten-3;
}
"""
    )

    COMPONENT_CLASSES: ClassVar[set[str]] = CustomTable.COMPONENT_CLASSES.union(
        ["datatable--row-bookmark", "datatable--row-search-result"]
    )

    active_search_queue: reactive[list[int] | None] = reactive(None)

    def __init__(self, *args, bookmarks: Bookmarks, **kwargs):
        super().__init__(*args, **kwargs)
        self._bookmarks = bookmarks
        self._bookmark_highlight: Style | None = None
        self._search_highlight: Style | None = None

    def on_mount(self):
        self._bookmark_highlight = self.get_component_rich_style("datatable--row-bookmark")
        self._search_highlight = self.get_component_rich_style("datatable--row-search-result")

    def _get_sel_col_bg_color(self, struct: pl.Struct):
        if self.active_search_queue and struct[INDEX_COL] in self.active_search_queue:
            return self._search_highlight.bgcolor.name
        if self._bookmarks.has_bookmarks and struct[INDEX_COL] in self._bookmarks.meta_dt[INDEX_COL]:
            return self._bookmark_highlight.bgcolor.name
        return super()._get_sel_col_bg_color(struct)

    def _get_row_bg_color_expr(self, cursor_row_idx: int) -> pl.Expr:
        tmp = super()._get_row_bg_color_expr(cursor_row_idx)
        if self.active_search_queue:
            tmp = (
                pl.when(pl.col(INDEX_COL).is_in(self.active_search_queue))
                .then(pl.lit(self._search_highlight.bgcolor.name))
                .otherwise(tmp)
            )
        if self._bookmarks.has_bookmarks:
            tmp = (
                pl.when(pl.col(INDEX_COL).is_in(self._bookmarks.meta_dt[INDEX_COL]))
                .then(pl.lit(self._bookmark_highlight.bgcolor.name))
                .otherwise(tmp)
            )
        return tmp


_ALREADY_DT = "dt"


def _guess_timestamp_cols(df: pl.DataFrame):
    date_range = pl.Series(values=[datetime.date(2001, 1, 1), datetime.date(2042, 1, 1)])
    converts = [(x,) + tuple(date_range.dt.epoch(x)) for x in ("s", "ms", "us", "ns")]

    for col, dtype in df.schema.items():
        if dtype.is_integer():
            for suffix, min_val, max_val in converts:
                all_in_range = (
                    df.lazy()
                    .select(is_zero=pl.col(col) == 0, is_inside=((pl.col(col) >= min_val) & (pl.col(col) <= max_val)))
                    .select(
                        count=pl.col("is_inside").any()
                        & (pl.any_horizontal(pl.col("is_zero"), pl.col("is_inside")).sum() == pl.len())
                    )
                    .collect()
                    .get_column("count")[0]
                )
                if all_in_range:
                    yield (col, suffix)
                    break
        elif dtype.is_temporal():
            yield (col, _ALREADY_DT)


class SpinnerWidget(Static):
    def __init__(self, style: str):
        super().__init__("")
        self._spinner = Spinner(style)
        self.styles.width = 1
        self.update_render: Timer | None = None

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
    cur_row = reactive(0)
    cur_total_rows = reactive(0)
    total_rows = reactive(0)
    total_rows_display = reactive("", layout=True)

    cur_row_display = reactive(0)

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

    def compute_cur_row_display(self):
        return self.cur_row + 1

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
            yield ReactiveLabel().data_bind(value=TableFooter.cur_row_display)
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
    row_df = reactive(pl.DataFrame(), always_update=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "Row Detail"
        self._dt = CustomTable(
            pl.DataFrame(),
            pl.DataFrame().with_row_index(name=INDEX_COL).select([INDEX_COL]),
            cursor_type=CustomTable.CursorType.NONE,
        )
        self.styles.width = "auto"

    def watch_row_df(self):
        if self.row_df.is_empty():
            return

        display_df = self.row_df.with_columns(
            [
                pl.col(col).cast(pl.Utf8) if dtype == pl.Categorical else pl.col(col)
                for col, dtype in zip(self.row_df.columns, self.row_df.dtypes)
            ]
        ).transpose(include_header=True, header_name="Field", column_names=["Value"])
        self._dt.set_dt(display_df, display_df.with_row_index(name=INDEX_COL).select([INDEX_COL]))
        self._dt.styles.width = (
            display_df["Field"].str.len_chars().max() + display_df["Value"].str.len_chars().max() + 3
        )
        self._dt.refresh()

    def compose(self):
        yield self._dt


def from_file_path(path: pathlib.Path, has_header: bool = True) -> pl.DataFrame:

    if path.suffix in [".arrow", ".feather"]:
        return pl.read_ipc(path)
    if path.suffix in [".arrows", ".arrowstream"]:
        return pl.read_ipc_stream(path)
    if path.suffix == ".json":
        return pl.read_json(path)
    if path.suffix == ".csv":
        return pl.read_csv(path, has_header=has_header)
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    raise TypeError(f"Dont know how to load file type {path.suffix} for {path}")


class DtBrowser(Widget):  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """A Textual app to manage stopwatches."""

    BINDINGS = [
        ("f", "show_filter", "Filter rows"),
        ("/", "show_search", "Search"),
        ("n", "iter_search(True)", "Next"),
        Binding("N", "iter_search(False)", "Prev", key_display="shift+N"),
        ("b", "toggle_bookmark", "Add/Del Bookmark"),
        Binding("B", "show_bookmarks", "Bookmarks", key_display="shift+B"),
        Binding("c", "column_selector", "Columns..."),
        ("t", "timestamp_selector", "Timestamps..."),
        ("r", "toggle_row_detail", "Toggle Row Detail"),
        Binding("C", "show_colors", "Colors...", key_display="shift+C"),
    ]

    color_by: reactive[tuple[str, ...]] = reactive(tuple(), init=False)
    visible_columns: reactive[tuple[str, ...]] = reactive(tuple())
    all_columns: reactive[tuple[str, ...]] = reactive(tuple())
    timestamp_columns: reactive[tuple[str, ...]] = reactive(tuple())
    available_timestamp_columns: reactive[tuple[str, ...]] = reactive(tuple())

    is_filtered = reactive(False)
    current_filter = reactive[str | None](None)

    cur_row = reactive(0)
    cur_total_rows = reactive(0)
    total_rows = reactive(0)

    show_row_detail = reactive(True)

    active_search_queue: reactive[list[int] | None] = reactive(None)
    active_search_idx: reactive[int | None] = reactive(None)
    active_search: reactive[str | None] = reactive(None)
    # active_dt: reactive[pl.DataFrame] = reactive(pl.DataFrame(), init=False, always_update=True)

    def __init__(self, table_name: str, source_file_or_table: pathlib.Path | pl.DataFrame):
        super().__init__()
        bt = (
            from_file_path(source_file_or_table)
            if isinstance(source_file_or_table, (str, pathlib.Path))
            else source_file_or_table
        )
        # bt = bt.with_columns(TestTs=datetime.datetime.now())
        self._display_dt = self._filtered_dt = self._original_dt = bt
        self._meta_dt = self._original_meta = self._original_dt.with_row_index(name=INDEX_COL).select([INDEX_COL])
        self._table_name = table_name
        self._bookmarks = Bookmarks()
        self._suggestor = ColumnNameSuggestor()
        self.visible_columns = tuple(self._original_dt.columns)
        self.all_columns = self.visible_columns
        self._filter_box = FilterBox(suggestor=self._suggestor, id="filter", classes="toolbox")
        self._select_interest: str | None = None
        self._column_selector = ColumnSelector(id=_SHOW_COLUMNS_ID, title="Show/Hide/Reorder Columns")
        self._color_selector = ColumnSelector(
            allow_reorder=False, id=_COLOR_COLUMNS_ID, title="Select columns to color by"
        )

        self._ts_cols = dict(_guess_timestamp_cols(self._original_dt))
        self._ts_col_selector = ColumnSelector(id=_TS_COLUMNS_ID, title="Select epoch timestamp columns")
        self._ts_col_names: dict[str, str] = {}
        self.available_timestamp_columns = tuple(self._ts_cols.keys())

        # Necessary to prevent the main table from resizing to 0 when the col selectors are mounted and then immediately resizing
        # (apparently that happens when col selector width = auto)
        self._color_selector.styles.width = 1
        self._column_selector.styles.width = 1
        self._ts_col_selector.styles.width = 1

        self._row_detail = RowDetail()

        self._color_by_cache: LRUCache[tuple[str, ...], pl.Series] = LRUCache(5)
        self._last_message_ts = time.time()

    def _set_last_message(self, *_):
        self._last_message_ts = time.time()

    def _maybe_gc(self):
        if time.time() - self._last_message_ts > 3:
            self.app.log("Triggering GC!")
            gc.collect()

    def watch_visible_columns(self):
        self._suggestor.columns = self.visible_columns

    @on(FilterBox.FilterSubmitted)
    async def update_filter(self, event: FilterBox.FilterSubmitted):
        self.current_filter = event.value
        self.apply_filter()

    @work(exclusive=True)
    async def apply_filter(self):
        if not self.current_filter:
            self.is_filtered = False
            idx = self.query_one("#main_table", CustomTable).cursor_coordinate.row
            self._set_filtered_dt(
                self._original_dt,
                self._original_meta,
                new_row=self._meta_dt[INDEX_COL][idx],
            )
        else:
            (foot := self.query_one(TableFooter)).filter_pending = True
            ctx = pl.SQLContext(frames={"dt": pl.concat([self._original_dt, self._original_meta], how="horizontal")})
            try:
                query = self.current_filter.replace(" && ", " and ").replace(" || ", " or ")
                dt = await ctx.execute(f"select * from dt where {query}").collect_async()
                meta = dt.select([x for x in dt.columns if x.startswith("__")])
                dt = dt.select([x for x in dt.columns if not x.startswith("__")])
                self.is_filtered = True
                if dt.is_empty():
                    self.notify(f"No results found for filter: {query}", severity="warn", timeout=5)
                else:
                    self._set_filtered_dt(dt, meta, new_row=0)
            except Exception as e:
                self.query_one(FilterBox).query_failed(query)
                self.notify(f"Failed to apply filter due to: {e}", severity="error", timeout=10)
                self.current_filter = None
            foot.filter_pending = False

    @on(FilterBox.GoToSubmitted)
    async def apply_search(self, event: FilterBox.GoToSubmitted):
        self.active_search = event.value

    @work(exclusive=True)
    async def watch_active_search(self, goto: bool = True):
        if not self.active_search:
            self.active_search_queue = None
            self.active_search_idx = 0
            return

        (foot := self.query_one(TableFooter)).search_pending = True
        try:
            ctx = pl.SQLContext(frames={"dt": (pl.concat([self._display_dt, self._meta_dt], how="horizontal"))})
            query = self.active_search.replace(" && ", " and ").replace(" || ", " or ")
            search_queue = list(
                (await ctx.execute(f"select {INDEX_COL} from dt where {query}").collect_async())[INDEX_COL]
            )

            foot.search_pending = False
            if not search_queue:
                self.notify("No results found for search", severity="warn", timeout=5)
            else:
                self.active_search_queue = search_queue
                self.active_search_idx = -1
                if goto:
                    self.action_iter_search(True)
        except Exception as e:
            self.query_one(FilterBox).query_failed(query)
            self.notify(f"Failed to run search due to: {e}", severity="error", timeout=10)
            foot.search_pending = False

    def action_iter_search(self, forward: bool):
        table = self.query_one("#main_table", CustomTable)
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
        row_idx = self.query_one("#main_table", CustomTable).cursor_coordinate.row
        did_add = self._bookmarks.toggle_bookmark(self._display_dt[row_idx], self._meta_dt[row_idx])
        self.refresh_bindings()
        self.query_one("#main_table", CustomTable).refresh(repaint=True)

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
        await self.query_one("#main_hori", Horizontal).mount(self._column_selector)

    async def action_show_colors(self):

        await self.query_one("#main_hori", Horizontal).mount(self._color_selector)

    async def action_timestamp_selector(self):
        await self.query_one("#main_hori", Horizontal).mount(self._ts_col_selector)

    def _set_filtered_dt(self, filtered_dt: pl.DataFrame, filtered_meta: pl.DataFrame, **kwargs):
        self._filtered_dt = filtered_dt
        self._meta_dt = filtered_meta
        self._set_active_dt(self._filtered_dt, **kwargs)

    def _set_active_dt(self, active_dt: pl.DataFrame, new_row: int | None = None):
        ordered_cols: list[pl.Expr] = []
        for col in self.visible_columns:
            ordered_cols.append(pl.col(col))
            if col in self._ts_col_names:
                ordered_cols.append(pl.col(self._ts_col_names[col]))
        active_dt = active_dt.select(ordered_cols)

        self._display_dt = active_dt
        self.cur_total_rows = len(self._display_dt)
        self.watch_active_search(goto=False)
        (table := self.query_one("#main_table", CustomTable)).set_dt(self._display_dt, self._meta_dt)
        if new_row is not None:
            table.move_cursor(row=new_row, column=None)
            self.cur_row = new_row

    @on(ColumnSelector.ColumnSelectionChanged, f"#{_SHOW_COLUMNS_ID}")
    def reorder_columns(self, event: ColumnSelector.ColumnSelectionChanged):
        self.visible_columns = tuple(event.selected_columns)
        self._set_active_dt(self._filtered_dt)

    @on(ColumnSelector.ColumnSelectionChanged, f"#{_COLOR_COLUMNS_ID}")
    async def set_color_by(self, event: ColumnSelector.ColumnSelectionChanged):
        self.color_by = tuple(event.selected_columns)

    @on(ColumnSelector.ColumnSelectionChanged, f"#{_TS_COLUMNS_ID}")
    async def set_timestamp_cols(self, event: ColumnSelector.ColumnSelectionChanged):
        self.timestamp_columns = tuple(event.selected_columns)

    @work(exclusive=True)
    async def watch_timestamp_columns(self):
        old_cols = [v for k, v in self._ts_col_names.items() if k not in self.timestamp_columns]
        self._original_dt = self._original_dt.drop(old_cols)
        self._ts_col_names = {
            x: f"{x} (ns)" if self._ts_cols[x] == _ALREADY_DT else f"{x} (Local)" for x in self.timestamp_columns
        }
        if self._ts_col_names:
            (foot := self.query_one(TableFooter)).pending_action = "Computing Ts columns"
            try:
                calc_expr = [
                    (
                        pl.col(x).dt.epoch("ns")
                        if self._ts_cols[x] == _ALREADY_DT
                        else pl.from_epoch(pl.col(x), time_unit=self._ts_cols[x]).dt.convert_time_zone(_TIMEZONE)
                    ).alias(self._ts_col_names[x])
                    for x in self.timestamp_columns
                ]
                print(calc_expr)
                self._original_dt = self._original_dt.with_columns(calc_expr)
                print(self._original_dt.columns)
            except Exception as e:
                self.notify(f"Failed to compute timestamp columns: {e}", severity="warn", timeout=5)
            finally:
                foot.pending_action = None
        self.apply_filter()

    @on(SelectFromTable)
    def enable_select_from_table(self, event: SelectFromTable):
        self._select_interest = f"#{event.interested_widget.id}"
        self.query_one("#main_table", CustomTable).focus()

    @on(CustomTable.CellHighlighted)
    async def handle_cell_highlight(self, event: CustomTable.CellHighlighted):
        self.cur_row = event.coordinate.row
        self._row_detail.row_df = self._display_dt[self.cur_row]

    @on(CustomTable.CellSelected)
    def handle_cell_select(self, event: CustomTable.CellSelected):
        if self._select_interest:
            self.query_one(self._select_interest, ReceivesTableSelect).on_table_select(event.value)
            self._select_interest = None

    @on(Bookmarks.BookmarkSelected)
    def handle_bookmark_select(self, event: Bookmarks.BookmarkSelected):
        dt = self.query_one("#main_table", CustomTable)
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
        event.stop()
        self.refresh_bindings()
        self.query_one("#main_table", CustomTable).refresh(repaint=True)

    async def action_show_filter(self):
        # import gc
        # gc.set_debug(gc.DEBUG_SAVEALL)
        # def print_gc(*_):
        #     by_typ = {}
        #     for x in gc.garbage:
        #         r = by_typ.setdefault(type(x), 1)
        #         by_typ[type(x)] = r + 1
        #     for x, k in reversed(sorted(by_typ.items(), key=lambda v: v[1])):
        #         print(f"{x}={k}")
        # gc.callbacks.append(print_gc)

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
        if not (edtq := self.query_one("#main_table", CustomTable)):
            return False

        if not edtq.has_focus and action in (x.action if isinstance(x, Binding) else x[1] for x in DtBrowser.BINDINGS):
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

        if action == "timestamp_selector":
            return len(self._ts_cols) > 0

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

        self.query_one("#main_table", CustomTable).set_metadata(self._meta_dt)

    def on_mount(self):
        self.cur_total_rows = len(self._display_dt)
        self.total_rows = len(self._original_dt)
        self._row_detail.row_df = self._display_dt[0]
        gc.disable()
        # self.set_interval(3, self._maybe_gc)
        # message_hook.set(self._set_last_message)

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        self._ts_col_selector.data_bind(
            selected_columns=DtBrowser.timestamp_columns, available_columns=DtBrowser.available_timestamp_columns
        )
        self._color_selector.data_bind(selected_columns=DtBrowser.color_by, available_columns=DtBrowser.all_columns)
        self._column_selector.data_bind(
            selected_columns=DtBrowser.visible_columns, available_columns=DtBrowser.all_columns
        )

        with Horizontal(id="main_hori"):
            yield TableWithBookmarks(
                self._original_dt,
                metadata_dt=self._meta_dt,
                bookmarks=self._bookmarks,
                cursor_type=CustomTable.CursorType.CELL,
                id="main_table",
            ).data_bind(DtBrowser.active_search_queue)
        yield TableFooter().data_bind(
            DtBrowser.cur_row,
            DtBrowser.cur_total_rows,
            DtBrowser.is_filtered,
            DtBrowser.total_rows,
            DtBrowser.active_search_queue,
            DtBrowser.active_search_idx,
        )


class DtBrowserApp(App):  # pylint: disable=too-many-public-methods,too-many-instance-attributes

    def __init__(self, table_name: str, source_file_or_table: pathlib.Path | pl.DataFrame):
        super().__init__()
        self._table_name = table_name
        self._source = source_file_or_table

    def compose(self):
        yield DtBrowser(self._table_name, self._source)


@click.command()
@click.argument("source_file", nargs=1, required=True, type=pathlib.Path)
def run(source_file: pathlib.Path):
    app = DtBrowserApp(source_file, source_file)
    app.run(mouse=False)


if __name__ == "__main__":
    run()  # pylint: disable=no-value-for-parameter
