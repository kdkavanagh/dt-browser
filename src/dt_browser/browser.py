import pathlib
from abc import abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import click
import polars as pl
from rich.style import Style
from rich.text import Text
from textual import on
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.cache import LRUCache
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.suggester import Suggester
from textual.widget import Widget
from textual.widgets import (
    Footer,
    Input,
    Label,
    ListItem,
    ListView,
    Rule,
    SelectionList,
)
from textual.widgets.selection_list import Selection
from textual_fastdatatable import DataTable

from dt_browser.polars_backend import PolarsBackend

_COLORS = pl.Enum(
    (
        "#576176",
        "#FAA5AB",
        "#A5CD84",
        "#EFBD58",
        "#8DC3F1",
        "#DEAEED",
        "#27FFDF",
        "#CACCD3",
    )
)

_COLORS_STYLES = "\n".join(
    f"""
ExtendedDataTable > .datatable--row{i} {{
    color: {x};
}}                        
"""
    for i, x in enumerate(_COLORS.categories)
)

_INDEX_COL = "__index"
_COLOR_COL = "__color"

_SHOW_COLUMNS_ID = "showColumns"
_COLOR_COLUMNS_ID = "colorColumns"


class ExtendedDataTable(DataTable):

    DEFAULT_CSS = DataTable.DEFAULT_CSS + _COLORS_STYLES
    COMPONENT_CLASSES: ClassVar[set[str]] = DataTable.COMPONENT_CLASSES.union(
        [f"datatable--row{i}" for i in range(len(_COLORS.categories))]
    )

    def __init__(self, *args, metadata_dt: pl.DataFrame, **kwargs):
        super().__init__(*args, **kwargs)
        self.meta_dt = metadata_dt

    def _get_row_style(self, row_index: int, base_style: Style) -> Style:
        style = super()._get_row_style(row_index, base_style)
        if row_index < 0 or _COLOR_COL not in self.meta_dt.columns:
            return style
        color_idx = self.meta_dt[_COLOR_COL][row_index]
        row_color = self.get_component_styles(f"datatable--row{color_idx}").rich_style
        return Style.combine([style, row_color])


class ColumnNameSuggestor(Suggester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, case_sensitive=True, **kwargs)
        self.columns = tuple[str, ...]()

    async def get_suggestion(self, value: str) -> str | None:
        if not value:
            return None
        tokens = value.rsplit(maxsplit=3)
        last_token = tokens[-1]
        if len(tokens) > 1:
            combos = ("and", "or", "AND", "OR")
            for misc in combos:
                if misc.startswith(last_token) and tokens[-2] not in combos:
                    return f"{value}{misc.removeprefix(last_token)}"

        for col in self.columns:
            if col.startswith(last_token):
                return f"{value}{col.removeprefix(last_token)}"

        return None


class ReceivesTableSelect(Widget):

    BINDINGS = [
        ("ctrl+t", "select_from_table()", "Select/copy value from table"),
    ]

    def action_select_from_table(self):
        self.post_message(SelectFromTable(interested_widget=self))

    @abstractmethod
    def on_table_select(self, value: str):
        pass


@dataclass
class SelectFromTable(Message):
    interested_widget: ReceivesTableSelect


class HasState:

    @abstractmethod
    def save_state(self, existing: dict) -> dict:
        """
        Generate any persistent data from this object

        Args:
            existing: Any existing state for this object which should be merged with the current state.
            (e.g if there are multiple instances of the browser which should be merged into a single state object)
        """

    @abstractmethod
    def load_state(self, state: dict, table_name: str, df: pl.DataFrame):
        """
        Apply the provided state to the current object

        Args:
            state: the state
            table_name: the current table name being displayed
            df: The full dataframe being displayed
        """


class FilterBox(ReceivesTableSelect, HasState):
    DEFAULT_CSS = """
FilterBox {
    dock: bottom;
    height: 15;
    border: tall white;

}

.filterbox--filterrow {
    height: 3;

}

.filterbox--input {
    width: 1fr;
    margin: 0 1;

}

.filterbox--history {
    padding: 0 1;
}
"""

    BINDINGS = [
        ("escape", "close()", "Close"),
        Binding("tab", "toggle_tab()", show=False),
    ]

    @dataclass
    class FilterSubmitted(Message):
        value: str | None

    @dataclass
    class GoToSubmitted(Message):
        value: str | None

    is_goto = reactive(False)

    def __init__(self, *args, suggestor: Suggester | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._history: list[str] = []
        self._active_filter: dict[bool, str | None] = {True: None, False: None}
        self._suggestor = suggestor

    def save_state(self, existing: dict) -> dict:
        history = self._history.copy()
        for x in existing["history"]:
            if x not in history:
                history.append(x)
        return {"history": history}

    def load_state(self, state: dict, *_):
        self._history = state["history"]

    def query_failed(self, query: str):
        self._history.remove(query)
        for child in self.walk_children(ListItem):
            if child.name == query:
                child.remove()

    def on_table_select(self, value: str):
        box = self.query_one(Input)
        box.value = f"{box.value}{value}"
        box.focus()

    @on(Input.Submitted)
    async def apply_filter(self, event: Input.Submitted):
        new_value = event.value
        if new_value == self._active_filter[self.is_goto]:
            return

        the_list = self.query_one(ListView)
        if new_value:
            self.query_one(Input).value = new_value
            if new_value not in self._history:
                self._history.append(new_value)
                the_list.index = None
                await the_list.insert(0, [ListItem(Label(new_value), name=new_value)])
                the_list.index = 0
        else:
            the_list.index = None
        self._active_filter[self.is_goto] = new_value
        if self.is_goto:
            msg = FilterBox.GoToSubmitted(value=new_value)
        else:
            msg = FilterBox.FilterSubmitted(value=new_value)
        self.post_message(msg)

    @on(ListView.Selected)
    def input_historical(self, event: ListView.Selected):
        box = self.query_one(Input)
        box.value = event.item.name
        box.focus()

    def key_down(self):
        if not (box := self.query_one(ListView)).has_focus and box.children:
            box.focus()

    def action_toggle_tab(self):
        if not (box := self.query_one(Input)).has_focus:
            box.focus()
        if box._suggestion:  # pylint: disable=protected-access
            box.action_cursor_right()

    def action_close(self):
        self.remove()

    def compose(self):
        if self.is_goto:
            self.border_title = "Search dataframe"
        else:
            self.border_title = "Filter dataframe"

        with Vertical():
            with Horizontal(classes="filterbox--filterrow"):
                yield Input(
                    value=self._active_filter[self.is_goto],
                    classes="filterbox--input",
                    suggester=self._suggestor,
                )
            yield Rule()
            yield ListView(
                *(ListItem(Label(x), name=x) for x in reversed(self._history)),
                classes="filterbox--history",
            )

    def on_mount(self):
        self.query_one(Input).focus()
        idx = None
        if self._active_filter[self.is_goto] and self._active_filter[self.is_goto] in self._history:
            idx = len(self._history) - (self._history.index(self._active_filter[self.is_goto]) + 1)
        self.query_one(ListView).index = idx


class Bookmarks(Widget):

    @dataclass
    class BookmarkSelected(Message):
        selected_index: int

    DEFAULT_CSS = """
    Bookmarks {
        dock: bottom;
        height: 15;
        border: tall white;

    }

    .bookmarks--history {
        padding: 0 1;
    }
    """

    BINDINGS = [
        ("escape", "close()", "Close"),
        ("delete", "remove_bookmark()", "Remove bookmark"),
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bookmark_df = PolarsBackend(pl.DataFrame())
        self._history: dict[str, list[int]] = {}
        self._meta_dt = pl.DataFrame()

    # def save_state(self, existing: dict):
    #     max_history = 10
    #     history = self._history.copy()
    #     for k, v in existing[history]:
    #         if k not in history:
    #             history[k] = v

    #     if len(history) > max_history:
    #         history = {k: v for k, v in list(history.items())[-max_history:]}
    #     return {"history": {**history, self._table_name: self._bookmark_df.data[_INDEX_COL].to_dict()}}

    # def load_state(self, state: dict, table_name: str, df: pl.DataFrame):
    #     self._history = state["history"]
    #     if self._table_name in self._history:
    #         self.add_bookmark(df[*state["history"][table_name]])

    def compose(self):
        yield DataTable(backend=self._bookmark_df, cursor_type="row", max_rows=5)

    def add_bookmark(self, df: pl.DataFrame, meta_df: pl.DataFrame):
        self._bookmark_df.append_rows(df)
        self._meta_dt = pl.concat([self._meta_dt, meta_df])

    @property
    def has_bookmarks(self):
        print(self._bookmark_df.data)
        return not self._bookmark_df.data.is_empty()

    def action_close(self):
        self.remove()

    def on_mount(self):
        self.query_one(DataTable).focus()

    def action_remove_bookmark(self):
        dt = self.query_one(DataTable)
        if len(self._bookmark_df.data) == 1:
            self._bookmark_df.drop_row(0)
            self.remove()
        else:
            dt.remove_row(dt.cursor_row)

    @on(DataTable.RowSelected)
    def handle_select(self, event: DataTable.RowSelected):
        sel_row = event.cursor_row
        index = int(self._meta_dt[sel_row][_INDEX_COL][0])
        self.post_message(Bookmarks.BookmarkSelected(selected_index=index))


class ColumnSelector(Widget, HasState):

    DEFAULT_CSS = """
    ColumnSelector {
        dock: right;

    }
    """

    BINDINGS = [
        ("escape", "close()", "Close"),
        ("shift+up", "move(True)", "Move up"),
        ("shift+down", "move(False)", "Move Down"),
    ]

    @dataclass
    class ColumnSelectionChanged(Message):
        selected_columns: tuple[str, ...]
        selector: "ColumnSelector"

        @property
        def control(self):
            return self.selector

    available_columns: reactive[tuple[str, ...]] = reactive(tuple())
    selected_columns: reactive[tuple[str, ...]] = reactive(tuple(), init=False)
    display_columns: reactive[tuple[str, ...]] = reactive(tuple(), init=False, bindings=True)

    def __init__(self, *args, title: str | None = None, allow_reorder: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_reorder = allow_reorder
        self._title = title

    def action_close(self):
        self.remove()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "move":
            if not self.display_columns or not self._allow_reorder:
                return False
            cur_idx = self.query_one(SelectionList).highlighted
            if cur_idx is None:
                return False
            if parameters[0]:
                return cur_idx > 0
            return cur_idx < len(self.display_columns) - 1
        return True

    @on(SelectionList.SelectionHighlighted)
    def _refresh_actions(self):
        self.refresh_bindings()

    def _refresh_options(self):
        sel_list = self.query_one(SelectionList)
        sel_idx = sel_list.highlighted
        if sel_idx is not None:
            sel_val: str | None = sel_list.get_option_at_index(sel_idx).value
        else:
            sel_val = None
        sel_list.clear_options()
        for i, x in enumerate(self.display_columns):
            sel_list.add_option(Selection(x, x, x in self.selected_columns))
            if x == sel_val:
                sel_list.highlighted = i

    def action_move(self, is_up: bool):
        sel_list = self.query_one(SelectionList)
        if (idx := sel_list.highlighted) is None:
            return
        if is_up:
            self.display_columns = (
                self.display_columns[0 : idx - 1]
                + (self.display_columns[idx], self.display_columns[idx - 1])
                + self.display_columns[idx + 1 :]
            )
        else:
            self.display_columns = (
                self.display_columns[0:idx]
                + (self.display_columns[idx + 1], self.display_columns[idx])
                + self.display_columns[idx + 2 :]
            )
        self._refresh_options()

    def watch_available_columns(self):
        new_disp = []
        for x in self.available_columns:
            if x not in self.display_columns:
                new_disp.append(x)
        if new_disp:
            self.display_columns = self.display_columns + tuple(new_disp)
        self.styles.width = max(max([len(x) for x in self.available_columns] + [0]) + 10, 35)

    def watch_display_columns(self):
        self.selected_columns = [x for x in self.display_columns if x in self.selected_columns]
        self._refresh_options()

    def watch_selected_columns(self):
        self.post_message(ColumnSelector.ColumnSelectionChanged(selected_columns=self.selected_columns, selector=self))

    def save_state(self, *_) -> dict:
        return {"selected": self.selected_columns}

    def load_state(self, state: dict):
        self.selected_columns = tuple(x for x in state["selected"] if x in self.available_columns)

    def on_mount(self):
        (sel := self.query_one(SelectionList)).focus()
        sel.highlighted = 0

    def compose(self):
        sel = SelectionList[int](
            *(Selection(x, x, x in self.selected_columns) for x in self.display_columns),
        )

        sel.border_title = self._title
        yield sel

    @on(SelectionList.SelectedChanged)
    def on_column_selection(self, event: SelectionList.SelectedChanged):
        event.stop()
        sels = event.selection_list.selected
        self.selected_columns = [x for x in self.display_columns if x in sels]


class FooterRowCount(Widget):

    DEFAULT_CSS = """
    FooterRowCount {
        background: $accent-darken-1;
        width: auto;
        height: 1;

        padding: 0 2;
    }

    FooterRowCount > .tablefooter--label {
        background: $secondary;
        text-style: bold;
    }

    """

    is_filtered = reactive(False)
    cur_row = reactive(1)
    cur_total_rows = reactive(0)
    total_rows = reactive(0)

    def render(self):
        text = Text(
            no_wrap=True,
            overflow="ellipsis",
            justify="right",
            end="",
        )
        # key_style = self.get_component_rich_style("footerrowcount--rowcount")
        key_text = Text.assemble(str(self.cur_row), "/", str(self.cur_total_rows))
        if self.cur_total_rows != self.total_rows:
            key_text = Text.assemble(key_text, " (", str(self.total_rows), ")")
        text.append(key_text)
        return text


class ReactiveLabel(Label):

    value: reactive[str] = reactive("", layout=True)

    def render(self):
        if isinstance(self.value, (int, float)):
            return f"{self.value:,}"
        return str(self.value)


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

    FooterRowCount > .tablefooter--label {
        background: $secondary;
        text-style: bold;
    }

    """

    is_filtered = reactive(False)
    cur_row = reactive(1)
    cur_total_rows = reactive(0)
    total_rows = reactive(0)
    total_rows_display = reactive("", layout=True)

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
        if self.active_search_len is not None:
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
        widths.append("auto")
        self.styles.grid_columns = " ".join(widths)
        self.styles.grid_size_columns = len(widths)

    def compute_total_rows_display(self):
        return f" (Filtered from {self.total_rows:,})" if self.cur_total_rows != self.total_rows else ""

    def compute_search_idx_display(self):
        return self.active_search_idx + 1


class DtBrowser(App):  # pylint: disable=too-many-public-methods,too-many-instance-attributes
    """A Textual app to manage stopwatches."""

    BINDINGS = [
        ("f", "show_filter", "Filter rows"),
        ("/", "show_search", "Search"),
        ("n", "iter_search(True)", "Next"),
        Binding("N", "iter_search(False)", "Prev", key_display="shift+N"),
        ("b", "add_bookmark", "Add Bookmark"),
        Binding("B", "show_bookmarks", "Bookmarks", key_display="shift+B"),
        ("c", "column_selector", "Columns..."),
        Binding("C", "show_colors", "Colors...", key_display="shift+C"),
    ]

    DEFAULT_CSS = """
DtBrowser {
layers: regular above;
}
.dock-bottom {
    dock: bottom;
}

.dtbrowser--toolbox {
    height: 15;
}

"""

    color_by: reactive[tuple[str, ...]] = reactive(tuple(), init=False)
    visible_columns: reactive[tuple[str, ...]] = reactive(tuple())
    all_columns: reactive[tuple[str, ...]] = reactive(tuple())
    is_filtered = reactive(False)
    cur_row = reactive(1)
    cur_total_rows = reactive(0)
    total_rows = reactive(0)

    active_search_queue: reactive[list[int] | None] = reactive(None)
    active_search_idx: reactive[int | None] = reactive(None)
    active_search: reactive[str | None] = reactive(None)
    # active_dt: reactive[pl.DataFrame] = reactive(pl.DataFrame(), init=False, always_update=True)

    def __init__(self, table_name: str, source_file: pathlib.Path):
        super().__init__()
        self._source = source_file

        self._display_dt = self._filtered_dt = self._original_dt = PolarsBackend.from_file_path(self._source).data
        self._meta_dt = self._original_meta = self._original_dt.with_row_index(name=_INDEX_COL).select([_INDEX_COL])
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

        # self.set_reactive(DtBrowser.color_by, self._backend.columns[0:1])
        self.color_by = tuple(self._backend.columns[0:1])

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
    async def apply_filter(self, event: FilterBox.FilterSubmitted):
        if not event.value:
            self.is_filtered = False
            idx = self.query_one(ExtendedDataTable).cursor_coordinate.row
            await self._set_filtered_dt(
                self._original_dt,
                self._original_meta,
                new_row=self._meta_dt[_INDEX_COL][idx],
                focus=False,
            )
        else:
            ctx = pl.SQLContext(frames={"dt": pl.concat([self._original_dt, self._original_meta], how="horizontal")})
            try:
                dt = ctx.execute(f"select * from dt where {event.value}").collect()
                meta = dt.select([x for x in dt.columns if x.startswith("__")])
                dt = dt.select([x for x in dt.columns if not x.startswith("__")])
                self.is_filtered = True
                await self._set_filtered_dt(dt, meta, new_row=0, focus=False)
            except Exception as e:
                self.query_one(FilterBox).query_failed(event.value)
                self.notify(f"Failed to apply filter due to: {e}", severity="error", timeout=10)

    @on(FilterBox.GoToSubmitted)
    async def apply_search(self, event: FilterBox.GoToSubmitted):
        self.active_search = event.value

    def watch_active_search(self):
        if not self.active_search:
            self.active_search_queue = None
            self.active_search_idx = 0
            return

        try:
            ctx = pl.SQLContext(frames={"dt": (pl.concat([self._display_dt, self._meta_dt], how="horizontal"))})
            search_queue = list(
                ctx.execute(f"select {_INDEX_COL} from dt where {self.active_search}").collect()[_INDEX_COL]
            )
            if not search_queue:
                self.notify("No results found for search", severity="warn", timeout=5)
            else:
                self.active_search_queue = search_queue
                self.active_search_idx = -1
                self.action_iter_search(True)
        except Exception as e:
            self.query_one(FilterBox).query_failed(self.active_search)
            self.notify(f"Failed to apply filter due to: {e}", severity="error", timeout=10)

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

    def action_add_bookmark(self):
        row_idx = self.query_one(ExtendedDataTable).cursor_coordinate.row
        self._bookmarks.add_bookmark(self._display_dt[row_idx], self._meta_dt[row_idx])
        self.refresh_bindings()
        self.notify("Bookmark added!", severity="information", timeout=3)

    async def action_show_bookmarks(self):
        await self.mount(self._bookmarks, before=self.query_one(TableFooter))

    async def action_column_selector(self):
        self._column_selector.data_bind(
            selected_columns=DtBrowser.visible_columns, available_columns=DtBrowser.all_columns
        )
        await self.mount(self._column_selector)

    async def action_show_colors(self):
        self._color_selector.data_bind(selected_columns=DtBrowser.color_by, available_columns=DtBrowser.all_columns)
        await self.mount(self._color_selector)

    async def _set_filtered_dt(self, filtered_dt: pl.DataFrame, filtered_meta: pl.DataFrame, **kwargs):
        self._filtered_dt = filtered_dt
        self._meta_dt = filtered_meta
        await self._set_active_dt(self._filtered_dt, **kwargs)

    async def _set_active_dt(self, active_dt: pl.DataFrame, **kwargs):
        self._display_dt = active_dt.select(self.visible_columns)
        self.cur_total_rows = len(self._display_dt)
        self.watch_active_search()
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
    def handle_cell_highlight(self, event: ExtendedDataTable.CellHighlighted):
        self.cur_row = event.coordinate.row

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
            filt = self._meta_dt.with_row_index("__displayIndex").filter(pl.col(_INDEX_COL) == sel_idx)
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

    async def action_show_color_select(self):
        if existing := self.query("#color_select"):
            self.color_by = set(next(iter(existing.results(SelectionList))).selected)
            existing.remove()
            return

        sel = SelectionList[int](
            *(Selection(x, x, x in self.color_by) for i, x in enumerate(self._backend.data.columns)),
            id="color_select",
            classes="dock-bottom toolbox",
        )
        sel.border_title = "Color rows by column(s)"
        await self.mount(sel, before=self.query_one(TableFooter))
        sel.focus(True)

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

    async def watch_color_by(self):
        if not self.color_by:
            self._meta_dt = self._meta_dt.drop(_COLOR_COL, strict=False)
            self._original_meta = self._original_meta.drop(_COLOR_COL, strict=False)
        else:
            cols = tuple(self.color_by)
            if cols not in self._color_by_cache:
                self._color_by_cache.set(
                    cols,
                    self._original_dt.with_columns(
                        __color=(
                            (pl.any_horizontal(*(pl.col(x) != pl.col(x).shift(1) for x in cols))).cum_sum().fill_null(0)
                            % len(_COLORS.categories)
                        )
                    )[_COLOR_COL],
                )
            self._original_meta = self._original_meta.with_columns(__color=self._color_by_cache.get(cols))
            self._meta_dt = self._meta_dt.drop(_COLOR_COL, strict=False).join(
                self._original_meta.select([_INDEX_COL, _COLOR_COL]), how="left", on=_INDEX_COL
            )

        await self._redraw(focus=False)

    async def _redraw(self, new_row: int | None = None, focus: bool = True):
        self._backend = PolarsBackend.from_dataframe(self._display_dt)
        existing_q = self.query(ExtendedDataTable)
        if not existing_q:
            return
        existing = existing_q.only_one()
        coord = existing.cursor_coordinate
        ys = new_row if new_row is not None else existing.scroll_y
        xs = existing.scroll_x
        await existing.remove()
        dt = ExtendedDataTable(backend=self._backend, id="table", metadata_dt=self._meta_dt)
        await self.mount(dt, before=0)

        if focus:
            dt.focus()
        with self.app.batch_update():
            dt.scroll_to(xs, ys, animate=False, force=True)
            dt.move_cursor(column=coord.column, row=new_row if new_row is not None else coord.row)

        if new_row is not None:
            self.cur_row = new_row

    def on_mount(self):
        self.cur_total_rows = len(self._display_dt)
        self.total_rows = len(self._original_dt)
        self.cur_row = 1

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield ExtendedDataTable(backend=self._backend, id="table", metadata_dt=self._meta_dt)
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
    run()
