from dataclasses import dataclass

import polars as pl
from textual import on
from textual.message import Message
from textual.widget import Widget
from textual_fastdatatable import DataTable

from dt_browser import INDEX_COL
from dt_browser.polars_backend import PolarsBackend


class Bookmarks(Widget):

    @dataclass
    class BookmarkSelected(Message):
        selected_index: int

    @dataclass
    class BookmarkRemoved(Message):
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
        dt = DataTable(backend=self._bookmark_df, cursor_type="row", max_rows=5)
        dt.styles.height = "auto"
        yield dt

    def add_bookmark(self, df: pl.DataFrame, meta_df: pl.DataFrame):
        if not self._meta_dt.is_empty() and meta_df[INDEX_COL][0] in self._meta_dt[INDEX_COL]:
            return False
        self._bookmark_df.append_rows(df)
        self._meta_dt = pl.concat([self._meta_dt, meta_df])
        return True

    @property
    def has_bookmarks(self):
        return not self._bookmark_df.data.is_empty()

    def action_close(self):
        self.remove()

    def on_mount(self):
        self.query_one(DataTable).focus()

    def action_remove_bookmark(self):
        dt = self.query_one(DataTable)
        idx = self._meta_dt[dt.cursor_row][INDEX_COL][0]
        if len(self._bookmark_df.data) == 1:
            self._bookmark_df.drop_row(0)
            self._meta_dt = self._meta_dt.clear()
            self.remove()
        else:
            dt.remove_row(dt.cursor_row)
            above = self._meta_dt.slice(0, dt.cursor_row)
            below = self._meta_dt.slice(dt.cursor_row + 1)
            self._meta_dt = pl.concat([above, below])
        self.post_message(Bookmarks.BookmarkRemoved(selected_index=idx))

    @on(DataTable.RowSelected)
    def handle_select(self, event: DataTable.RowSelected):
        sel_row = event.cursor_row
        index = int(self._meta_dt[sel_row][INDEX_COL][0])
        self.post_message(Bookmarks.BookmarkSelected(selected_index=index))
