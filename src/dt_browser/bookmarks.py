from dataclasses import dataclass

import polars as pl
from textual import on
from textual.message import Message
from textual.widget import Widget

from dt_browser import INDEX_COL
from dt_browser.custom_table import CustomTable


def remove_row(dt: pl.DataFrame, rem_row: int):
    above = dt.slice(0, rem_row)
    below = dt.slice(rem_row + 1)
    return pl.concat([above, below])


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
        self._bookmark_df = pl.DataFrame()
        self._history: dict[str, list[int]] = {}
        self._meta_dt = pl.DataFrame()

    @property
    def meta_dt(self):
        return self._meta_dt

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
        dt = CustomTable(self._bookmark_df, metadata_dt=self._meta_dt, cursor_type=CustomTable.CursorType.ROW)
        dt.styles.height = "auto"
        yield dt

    def toggle_bookmark(self, df: pl.DataFrame, meta_df: pl.DataFrame):
        if not self._meta_dt.is_empty() and meta_df[INDEX_COL][0] in self._meta_dt[INDEX_COL]:
            self.remove_bookmark(meta_df[INDEX_COL][0])
            return False
        self._bookmark_df = pl.concat([self._bookmark_df, df])
        self._meta_dt = pl.concat([self._meta_dt, meta_df])
        dt = self.query_children(CustomTable)
        if dt:
            dt.first().set_dt(self._bookmark_df, self._meta_dt)
        return True

    @property
    def has_bookmarks(self):
        return not self._bookmark_df.is_empty()

    def action_close(self):
        self.remove()

    async def on_mount(self):
        self.query_one(CustomTable).focus()

    def action_remove_bookmark(self):
        dt = self.query_one(CustomTable)
        idx = self._meta_dt[dt.cursor_coordinate.row][INDEX_COL][0]
        self.remove_bookmark(idx)

    def remove_bookmark(self, idx: int):
        dt = self.query_children(CustomTable)
        if len(self._bookmark_df) == 1:
            self._bookmark_df = self._bookmark_df.clear()
            self._meta_dt = self._meta_dt.clear()
            if dt:
                self.remove()
        else:
            rem_row = self._meta_dt.with_row_index(name="__tmp").filter(pl.col(INDEX_COL) == idx)["__tmp"][0]
            self._meta_dt = remove_row(self._meta_dt, rem_row)
            self._bookmark_df = remove_row(self._bookmark_df, rem_row)
            if dt:
                dt.first().set_dt(self._bookmark_df, self._meta_dt)

        self.post_message(Bookmarks.BookmarkRemoved(selected_index=idx))

    @on(CustomTable.CellSelected)
    def handle_select(self, event: CustomTable.CellSelected):
        event.stop()
        sel_row = event.coordinate.row
        index = int(self._meta_dt[sel_row][INDEX_COL][0])
        self.post_message(Bookmarks.BookmarkSelected(selected_index=index))
