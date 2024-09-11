from abc import abstractmethod
from dataclasses import dataclass

import polars as pl
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Label

COLORS = pl.Enum(
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

COLORS_STYLES = "\n".join(
    f"""
ExtendedDataTable > .datatable--row{i} {{
    color: {x};
}}
"""
    for i, x in enumerate(COLORS.categories)
)

INDEX_COL = "__index"
COLOR_COL = "__color"
DISPLAY_IDX_COL = "_display_index"


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


class ReactiveLabel(Label):

    value: reactive[str] = reactive("", layout=True)

    def render(self):
        if isinstance(self.value, (int, float)):
            return f"{self.value:,}"
        return str(self.value)


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
