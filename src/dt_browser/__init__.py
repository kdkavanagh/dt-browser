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
    pass


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
