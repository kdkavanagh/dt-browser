from dataclasses import dataclass
from typing import cast

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import Reactive, reactive
from textual.suggester import Suggester
from textual.widget import Widget
from textual.widgets import Input, Label, ListItem, ListView, Rule

from dt_browser import ReceivesTableSelect


class _ExpressionList(Widget):
    DEFAULT_CSS = """
.expressionbox--list {
    padding: 0 1;
}
"""
    current_expressions: Reactive[dict[str, str]] = reactive(dict, recompose=True)

    def compose(self):
        yield ListView(
            *(
                ListItem(Label(f"{column_name} = {value}"), name=column_name, id=column_name)
                for column_name, value in reversed(self.current_expressions.items())
            ),
            id="active_exprs",
            classes="expressionbox--list",
        )


class ExpressionBox(ReceivesTableSelect):
    DEFAULT_CSS = """
ExpressionBox {
    height: 15;
    border: tall white;

}

.expressionbox--exprrow {
    height: 3;

}

.expressionbox--input {
    width: 1fr;
    margin: 0 1;

}

.expressionbox--list {
    padding: 0 1;
}
"""

    BINDINGS = [
        ("escape", "close()", "Close"),
        ("delete", "delete_expression", "Delete expression"),
        Binding("tab", "toggle_tab()", show=False),
    ]

    @dataclass
    class ExpressionSubmitted(Message):
        column_name: str
        value: str

    @dataclass
    class ExpressionDeleted(Message):
        column_name: str

    current_expressions: Reactive[dict[str, str]] = reactive(dict)
    read_only_columns: reactive[tuple[str, ...]] = reactive(tuple())

    def __init__(self, *args, suggestor: Suggester | None = None, **kwargs):
        super().__init__(*args, **kwargs)

        self._suggestor = suggestor

    async def expression_added(self, column_name: str, expr: str):
        self.current_expressions = self.current_expressions.copy() | {column_name: expr}

    def on_table_select(self, value: str):
        box = self.query_one(Input)
        box.value = f"{box.value}{value}"
        box.focus()

    @on(Input.Submitted)
    async def submit_expression(self, event: Input.Submitted):
        new_value = event.value

        the_list = self.query_one("#active_exprs", ListView)
        if new_value:
            column_name, expr = new_value.split("=", 1)
            column_name = column_name.strip()
            if column_name in self.read_only_columns:
                self.notify(f"Cannot modify read-only column '{column_name}'", severity="error", timeout=5)
                return

            expr = expr.strip()
            new_value = f"{column_name} = {expr}"

            self.query_one(Input).value = new_value
            msg = ExpressionBox.ExpressionSubmitted(column_name=column_name, value=expr)
            self.post_message(msg)
        else:
            the_list.index = None

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """Check if an action may run."""
        list_view = self.query_one("#active_exprs", ListView)
        if action == "remove_expression":
            return list_view.has_focus and len(list_view) > 0
        return True

    async def action_delete_expression(self):
        the_list = self.query_one(ListView)
        sel_item = cast(ListItem, the_list.children[the_list.index])
        column_name = sel_item.id
        self.current_expressions.pop(column_name)
        self.post_message(ExpressionBox.ExpressionDeleted(column_name=column_name))
        await the_list.remove_items([the_list.index])
        if not the_list.children:
            self.query_one(Input).focus()

    @on(ListView.Selected)
    def input_historical(self, event: ListView.Selected):
        box = self.query_one(Input)
        box.value = f"{event.item.name} = {self.current_expressions[event.item.name]}"
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
        self.border_title = "Compute Expressions"

        with Vertical():
            with Horizontal(classes="expressionbox--exprrow"):
                yield Input(
                    placeholder="Enter SQL query to compute column (format: <column name> = <expression>)",
                    classes="expressionbox--input",
                    suggester=self._suggestor,
                )
            yield Rule()
            yield _ExpressionList().data_bind(current_expressions=ExpressionBox.current_expressions)

    def on_mount(self):
        self.query_one(Input).focus()
