from dataclasses import dataclass

from textual import on
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.reactive import reactive
from textual.suggester import Suggester
from textual.widgets import Input, Label, ListItem, ListView, Rule

from dt_browser import HasState, ReceivesTableSelect


class FilterBox(ReceivesTableSelect, HasState):
    DEFAULT_CSS = """
FilterBox {
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

    def get_value(self):
        return self._active_filter[self.is_goto]

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
