from dataclasses import dataclass

from textual import events, on
from textual.binding import Binding
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Input, SelectionList
from textual.widgets.selection_list import Selection

_INPUT_HINT = "Type to filter columns list..."


class ColumnSelector(Widget):

    DEFAULT_CSS = """
    ColumnSelector {
        dock: right;
        border: tall $accent;
        height: auto;
    }

    .columnselector--inputbox {
        height: auto;
        margin: 0 0 0 0;
    }
    """

    BINDINGS = [
        ("escape", "close()", "Close and Apply"),
        Binding("ctrl+a", "apply()", "Apply", key_display="ctrl+a"),
        ("shift+up", "move(True)", "Move up"),
        ("shift+down", "move(False)", "Move Down"),
        Binding("ctrl+s", "select_all()", "Select All", key_display="ctrl+s"),
        Binding("ctrl+x", "deselect_all()", "Deselect All", key_display="ctrl+x"),
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
    ordered_columns: reactive[tuple[str, ...]] = reactive(tuple(), init=False, bindings=True)
    filtered_ordered_columns: reactive[tuple[str, ...]] = reactive(tuple(), init=False, bindings=True)

    def __init__(self, *args, title: str | None = None, allow_reorder: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self._allow_reorder = allow_reorder
        self._title = title
        self.filter_value: str | None = None
        self._message: ColumnSelector.ColumnSelectionChanged | None = None

    def action_close(self):
        self.action_apply()
        self.remove()

    def action_apply(self):
        if self._message is not None:
            self.post_message(self._message)
            self._message = None

    def action_deselect_all(self):
        self.query_one(SelectionList).deselect_all()

    def action_select_all(self):
        self.query_one(SelectionList).select_all()

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        if action == "move":
            filter_value = self.query_one(Input).value
            if filter_value:
                return False
            if not self.ordered_columns or not self._allow_reorder:
                return False
            cur_idx = self.query_one(SelectionList).highlighted
            if cur_idx is None:
                return False
            if parameters[0]:
                return cur_idx > 0
            return cur_idx < len(self.ordered_columns) - 1
        return True

    async def on_key(self, event: events.Key) -> None:
        box = self.query_one(Input)
        if box.check_consume_key(event.key, event.character):
            await box.on_event(event)
            return
        if event.key == "backspace":
            box.action_delete_left()
            return

    @on(SelectionList.SelectionHighlighted)
    def _refresh_actions(self):
        self.refresh_bindings()

    @on(Input.Changed)
    def _update_filter(self, event: Input.Changed):
        self.filter_value = event.value
        event.prevent_default()
        self._refresh_options()

    def compute_filtered_ordered_columns(self):
        if not self.filter_value:
            return self.ordered_columns
        return tuple(x for x in self.ordered_columns if self.filter_value.lower() in x.lower())

    def _refresh_options(self):
        sel_list = self.query_one(SelectionList)
        sel_idx = sel_list.highlighted
        if sel_idx is not None:
            sel_val: str | None = sel_list.get_option_at_index(sel_idx).value
        else:
            sel_val = None
        sel_list.clear_options()
        for i, x in enumerate(self.filtered_ordered_columns):
            sel_list.add_option(Selection(x, x, x in self.selected_columns))
            if x == sel_val:
                sel_list.highlighted = i

    def action_move(self, is_up: bool):
        sel_list = self.query_one(SelectionList)
        if (idx := sel_list.highlighted) is None:
            return
        if is_up:
            self.ordered_columns = (
                self.ordered_columns[0 : idx - 1]
                + (self.ordered_columns[idx], self.ordered_columns[idx - 1])
                + self.ordered_columns[idx + 1 :]
            )
        else:
            self.ordered_columns = (
                self.ordered_columns[0:idx]
                + (self.ordered_columns[idx + 1], self.ordered_columns[idx])
                + self.ordered_columns[idx + 2 :]
            )
        self._refresh_options()

    def watch_available_columns(self):
        new_disp = []
        for x in self.available_columns:
            if x not in self.ordered_columns:
                new_disp.append(x)
        if new_disp:
            self.ordered_columns = self.ordered_columns + tuple(new_disp)
        max_item = max([len(x) for x in self.available_columns] + [0])
        self.styles.width = max([max_item + 10, 35, len(_INPUT_HINT)])

    def watch_ordered_columns(self):
        self.selected_columns = [x for x in self.ordered_columns if x in self.selected_columns]
        self._refresh_options()

    def watch_selected_columns(self):
        self._message = ColumnSelector.ColumnSelectionChanged(selected_columns=self.selected_columns, selector=self)

    def on_mount(self):
        self.query_one(SelectionList).focus()
        self.query_one(SelectionList).highlighted = 0

    def compose(self):
        inp = Input(classes="columnselector--inputbox", type="text", placeholder=_INPUT_HINT)
        inp.can_focus = False
        yield inp
        yield SelectionList[int](
            *(Selection(x, x, x in self.selected_columns) for x in self.ordered_columns),
        )

        self.border_title = self._title

    @on(SelectionList.SelectedChanged)
    def on_column_selection(self, event: SelectionList.SelectedChanged):
        event.stop()
        sels = event.selection_list.selected
        new_sels: list[str] = []

        for x in self.ordered_columns:
            # Retain existing selections not in filtered set
            if x not in self.filtered_ordered_columns:
                if x in self.selected_columns:
                    new_sels.append(x)
            elif x in sels:
                new_sels.append(x)

        self.selected_columns = tuple(new_sels)
