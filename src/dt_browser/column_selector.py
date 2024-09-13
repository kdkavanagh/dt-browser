from dataclasses import dataclass

from textual import on
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import SelectionList
from textual.widgets.selection_list import Selection


class ColumnSelector(Widget):

    DEFAULT_CSS = """
    ColumnSelector {
        dock: right;
    }
    """

    BINDINGS = [
        ("escape", "close()", "Close and Apply"),
        ("a", "apply()", "Apply"),
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
        self._message: ColumnSelector.ColumnSelectionChanged | None = None

    def action_close(self):
        self.action_apply()
        self.remove()

    def action_apply(self):
        if self._message is not None:
            self.post_message(self._message)
            self._message = None

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
        self._message = ColumnSelector.ColumnSelectionChanged(selected_columns=self.selected_columns, selector=self)

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
