import itertools
import pathlib

from textual import on, work
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Grid
from textual.message import Message
from textual.screen import ModalScreen
from textual.suggester import Suggester
from textual.widgets import Button, DirectoryTree, Footer, Input, Label


class CurDirSuggestor(Suggester):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, case_sensitive=True, **kwargs)

    async def get_suggestion(self, value: str) -> str | None:
        as_path = pathlib.Path(value)
        if not as_path.is_dir():
            as_path = as_path.parent
        if as_path.exists():
            for x in sorted(itertools.islice(as_path.iterdir(), 100)):
                if str(x).startswith(value):
                    return str(x)

        return None


class SaveDataframe(Message):
    def __init__(self, path: pathlib.Path) -> None:
        super().__init__()
        self.path = path


class OverwriteWarning(ModalScreen):
    DEFAULT_CSS = """
OverwriteWarning {
 align: center middle;

}
    
OverwriteWarning > #dialog {
    grid-size: 2;
    grid-gutter: 1 2;
    grid-rows: 1fr 3;
    padding: 0 1;
    width: 60;
    height: 11;
    border: thick $background 80%;
    background: $surface;
}

OverwriteWarning > #dialog > #question {
    column-span: 2;
    height: 1fr;
    width: 1fr;
    content-align: center middle;
}

OverwriteWarning > #dialog > Button {
    width: 100%;
}
"""

    def __init__(self, target: pathlib.Path):
        super().__init__()
        self._target = target

    def compose(self) -> ComposeResult:
        yield Grid(
            Label(f"Overwrite existing file {self._target}?", id="question"),
            Button("Overwrite", variant="error", id="overwrite"),
            Button("Cancel", variant="primary", id="cancel"),
            id="dialog",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "overwrite":
            self.dismiss(True)
        else:
            self.dismiss(False)


class SaveModal(ModalScreen):
    SUPPORTED_EXTS = ("arrows", "arrow", "feather", "json", "parquet", "pqt", "csv")

    DEFAULT_CSS = """
SaveModal {
 align: center middle;

}
    
SaveModal > #dialog {
    grid-size: 1 5;
    grid-gutter: 1;
    grid-rows: 1 1fr 3 3 1;
    padding: 0 1;
    width: 60;
    height: 30;
    border: $primary 80%;
    background: $surface;
}

"""

    BINDINGS = [
        ("escape", "close()", "Close"),
        Binding("tab", "toggle_tab()", show=False),
    ]

    def compose(self) -> ComposeResult:
        with Grid(id="dialog"):
            yield Label("Save dataframe to disk")
            yield DirectoryTree("./", name="Current Directory")
            yield Input(placeholder="Enter filename...", select_on_focus=False, suggester=CurDirSuggestor())
            yield Button("Save", variant="primary")
            foot = Footer()
            foot.styles.padding = 0
            yield foot

    def action_close(self):
        self.dismiss(None)

    def action_toggle_tab(self):
        box = self.query_one(Input)
        save = self.query_one(Button)
        if self.query_one(DirectoryTree).has_focus:
            box.focus()
        if box.has_focus:
            if box._suggestion:  # pylint: disable=protected-access
                box.action_cursor_right()
            else:
                save.focus()

    def on_mount(self):
        self.query_one(DirectoryTree).focus()

    @on(Input.Submitted)
    def on_enter(self):
        self.query_one(Button).focus()

    @on(DirectoryTree.FileSelected)
    def on_file_select(self):
        (inp := self.query_one(Input)).focus()
        inp.cursor_position = len(inp.value)

    @on(DirectoryTree.NodeHighlighted)
    def on_highlight(self, msg: DirectoryTree.NodeHighlighted):
        self.query_one(Input).value = str(pathlib.Path(msg.node.data.path))  # type: ignore

    @on(Button.Pressed)
    @work
    async def on_save_selected(self):
        target = self.query_one(Input).value
        if not target:
            self.notify("Must specify a filename to save to", severity="error", timeout=3)
            self.query_one(Input).focus()
            return
        target_path = pathlib.Path(target)
        if target_path.is_dir():
            self.notify(f"Must specify a filename to save to in directory {target}", severity="error", timeout=3)
            self.query_one(Input).focus()
            return

        if target_path.suffix.lstrip(".") not in SaveModal.SUPPORTED_EXTS:
            self.notify(
                f"Target filename must end with one of {'/'.join(SaveModal.SUPPORTED_EXTS)}",
                severity="error",
                timeout=3,
            )
            self.query_one(Input).focus()
            return

        if target_path.exists():
            confirm = await self.app.push_screen_wait(OverwriteWarning(target_path))
            if not confirm:
                self.query_one(Input).focus()
                return

        self.dismiss(target_path)
