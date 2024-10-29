from __future__ import annotations

import datetime
from enum import Enum, auto
from itertools import accumulate
from typing import Any, ClassVar, cast

import polars as pl
import polars.datatypes as pld
import rich.repr
from polars.interchange.protocol import Column
from rich.align import Align
from rich.console import Console, RenderableType
from rich.errors import MarkupError
from rich.markup import escape
from rich.protocol import is_renderable
from rich.segment import Segment
from rich.style import Style
from rich.text import Text
from textual import events
from textual.coordinate import Coordinate
from textual.geometry import Region, Size
from textual.message import Message
from textual.reactive import Reactive
from textual.scroll_view import ScrollView
from textual.strip import Strip

from dt_browser import COLOR_COL, COLORS, DISPLAY_IDX_COL, INDEX_COL


def cell_formatter(obj: object, null_rep: Text, col: Column | None = None) -> RenderableType:
    """Convert a cell into a Rich renderable for display.

    For correct formatting, clients should call `locale.setlocale()` first.

    Args:
        obj: Data for a cell.
        col: Column that the cell came from (used to compute width).

    Returns:
        A renderable to be displayed which represents the data.
    """
    if obj is None:
        return Align(null_rep, align="center")
    if isinstance(obj, str):
        try:
            rich_text: Text | str = Text.from_markup(obj)
        except MarkupError:
            rich_text = escape(obj)
        return rich_text
    if isinstance(obj, bool):
        return Align(
            f"[dim]{'✓' if obj else 'X'}[/] {obj}{' ' if obj else ''}",
            style="bold" if obj else "",
            align="right",
        )
    if isinstance(obj, (float, pl.Decimal)):
        return Align(f"{obj:n}", align="right")
    if isinstance(obj, int):
        if col is not None and col.is_id:
            # no separators in ID fields
            return Align(str(obj), align="right")
        return Align(f"{obj:n}", align="right")
    if isinstance(obj, (datetime.datetime, datetime.time)):
        return Align(obj.isoformat(timespec="milliseconds").replace("+00:00", "Z"), align="right")
    if isinstance(obj, datetime.date):
        return Align(obj.isoformat(), align="right")
    if isinstance(obj, datetime.timedelta):
        return Align(str(obj), align="right")
    if not is_renderable(obj):
        return str(obj)

    return cast(RenderableType, obj)


def measure_width(obj: object, console: Console) -> int:
    renderable = cell_formatter(obj, null_rep=Text(""))
    return console.measure(renderable).maximum


HEADER_HEIGHT = 1
COL_PADDING = 1
PADDING_STR = " " * COL_PADDING


class CustomTable(ScrollView, can_focus=True, inherit_bindings=False):

    class CursorType(Enum):
        ROW = auto()
        CELL = auto()
        NONE = auto()

    class CellHighlighted(Message):
        """Posted when the cursor moves to highlight a new cell.

        This is only relevant when the `cursor_type` is `"cell"`.
        It's also posted when the cell cursor is
        re-enabled (by setting `show_cursor=True`), and when the cursor type is
        changed to `"cell"`. Can be handled using `on_data_table_cell_highlighted` in
        a subclass of `DataTable` or in a parent widget in the DOM.
        """

        def __init__(
            self,
            data_table: CustomTable,
            value: Any,
            coordinate: Coordinate,
        ) -> None:
            self.data_table = data_table
            """The data table."""
            self.value: Any = value
            """The value in the highlighted cell."""
            self.coordinate: Coordinate = coordinate
            """The coordinate of the highlighted cell."""
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            yield "value", self.value
            yield "coordinate", self.coordinate

        @property
        def control(self) -> CustomTable:
            """Alias for the data table."""
            return self.data_table

    class CellSelected(Message):
        """Posted by the `DataTable` widget when a cell is selected.

        This is only relevant when the `cursor_type` is `"cell"`. Can be handled using
        `on_data_table_cell_selected` in a subclass of `DataTable` or in a parent
        widget in the DOM.
        """

        def __init__(
            self,
            data_table: CustomTable,
            value: Any,
            coordinate: Coordinate,
        ) -> None:
            self.data_table = data_table
            """The data table."""
            self.value: Any = value
            """The value in the highlighted cell."""
            self.coordinate: Coordinate = coordinate
            """The coordinate of the highlighted cell."""
            super().__init__()

        def __rich_repr__(self) -> rich.repr.Result:
            yield "value", self.value
            yield "coordinate", self.coordinate

        @property
        def control(self) -> CustomTable:
            """Alias for the data table."""
            return self.data_table

    DEFAULT_CSS = """
    CustomTable:dark {
        background: initial;
    }
    CustomTable {
        background: $surface ;
        color: $text;
    }
    CustomTable > .datatable--header {
        text-style: bold;
        background: $primary;
        color: $text;
    }
    CustomTable > .datatable--cursor {
        background: $secondary;
        color: $text;
    }
    CustomTable > .datatable--even-row {
        background: $primary-background-lighten-3  10%;
    }
    """

    BINDINGS: ClassVar[list] = []

    COMPONENT_CLASSES: ClassVar[set[str]] = {"datatable--header", "datatable--cursor", "datatable--even-row"}

    cursor_coordinate: Reactive[Coordinate] = Reactive(Coordinate(0, 0), repaint=False)

    def __init__(
        self, dt: pl.DataFrame, metadata_dt: pl.DataFrame, *args, cursor_type: CustomTable.CursorType, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._dt = dt
        self._metadata_dt = metadata_dt
        self._cursor_type = cursor_type

        self._lines = list[Strip]()
        self._widths: dict[str, int] = {}
        self._cum_widths: dict[str, int] = {}
        self._formatters: dict[str, pl.Expr] = {}

        self._cell_highlight: Style | None = None
        self._header_style: Style | None = None
        self._row_col_highlight: Style | None = None

        self._header: dict[str, Segment] = {}
        self._header_pad: list[Segment] = []

        self._render_header_and_table: tuple[Strip, pl.DataFrame] | None = None

        self.set_dt(dt, metadata_dt)

    def on_mount(self):
        self._cell_highlight = self.get_component_rich_style("datatable--cursor")
        self._header_style = self.get_component_rich_style("datatable--header")
        self._row_col_highlight = self.get_component_rich_style("datatable--even-row")
        self._header_pad = [Segment(PADDING_STR, style=self._header_style)]
        self._build_header_contents()

    def _build_header_contents(self):
        self._header: dict[str, Segment] = (
            {}
            if not self._dt.columns
            else {
                x.strip(): Segment(f"{PADDING_STR}{x}", style=self._header_style)
                for x in (
                    pl.DataFrame({x: [x] for x in self._dt.columns})
                    .with_columns(self._formatters.values())
                    .slice(0, 1)
                    .transpose()["column_0"]
                )
            }
        )

        _, header_width = self._build_base_header(self._dt.columns)
        self.virtual_size = Size(header_width, len(self._dt) + HEADER_HEIGHT)

    def set_metadata(self, metadata_dt: pl.DataFrame):
        self._metadata_dt = metadata_dt
        self._render_header_and_table = None
        self.refresh(repaint=True)

    def set_dt(self, dt: pl.DataFrame, metadata_dt: pl.DataFrame):
        self._dt = dt
        self._metadata_dt = metadata_dt
        self._widths = {x: max(len(x), self._measure(self._dt[x])) for x in self._dt.columns}
        self._cum_widths = {
            k: v - (self._widths[k] + COL_PADDING)
            for k, v in zip(self._dt.columns, accumulate(x + COL_PADDING for x in self._widths.values()))
        }
        self._render_header_and_table = None
        self._formatters = {x: self._build_cast_expr(x) for x in self._dt.columns}
        self._build_header_contents()
        self.scroll_to(0, 0, animate=False)

    def render_line(self, y, *_):
        if y >= len(self._lines):
            pad = " " * (self.content_region.width)
            return Strip([Segment(pad)])
        return self._lines[y]

    def _find_minimal_x_offset(self, coordinate: Coordinate):
        col_name = self._dt.columns[coordinate.column]
        effective_width = self.scrollable_content_region.width - COL_PADDING
        free_space = effective_width - (self._widths[col_name] + COL_PADDING)

        idx = self.cursor_coordinate.column - 1
        x_offset = self._cum_widths[col_name]
        while idx >= 0:
            test_col = self._dt.columns[idx]
            free_space -= self._widths[test_col] + COL_PADDING
            if free_space >= 0:
                idx -= 1
                x_offset = self._cum_widths[test_col]
            else:
                break
        return x_offset

    def _is_col_visible(self, column: int):
        x_offset = self.scroll_offset.x
        col_name = self._dt.columns[column]
        needed_x_offset = self._cum_widths[col_name]
        needed_max_x = needed_x_offset + self._widths[col_name]
        if x_offset >= needed_x_offset or needed_max_x >= (x_offset + self.scrollable_content_region.width):
            return False
        return True

    def _is_coordinate_visible(self, coordinate: Coordinate):
        y_offset = self.scroll_offset.y
        row_offset = coordinate.row - y_offset
        if row_offset < 0 or row_offset >= (self.window_region.height - 2):
            return False
        return self._is_col_visible(coordinate.column)

    def move_cursor(self, column: int | None, row: int | None):
        self.go_to_cell(
            Coordinate(
                row=self.cursor_coordinate.row if row is None else row,
                column=self.cursor_coordinate.column if column is None else column,
            )
        )

    def go_to_cell(self, coordinate: Coordinate):
        cur_visible = self._is_coordinate_visible(coordinate)
        if coordinate.column != self.cursor_coordinate.column:
            # Any col change requires re-gening col strings due to concat tuples changing
            self._render_header_and_table = None
        elif not cur_visible:
            # rengen if row not currently displayed
            self._render_header_and_table = None

        self.cursor_coordinate = coordinate

        # If it was off-screen, scroll to it, else refresh just to update the coloring
        if not cur_visible:
            self.app.log(
                f"Cell {coordinate} was determined to not be visible in scroll offset={self.scroll_offset}, scrolling..."
            )
            self.scroll_to(y=coordinate.row, x=self._find_minimal_x_offset(coordinate), animate=False)
        else:
            self.app.log(
                f"Currently visible, no need to scorll.  Coord={self.cursor_coordinate} isNone={self._render_header_and_table is None}"
            )
            self.refresh(repaint=True)

    def on_resize(self, event: events.Resize):
        # Check maxmimal selection of new size
        if event.size.width == 0:
            return
        self._ensure_cursor()

    def _ensure_cursor(self, allow_refresh: bool = True):
        self._render_header_and_table = None

        max_idx = self.cursor_coordinate.column
        while not self._is_col_visible(max_idx) and max_idx > 0:
            max_idx -= 1

        if max_idx != self.cursor_coordinate.column or self.scrollable_content_region.height < (
            self.cursor_coordinate.row - self.scroll_offset.y
        ):
            cur_row = self.cursor_coordinate.row
            max_row = self.scroll_offset[1] + (self.scrollable_content_region.height - 1)
            self.app.log(
                f"Going to cell due to resize.  coord={self.cursor_coordinate}, NewCol={min(max_idx, self.cursor_coordinate.column)} MaxIdx={max_idx}, height={self.scrollable_content_region.height}"
            )
            self.go_to_cell(Coordinate(row=min(cur_row, max_row), column=min(max_idx, self.cursor_coordinate.column)))
        elif allow_refresh:
            self.app.log(f"Regenerating on resize. MaxIdx={max_idx}, cursor={self.cursor_coordinate}")
            self.refresh(repaint=True)

    def _post_cell_event(self, event_type: type[CustomTable.CellHighlighted | CustomTable.CellSelected]):
        col_name = self._dt.columns[self.cursor_coordinate.column]
        self.post_message(
            event_type(
                self,
                self._dt[self.cursor_coordinate.row, col_name],
                self.cursor_coordinate,
            )
        )

    def watch_cursor_coordinate(self):
        self._post_cell_event(CustomTable.CellHighlighted)

    def on_key(self, event: events.Key) -> None:
        x_offset, y_offset = self.scroll_offset
        requires_prep = True
        effective_height = self.scrollable_content_region.height - HEADER_HEIGHT
        old_cursor = self.cursor_coordinate
        match event.key:
            case "down":
                self.cursor_coordinate = Coordinate(
                    min(len(self._dt) - 1, self.cursor_coordinate.row + 1), self.cursor_coordinate.column
                )
                requires_prep = False
            case "up":
                self.cursor_coordinate = Coordinate(
                    max(0, self.cursor_coordinate.row - 1), self.cursor_coordinate.column
                )
                requires_prep = False
            case "pageup":
                self.cursor_coordinate = Coordinate(
                    max(0, self.cursor_coordinate.row - effective_height), self.cursor_coordinate.column
                )
            case "pagedown":
                self.cursor_coordinate = Coordinate(
                    min(len(self._dt) - 1, self.cursor_coordinate.row + effective_height),
                    self.cursor_coordinate.column,
                )
            case "left":
                if self._cursor_type == CustomTable.CursorType.CELL:
                    self.cursor_coordinate = Coordinate(
                        self.cursor_coordinate.row, max(0, self.cursor_coordinate.column - 1)
                    )
                    if self._cum_widths[self._dt.columns[self.cursor_coordinate.column]] < x_offset:
                        x_offset = self._cum_widths[self._dt.columns[self.cursor_coordinate.column]]
            case "right":
                if self._cursor_type == CustomTable.CursorType.CELL:
                    self.cursor_coordinate = Coordinate(
                        self.cursor_coordinate.row, min(len(self._dt.columns) - 1, self.cursor_coordinate.column + 1)
                    )
                    col_name = self._dt.columns[self.cursor_coordinate.column]
                    max_offset = self._cum_widths[col_name] + self._widths[col_name]
                    effective_width = self.scrollable_content_region.width
                    if max_offset >= x_offset + effective_width:
                        x_offset = self._find_minimal_x_offset(self.cursor_coordinate)

            case "home":
                self.cursor_coordinate = Coordinate(0, 0)
                x_offset = 0
            case "end":
                self.cursor_coordinate = Coordinate(len(self._dt) - 1, 0)
                x_offset = 0
            case "enter":
                self._post_cell_event(CustomTable.CellSelected)
                event.stop()
                return
            case _:
                return

        y_offset += self.cursor_coordinate.row - old_cursor.row

        if (
            self.cursor_coordinate.row
            >= (self.scroll_offset[1] + self.scrollable_content_region.height - HEADER_HEIGHT)
            or self.cursor_coordinate.row < self.scroll_offset[1]
            or x_offset != self.scroll_offset[0]
        ):
            self._render_header_and_table = None
            self.scroll_to(y=y_offset, x=x_offset, animate=False)

        else:
            if requires_prep:
                self._render_header_and_table = None
            self.refresh(repaint=True)
        event.stop()

    def _build_cast_expr(self, col: str):
        dtype = self._dt[col].dtype
        if dtype == pld.Categorical():
            dtype = pl.Utf8
        if dtype.is_numeric() or dtype.is_temporal():
            return pl.col(col).cast(pl.Utf8).fill_null("").str.pad_start(self._widths[col])
        return pl.col(col).cast(pl.Utf8).fill_null("").str.pad_end(self._widths[col])

    def _build_base_header(self, cols_to_render: list[str]):
        base_header = [v for k, v in self._header.items() if k in cols_to_render] + self._header_pad
        header_width = sum(len(x.text) for x in base_header)
        return (base_header, header_width)

    @property
    def render_header_and_table(self):
        if self._render_header_and_table is None:
            scroll_x, scroll_y = self.scroll_offset

            cols_to_render: list[str] = []
            min_col_idx: int | None = None
            effective_width = self.scrollable_content_region.width
            for i, x in enumerate(self._dt.columns):
                min_offset = self._cum_widths[x] - scroll_x
                max_offset = min_offset + self._widths[x]
                if min_offset < 0:
                    continue

                if max_offset >= effective_width:
                    break

                cols_to_render.append(x)
                if min_col_idx is None:
                    min_col_idx = i
            if not cols_to_render:
                return (Strip([]), pl.DataFrame())

            if min_col_idx is None:
                min_col_idx = 0

            dt_height = self.window_region.height - HEADER_HEIGHT

            base_header, header_width = self._build_base_header(cols_to_render)
            excess = self.scrollable_content_region.width - header_width
            header = Strip(base_header + (self._header_pad * (excess)))
            theo_max_offset = self._cum_widths[cols_to_render[0]] + effective_width

            rend = (
                self._dt.lazy()
                .select(cols_to_render)
                .with_columns(self._metadata_dt[INDEX_COL])
                .slice(scroll_y, dt_height)
                .with_row_index(DISPLAY_IDX_COL)
            )

            visible_cols = cols_to_render.copy()
            first_col_prefix_padding = COL_PADDING
            needed_padding = theo_max_offset - self._cum_widths[visible_cols[-1]] - first_col_prefix_padding

            if COLOR_COL in self._metadata_dt.columns:
                row_colors = self._metadata_dt.slice(scroll_y, dt_height).select(
                    pl.col(COLOR_COL).cast(COLORS).alias(COLOR_COL)
                )
                cols_to_render.insert(0, COLOR_COL)
                rend = rend.with_columns(row_colors)

            def build_selector(cols: list[str], needed_padding: int = 0):
                if not cols:
                    return pl.lit("").str.pad_end(needed_padding)

                fmts = [self._formatters[x] for x in cols]
                if needed_padding:
                    fmts[-1] = fmts[-1].str.pad_end(needed_padding - COL_PADDING, fill_char=PADDING_STR[0])
                concat = pl.concat_str(fmts, separator=PADDING_STR, ignore_nulls=True)

                return pl.concat_str(concat, pl.lit(PADDING_STR))

            if self._cursor_type in (CustomTable.CursorType.ROW, CustomTable.CursorType.NONE):
                rend = rend.select(
                    pl.col(DISPLAY_IDX_COL),
                    pl.col(INDEX_COL),
                    (
                        pl.col(COLOR_COL)
                        if COLOR_COL in cols_to_render
                        else pl.repeat(None, pl.len(), dtype=pl.Null).alias(COLOR_COL)
                    ),
                    before_selected=build_selector(visible_cols, needed_padding),
                    selected=pl.lit(""),
                    after_selected=pl.lit(""),
                )
            else:

                cursor_col_idx = self.cursor_coordinate.column - min_col_idx

                cols_before_selected: list[str] = visible_cols[0:cursor_col_idx]
                sel_col = visible_cols[cursor_col_idx]
                cols_after_selected = visible_cols[cursor_col_idx + 1 :]

                rend = rend.select(
                    pl.col(DISPLAY_IDX_COL),
                    pl.col(INDEX_COL),
                    (
                        pl.col(COLOR_COL)
                        if COLOR_COL in cols_to_render
                        else pl.repeat(None, pl.len(), dtype=pl.Null).alias(COLOR_COL)
                    ),
                    before_selected=build_selector(cols_before_selected),
                    selected=build_selector([sel_col]),
                    after_selected=build_selector(cols_after_selected, needed_padding),
                )

            self._render_header_and_table = (header, rend.collect())
        return self._render_header_and_table

    def _get_row_bg_color_expr(self, cursor_row_idx: int) -> pl.Expr:

        return (
            pl.when(pl.col(DISPLAY_IDX_COL) == cursor_row_idx)
            .then(
                pl.lit(
                    self._row_col_highlight.bgcolor.name
                    if self._cursor_type == CustomTable.CursorType.CELL
                    else self._cell_highlight.bgcolor.name
                )
            )
            .otherwise(pl.lit(None))
        )

    def _get_sel_col_bg_color(self, struct: pl.Struct):
        return (
            self._row_col_highlight.bgcolor.name
            if self._cursor_type == CustomTable.CursorType.CELL
            else struct["bgcolor"]
        )

    def render_lines(self, crop: Region):
        # if self._render_header_and_table and self.scrollable_content_region.height - HEADER_HEIGHT != len(
        #     self._render_header_and_table
        # ):
        #     # if we get re-rendered before the resize event is passed to us
        #     print(f"Ensure during render")
        #     self._ensure_cursor(allow_refresh=False)
        #     self._render_header_and_table = None
        cur_header, render_df = self.render_header_and_table

        self._lines.clear()
        self._lines.append(cur_header)
        if not render_df.is_empty():
            rend = render_df.lazy()
            if self._cursor_type == CustomTable.CursorType.NONE:
                rend = rend.select(
                    segments=pl.col("before_selected").map_elements(
                        lambda x: [Segment(PADDING_STR), Segment(x)],
                        return_dtype=pl.Object,
                    )
                )
            else:
                _, scroll_y = self.scroll_offset
                cursor_row_idx = self.cursor_coordinate.row - scroll_y
                rend = rend.select(
                    segments=pl.struct(
                        pl.col("*"),
                        self._get_row_bg_color_expr(cursor_row_idx).alias("bgcolor"),
                    ).map_elements(
                        lambda struct: [
                            Segment(
                                PADDING_STR,
                                style=Style(
                                    color=(
                                        self._cell_highlight.color.name
                                        if (
                                            self._cursor_type == CustomTable.CursorType.ROW
                                            and struct[DISPLAY_IDX_COL] == cursor_row_idx
                                        )
                                        else None
                                    ),
                                    bgcolor=struct["bgcolor"],
                                ),
                            ),
                            Segment(
                                struct["before_selected"],
                                style=Style(
                                    color=(
                                        self._cell_highlight.color.name
                                        if (
                                            self._cursor_type == CustomTable.CursorType.ROW
                                            and struct[DISPLAY_IDX_COL] == cursor_row_idx
                                        )
                                        else struct[COLOR_COL]
                                    ),
                                    bgcolor=struct["bgcolor"],
                                ),
                            ),
                            Segment(
                                struct["selected"],
                                style=Style(
                                    color=(
                                        self._cell_highlight.color.name
                                        if struct[DISPLAY_IDX_COL] == cursor_row_idx
                                        else struct[COLOR_COL]
                                    ),
                                    bgcolor=(
                                        self._cell_highlight.bgcolor.name
                                        if struct[DISPLAY_IDX_COL] == cursor_row_idx
                                        else self._get_sel_col_bg_color(struct)
                                    ),
                                ),
                            ),
                            Segment(
                                struct["after_selected"],
                                style=Style(
                                    color=(
                                        self._cell_highlight.color.name
                                        if (
                                            self._cursor_type == CustomTable.CursorType.ROW
                                            and struct[DISPLAY_IDX_COL] == cursor_row_idx
                                        )
                                        else struct[COLOR_COL]
                                    ),
                                    bgcolor=struct["bgcolor"],
                                ),
                            ),
                        ],
                        return_dtype=pl.Object,
                    )
                )

            self._lines.extend(
                Strip(x, cell_length=self.scrollable_content_region.width) for x in rend.collect()["segments"]
            )

        return super().render_lines(crop)

    def _measure(self, arr: pl.Series) -> int:
        # with some types we can measure the width more efficiently
        dtype = arr.dtype
        if dtype == pld.Categorical():
            if arr.cat.get_categories().is_empty():
                return len("<null>")
            return self._measure(arr.cat.get_categories())

        if dtype.is_integer():
            col_max = arr.max()
            col_min = arr.min()
            return max(measure_width(el, self._console) for el in [col_max, col_min])
        if dtype.is_temporal():
            try:
                value = arr.drop_nulls().slice(0, 1).cast(pl.Utf8)[0]
            except IndexError:
                return 0
            return measure_width(value, self._console)
        if dtype.is_(pld.Boolean()):
            return 7

        # for everything else, we need to compute it
        width = arr.cast(pl.Utf8(), strict=False).fill_null("<null>").str.len_chars().max()
        assert isinstance(width, int)
        return width
