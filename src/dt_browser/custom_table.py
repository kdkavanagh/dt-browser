import datetime
import re
from itertools import accumulate
from typing import ClassVar, cast

import polars as pl
import polars.datatypes as pld
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
from textual.reactive import Reactive
from textual.scroll_view import ScrollView
from textual.strip import Strip

from dt_browser import COLOR_COL, COLORS


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
        else:
            return Align(f"{obj:n}", align="right")
    if isinstance(obj, (datetime, datetime.time)):
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


def _get_color_escape(hex_str: str | tuple[int, int, int], background=False):
    if isinstance(hex_str, str):
        r, g, b = [int(x, 16) for x in re.findall("..", hex_str.removeprefix("#"))]
    else:
        r, g, b = hex_str
    return "\033[{};2;{};{};{}m".format(48 if background else 38, r, g, b)


_colors = pl.Enum((_get_color_escape(x) for x in COLORS.categories))


class CustomTable(ScrollView, can_focus=True, inherit_bindings=False):
    DEFAULT_CSS = """
    CustomTable:dark {
        background: initial;
    }
    CustomTable {
        background: $surface ;
        color: $text;
        height: auto;
        max-height: 100vh;
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

    cursor_coordinate: Reactive[Coordinate] = Reactive(Coordinate(0, 0), repaint=False, always_update=True)

    def __init__(self, dt: pl.DataFrame, metadata_dt: pl.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dt = dt
        self._metadata_dt = metadata_dt

        self._lines = list[Strip]()
        self._widths = {x: max(len(x), self._measure(self._dt[x])) for x in self._dt.columns}
        self._cum_widths = {
            k: v - (self._widths[k] + 1)
            for k, v in zip(self._dt.columns, accumulate(x + 1 for x in self._widths.values()))
        }
        self._needs_prep = True

        self._formatters = {x: self._build_cast_expr(x) for x in self._dt.columns}

        self._color_reset = "\033[0m"

    def on_mount(self):
        self._cell_highlight = self.get_component_rich_style("datatable--cursor")
        self._header_style = self.get_component_rich_style("datatable--header")
        self._row_col_highlight = self.get_component_rich_style("datatable--even-row")
        self._header = {
            x.strip(): Segment(f" {x}", style=self._header_style)
            for x in (
                pl.DataFrame({x: [x] for x in self._dt.columns})
                .with_columns(self._formatters.values())[0]
                .transpose()["column_0"]
            )
        }
        self._header_pad = [Segment(" ", style=self._header_style)]

        _, header_width = self._build_base_header(self._dt.columns)

        self.virtual_size = Size(header_width, len(self._dt))

    def render_line(self, y, *_):
        return self._lines[y]

    def _find_minimal_x_offset(self, coordinate: Coordinate):
        col_name = self._dt.columns[coordinate.column]
        effective_width = self.container_size.width - 2
        padding = 1
        free_space = effective_width - (self._widths[col_name] + padding)

        idx = self.cursor_coordinate.column - 1
        x_offset = self._cum_widths[col_name]
        while idx >= 0:
            test_col = self._dt.columns[idx]
            free_space -= self._widths[test_col] + padding
            if free_space >= 0:
                idx -= 1
                x_offset = self._cum_widths[test_col]
            else:
                break
        return x_offset

    def _is_coordinate_visible(self, coordinate: Coordinate):
        x_offset, y_offset = self.scroll_offset
        row_offset = coordinate.row - y_offset
        if row_offset < 0 or row_offset >= (self.window_region.height - 2):
            return False
        col_name = self._dt.columns[coordinate.column]
        needed_x_offset = self._cum_widths[col_name]
        needed_max_x = needed_x_offset + self._widths[col_name]
        if x_offset >= needed_x_offset and needed_max_x <= self.window_region.width - 2:
            return False
        return True

    def go_to_cell(self, coordinate: Coordinate):
        cur_visible = self._is_coordinate_visible(coordinate)
        if coordinate.column != self.cursor_coordinate.column:
            # Any col change requires re-gening col strings due to concat tuples changing
            self._needs_prep = True
        else:
            # rengen if row not currently displayed
            self._needs_prep = not cur_visible

        self.cursor_coordinate = coordinate

        # If it was off-screen, scroll to it, else refresh
        if not cur_visible:
            self.scroll_to(y=coordinate.row, x=self._find_minimal_x_offset(coordinate), animate=False)
        else:
            self.refresh(repaint=True)

    def on_resize(self, event: events.Resize):
        # Check maxmimal selection of new size
        max_idx = 0
        for i, (k, x) in enumerate(self._cum_widths.items()):
            if (x + self._widths[k]) >= (event.size.width - 2):
                max_idx = i - 1
                break

        if max_idx < self.cursor_coordinate.column or event.size.height < self.cursor_coordinate.row:
            cur_row = self.cursor_coordinate.row
            max_row = self.scroll_offset[1] + (event.size.height - 1)
            self.go_to_cell(Coordinate(row=min(cur_row, max_row), column=min(max_idx, self.cursor_coordinate.column)))
        else:
            self._needs_prep = True
            self.refresh(repaint=True)

    def on_key(self, event: events.Key) -> None:
        x_offset, y_offset = self.scroll_offset
        requires_prep = True
        match event.key:
            case "down":
                self.cursor_coordinate = Coordinate(
                    min(len(self._dt), self.cursor_coordinate.row + 1), self.cursor_coordinate.column
                )
                y_offset += 1
                requires_prep = False
            case "up":
                self.cursor_coordinate = Coordinate(
                    max(0, self.cursor_coordinate.row - 1), self.cursor_coordinate.column
                )
                y_offset -= 1
                requires_prep = False
            case "pageup":
                self.cursor_coordinate = Coordinate(
                    max(0, self.cursor_coordinate.row - self.container_size.height), self.cursor_coordinate.column
                )
                y_offset = self.cursor_coordinate.row
            case "pagedown":
                self.cursor_coordinate = Coordinate(
                    min(len(self._dt), self.cursor_coordinate.row + self.container_size.height),
                    self.cursor_coordinate.column,
                )
                y_offset = self.cursor_coordinate.row
            case "left":
                self.cursor_coordinate = Coordinate(
                    self.cursor_coordinate.row, max(0, self.cursor_coordinate.column - 1)
                )
                if self._cum_widths[self._dt.columns[self.cursor_coordinate.column]] < x_offset:
                    x_offset = self._cum_widths[self._dt.columns[self.cursor_coordinate.column]]
            case "right":
                self.cursor_coordinate = Coordinate(
                    self.cursor_coordinate.row, min(len(self._dt.columns) - 1, self.cursor_coordinate.column + 1)
                )
                col_name = self._dt.columns[self.cursor_coordinate.column]
                max_offset = self._cum_widths[col_name] + self._widths[col_name]
                effective_width = self.container_size.width - 2
                if max_offset >= x_offset + effective_width:
                    x_offset = self._find_minimal_x_offset(self.cursor_coordinate)

            case "home":
                self.cursor_coordinate = Coordinate(0, 0)
                x_offset = 0
            case "end":
                self.cursor_coordinate = Coordinate(len(self._dt) - 1, 0)
                x_offset = 0
            case _:
                return

        if (
            self.cursor_coordinate.row >= (self.scroll_offset[1] + self.window_region.height - 2)
            or self.cursor_coordinate.row < self.scroll_offset[1]
            or x_offset != self.scroll_offset[0]
        ):
            self._needs_prep = True
            self.scroll_to(y=y_offset, x=x_offset, animate=False)

        else:
            self._needs_prep |= requires_prep
            self.refresh(repaint=True)
        event.stop()

    def _build_cast_expr(self, col: str):
        dtype = self._dt[col].dtype
        if dtype == pld.Categorical():
            dtype = pl.Utf8
        if dtype.is_numeric() or dtype.is_temporal():
            return pl.col(col).cast(pl.Utf8).str.pad_start(self._widths[col])
        return pl.col(col).cast(pl.Utf8).str.pad_end(self._widths[col])

    def _build_base_header(self, cols_to_render: list[str]):
        base_header = [v for k, v in self._header.items() if k in cols_to_render] + self._header_pad
        header_width = sum(len(x.text) for x in base_header)
        return (base_header, header_width)

    def _prepare_col_strings_to_render(self, crop: Region):
        scroll_x, scroll_y = self.scroll_offset
        scroll_bar_width = 2

        cols_to_render: list[str] = []
        min_col_idx: int | None = None

        effective_width = crop.width - scroll_bar_width
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

        cursor_col_idx = self.cursor_coordinate.column - min_col_idx

        dt_height = crop.height - 1
        base_header, header_width = self._build_base_header(cols_to_render)
        excess = crop.width - header_width - scroll_bar_width
        header = Strip(base_header + (self._header_pad * (excess)))

        rend = self._dt.slice(scroll_y, dt_height)

        visible_cols = cols_to_render.copy()

        if COLOR_COL in self._metadata_dt.columns:
            row_colors = self._metadata_dt.slice(scroll_y, dt_height).select(
                COLOR_COL=((pl.col(COLOR_COL) + 1).cast(_colors))
            )
            cols_to_render.insert(0, COLOR_COL)
            rend = rend.with_columns(row_colors)

        # else:
        # row_colors = pl.repeat("", len(rend), eager=True)

        cols_before_selected: list[str] = visible_cols[0:cursor_col_idx]
        sel_col = visible_cols[cursor_col_idx]
        cols_after_selected = visible_cols[cursor_col_idx + 1 :]

        theo_max_offset = scroll_x + effective_width
        needed_padding = theo_max_offset - self._cum_widths[cols_after_selected[0] if cols_after_selected else sel_col]

        def build_selector(cols: list[str], pad: bool):
            if not cols:
                concat = pl.lit("")
                if not pad:
                    return concat
            else:
                concat = pl.concat_str(
                    [self._formatters[x] for x in cols],
                    separator=" ",
                ).fill_null("")
            if pad:
                concat = concat.str.pad_end(needed_padding)

            return pl.concat_str(concat, pl.lit(" "))

        self._rend = (
            rend.lazy()
            .select(cols_to_render)
            .with_row_index()
            .select(
                pl.col("index"),
                (
                    pl.col(COLOR_COL)
                    if COLOR_COL in cols_to_render
                    else pl.repeat(None, pl.len(), dtype=pl.Null).alias(COLOR_COL)
                ),
                before_selected=build_selector(cols_before_selected, False),
                selected=build_selector([sel_col], False),
                after_selected=build_selector(cols_after_selected, True),
            )
            .collect()
        )
        return header

    def render_lines(self, crop: Region):
        if self._needs_prep:
            cur_header = self._prepare_col_strings_to_render(crop)
        else:
            cur_header = self._lines[0]

        self._lines.clear()
        self._lines.append(cur_header)

        assert self._rend is not None
        _, scroll_y = self.scroll_offset
        scroll_bar_width = 2
        cursor_row_idx = self.cursor_coordinate.row - scroll_y
        self._lines.extend(
            Strip(x, cell_length=crop.width - scroll_bar_width)
            for x in self._rend.lazy()
            .select(
                segements=pl.struct(
                    pl.col("*"),
                    pl.when(pl.col("index") == cursor_row_idx)
                    .then(pl.lit(self._row_col_highlight.bgcolor.name))
                    .otherwise(pl.lit(None))
                    .alias("bgcolor"),
                ).map_elements(
                    lambda struct: [
                        Segment(
                            f" ",
                            style=Style(bgcolor=struct["bgcolor"]),
                        ),
                        Segment(
                            f"{struct['before_selected']}",
                            style=Style(color=struct[COLOR_COL], bgcolor=struct["bgcolor"]),
                        ),
                        Segment(
                            struct["selected"],
                            style=Style(
                                color=struct[COLOR_COL],
                                bgcolor=(
                                    self._cell_highlight.bgcolor.name
                                    if struct["index"] == cursor_row_idx
                                    else self._row_col_highlight.bgcolor.name
                                ),
                            ),
                        ),
                        Segment(
                            struct["after_selected"],
                            style=Style(color=struct[COLOR_COL], bgcolor=struct["bgcolor"]),
                        ),
                    ],
                    return_dtype=pl.Object,
                )
            )
            .collect()["segements"]
        )
        for line in self._lines:
            for x in line._segments:
                if x.text is None:
                    raise Exception(f"Bad segment for line {line}")

        return super().render_lines(crop)

    # def render_lines_old(self, crop: Region) -> list[Strip]:
    #     scroll_x, scroll_y = self.scroll_offset
    #     scroll_bar_width = 2
    #     cursor_row_idx = self.cursor_coordinate.row - scroll_y

    #     cols_to_render: list[str] = []
    #     min_col_idx: int | None = None

    #     effective_width = crop.width - scroll_bar_width
    #     for i, x in enumerate(self._dt.columns):
    #         min_offset = self._cum_widths[x] - scroll_x
    #         max_offset = min_offset + self._widths[x]
    #         if min_offset < 0:
    #             continue

    #         if max_offset >= effective_width:
    #             break

    #         cols_to_render.append(x)
    #         if min_col_idx is None:
    #             min_col_idx = i

    #     print(f"Rendering {cols_to_render}")
    #     cursor_col_idx = self.cursor_coordinate.column - min_col_idx

    #     dt_height = crop.height - 1
    #     base_header, header_width = self._build_base_header(cols_to_render)
    #     excess = crop.width - header_width - scroll_bar_width
    #     header = Strip(base_header + (self._header_pad * (excess)))

    #     rend = self._dt.slice(scroll_y, dt_height)

    #     visible_cols = cols_to_render.copy()

    #     if COLOR_COL in self._metadata_dt.columns:
    #         row_colors = self._metadata_dt.slice(scroll_y, dt_height).select(
    #             COLOR_COL=((pl.col(COLOR_COL) + 1).cast(_colors))
    #         )
    #         cols_to_render.insert(0, COLOR_COL)
    #         rend = rend.with_columns(row_colors)

    #     # else:
    #     # row_colors = pl.repeat("", len(rend), eager=True)

    #     cols_before_selected: list[str] = visible_cols[0:cursor_col_idx]
    #     sel_col = visible_cols[cursor_col_idx]
    #     cols_after_selected = visible_cols[cursor_col_idx + 1 :]

    #     theo_max_offset = scroll_x + effective_width
    #     needed_padding = theo_max_offset - self._cum_widths[cols_after_selected[0] if cols_after_selected else sel_col]

    #     def build_selector(cols: list[str], pad: bool):
    #         if not cols:
    #             concat = pl.lit("")
    #             if not pad:
    #                 return concat
    #         else:
    #             concat = pl.concat_str(
    #                 [self._formatters[x] for x in cols],
    #                 separator=" ",
    #             )
    #         if pad:
    #             concat = concat.str.pad_end(needed_padding)

    #         return pl.concat_str(concat, pl.lit(" "))

    #     self._lines.clear()
    #     self._lines.append(header)
    #     self._lines.extend(
    #         Strip(x, cell_length=crop.width - scroll_bar_width)
    #         for x in rend.lazy()
    #         .select(cols_to_render)
    #         .with_row_index()
    #         .select(
    #             segments=pl.struct(
    #                 pl.col("index"),
    #                 pl.when(pl.col("index") == cursor_row_idx)
    #                 .then(pl.lit(self._row_col_highlight.bgcolor.name))
    #                 .otherwise(pl.lit(None))
    #                 .alias("bgcolor"),
    #                 (
    #                     pl.col(COLOR_COL)
    #                     if COLOR_COL in cols_to_render
    #                     else pl.repeat(None, pl.len(), dtype=pl.Null).alias(COLOR_COL)
    #                 ),
    #                 before_selected=build_selector(cols_before_selected, False),
    #                 selected=build_selector([sel_col], False),
    #                 after_selected=build_selector(cols_after_selected, True),
    #             ).map_elements(
    #                 lambda struct: [
    #                     Segment(
    #                         f" ",
    #                         style=Style(bgcolor=struct["bgcolor"]),
    #                     ),
    #                     Segment(
    #                         f"{struct['before_selected']}",
    #                         style=Style(color=struct[COLOR_COL], bgcolor=struct["bgcolor"]),
    #                     ),
    #                     Segment(
    #                         struct["selected"],
    #                         style=Style(
    #                             color=struct[COLOR_COL],
    #                             bgcolor=(
    #                                 self._cell_highlight.bgcolor.name
    #                                 if struct["index"] == cursor_row_idx
    #                                 else self._row_col_highlight.bgcolor.name
    #                             ),
    #                         ),
    #                     ),
    #                     Segment(
    #                         struct["after_selected"],
    #                         style=Style(color=struct[COLOR_COL], bgcolor=struct["bgcolor"]),
    #                     ),
    #                 ],
    #                 return_dtype=pl.Object,
    #             )
    #         )
    #         .collect()["segments"]
    #     )
    #     return super().render_lines(crop)

    def _measure(self, arr: pl.Series) -> int:
        # with some types we can measure the width more efficiently
        dtype = arr.dtype
        if dtype == pld.Categorical():
            if arr.cat.get_categories().is_empty():
                return len("<null>")
            return self._measure(arr.cat.get_categories())

        if dtype.is_decimal() or dtype.is_float() or dtype.is_integer():
            col_max = arr.max()
            col_min = arr.min()
            return max([measure_width(el, self._console) for el in [col_max, col_min]])
        if dtype.is_temporal():
            try:
                value = arr.drop_nulls()[0]
            except IndexError:
                return 0
            else:
                return measure_width(value, self._console)
        if dtype.is_(pld.Boolean()):
            return 7

        # for everything else, we need to compute it

        arr = arr.cast(
            pl.Utf8(),
            strict=False,
        )
        width = arr.fill_null("<null>").str.len_chars().max()
        assert isinstance(width, int)
        return width