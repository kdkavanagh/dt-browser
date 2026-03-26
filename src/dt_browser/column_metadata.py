import polars as pl
from rich.table import Table as RichTable
from textual import work
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


def _categorical_stats(series: pl.Series) -> list[tuple[str, str]]:
    n_unique = series.n_unique()
    stats: list[tuple[str, str]] = [("Unique values", str(n_unique))]
    val_col = series.name
    vc = series.value_counts().sort(["count", val_col], descending=[True, False]).head(10)
    for row in vc.iter_rows(named=True):
        stats.append((f"  {row[val_col]}", str(row["count"])))
    return stats


def _numeric_stats(series: pl.Series) -> list[tuple[str, str]]:
    s = series.drop_nulls()
    if s.is_empty():
        return [("", "No data")]
    stats = [
        ("Min", str(s.min())),
        ("Q1", str(s.quantile(0.25))),
        ("Median", str(s.median())),
        ("Q3", str(s.quantile(0.75))),
        ("Max", str(s.max())),
    ]
    if s.dtype.is_float():
        nan_count = s.is_nan().sum()
        if nan_count > 0:
            stats.append(("NaN", str(nan_count)))
    return stats


def _temporal_stats(series: pl.Series) -> list[tuple[str, str]]:
    s = series.drop_nulls()
    if s.is_empty():
        return [("", "No data")]
    return [
        ("Min", str(s.min())),
        ("Max", str(s.max())),
    ]


def _boolean_stats(series: pl.Series) -> list[tuple[str, str]]:
    true_count = series.sum()
    null_count = series.null_count()
    false_count = len(series) - (true_count or 0) - null_count
    return [
        ("True", str(true_count)),
        ("False", str(false_count)),
    ]


def compute_column_stats(series: pl.Series) -> list[tuple[str, str]]:
    dtype = series.dtype
    if dtype == pl.Categorical:
        stats = _categorical_stats(series)
    elif dtype.is_numeric():
        stats = _numeric_stats(series)
    elif dtype.is_temporal():
        stats = _temporal_stats(series)
    elif dtype.is_(pl.Boolean):
        stats = _boolean_stats(series)
    else:
        return []
    null_count = series.null_count()
    if null_count > 0:
        stats.append(("Null", str(null_count)))
    return stats


class ColumnMetadata(Widget, can_focus=False, can_focus_children=False):
    DEFAULT_CSS = """
ColumnMetadata {
    width: 100%;
    height: auto;
    padding: 0 1;
    border: tall $primary;
}
"""
    column_info: reactive[tuple[str, pl.DataType] | None] = reactive(None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.border_title = "Column Metadata"
        self._source_df: pl.DataFrame = pl.DataFrame()
        self._cache: dict[str, list[tuple[str, str]]] = {}
        self._static = Static("")

    def set_source_df(self, df: pl.DataFrame) -> None:
        self._source_df = df

    def invalidate_cache(self) -> None:
        self._cache.clear()

    def _render_stats(self, col_name: str, stats: list[tuple[str, str]]) -> None:
        self.border_title = f"Column: {col_name}"
        if not stats:
            self._static.update("")
            return
        table = RichTable(show_header=False, box=None, padding=(0, 1), expand=True)
        table.add_column("Stat", no_wrap=True)
        table.add_column("Value", no_wrap=True, justify="right")
        for label, value in stats:
            table.add_row(label, value)
        self._static.update(table)
        if self.parent is not None and hasattr(self.parent, "update_width"):
            self.parent.update_width()

    def watch_column_info(self) -> None:
        if self.column_info is None or self._source_df.is_empty():
            return
        col_name, _ = self.column_info
        if col_name not in self._source_df.columns:
            return
        if col_name in self._cache:
            self._render_stats(col_name, self._cache[col_name])
        else:
            self.border_title = f"Column: {col_name}"
            self._static.update("Computing...")
            self._compute_stats(col_name)

    @work(exclusive=True)
    async def _compute_stats(self, col_name: str) -> None:
        series = self._source_df[col_name]
        stats = compute_column_stats(series)
        self._cache[col_name] = stats
        self._render_stats(col_name, stats)

    def compose(self):
        yield self._static
