import polars as pl
from rich.table import Table as RichTable
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Static


def _categorical_stats(series: pl.Series) -> list[tuple[str, str]]:
    n_unique = series.n_unique()
    stats: list[tuple[str, str]] = [("Unique values", str(n_unique))]
    vc = series.value_counts().sort("count", descending=True).head(10)
    col_name = vc.columns[0]
    for row in vc.iter_rows(named=True):
        stats.append((f"  {row[col_name]}", str(row["count"])))
    return stats


def _numeric_stats(series: pl.Series) -> list[tuple[str, str]]:
    s = series.drop_nulls()
    if s.is_empty():
        return [("", "No data")]
    return [
        ("Min", str(s.min())),
        ("Q1", str(s.quantile(0.25))),
        ("Median", str(s.median())),
        ("Q3", str(s.quantile(0.75))),
        ("Max", str(s.max())),
    ]


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
    stats: list[tuple[str, str]] = [
        ("True", str(true_count)),
        ("False", str(false_count)),
    ]
    if null_count > 0:
        stats.append(("Null", str(null_count)))
    return stats


def compute_column_stats(series: pl.Series) -> list[tuple[str, str]]:
    dtype = series.dtype
    if dtype == pl.Categorical:
        return _categorical_stats(series)
    if dtype.is_numeric():
        return _numeric_stats(series)
    if dtype.is_temporal():
        return _temporal_stats(series)
    if dtype.is_(pl.Boolean):
        return _boolean_stats(series)
    return []


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

    def watch_column_info(self) -> None:
        if self.column_info is None or self._source_df.is_empty():
            return
        col_name, _ = self.column_info
        if col_name not in self._source_df.columns:
            return
        if col_name not in self._cache:
            self._cache[col_name] = compute_column_stats(self._source_df[col_name])
        stats = self._cache[col_name]
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

    def compose(self):
        yield self._static
