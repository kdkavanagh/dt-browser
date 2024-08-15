from pathlib import Path
from typing import Any, Iterable

import polars as pl
import polars.datatypes as pld
from textual_fastdatatable.backend import PolarsBackend as _BasePolarsBackend


class PolarsBackend(_BasePolarsBackend):

    @classmethod
    def from_file_path(cls, path: Path, max_rows: int | None = None, has_header: bool = True) -> "PolarsBackend":
        if path.suffix in [".arrow", ".feather"]:
            tbl = pl.read_ipc(path)
        elif path.suffix == ".arrows":
            tbl = pl.read_ipc_stream(path)
        elif path.suffix == ".json":
            tbl = pl.read_json(path)
        elif path.suffix == ".csv":
            tbl = pl.read_csv(path, has_header=has_header)
        elif path.suffix == ".parquet":
            tbl = pl.read_parquet(path)
        else:
            raise TypeError(f"Dont know how to load file type {path.suffix} for {path}")
        return cls(tbl, max_rows=max_rows)

    def append_rows(self, records: Iterable[Iterable[Any]]) -> list[int]:
        if isinstance(records, pl.DataFrame):
            rows_to_add = records
        else:
            rows_to_add = pl.from_dicts([dict(zip(self.data.columns, row)) for row in records])
        indicies = list(range(self.row_count, self.row_count + len(rows_to_add)))
        self.data = pl.concat([self.data, rows_to_add], how="diagonal")
        self._reset_content_widths()
        return indicies

    def _measure(self, arr: pl.Series) -> int:
        dtype = arr.dtype
        if dtype == pld.Categorical() and arr.cat.get_categories().is_empty():
            return len("<null>")
        return super()._measure(arr)
