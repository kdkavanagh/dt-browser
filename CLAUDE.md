# CLAUDE.md — dt-browser

## What is this?

A terminal UI (TUI) for interactively browsing, searching, filtering, and exporting tabular data. Built with [Textual](https://textual.textualize.io/) and [Polars](https://pola.rs/). Designed for Apache Arrow files but supports CSV, JSON, and Parquet too.

**CLI entry point**: `dtbrowser <file_path>` → `dt_browser.browser:run` (Click command)

## Build & Development

```bash
# Setup virtualenv + install (uses uv)
make activate

# Format (isort + black), lint (pylint), typecheck (mypy)
make check

# Individual steps
make .make.formatted   # isort + black
make .make.linted      # pylint
make .make.typed       # mypy
```

- Python >= 3.11, targets py312
- black line-length: 120
- No test suite currently

## Project Structure

```
src/dt_browser/
├── __init__.py          # Base classes (HasState, ReceivesTableSelect, ReactiveLabel), constants (COLORS, INDEX_COL, COLOR_COL)
├── browser.py           # Main app (DtBrowserApp), main widget (DtBrowser), TableWithBookmarks, TableFooter, RowDetail
├── custom_table.py      # Core table rendering widget (CustomTable) — lazy Polars-based rendering
├── bookmarks.py         # Bookmark management widget
├── filter_box.py        # Filter/search input with SQL history (~/.cache/dtbrowser/filters.txt)
├── column_selector.py   # Column visibility, reorder, color-by, and timestamp selection
├── save_df_modal.py     # Modal for exporting filtered data to disk
└── suggestor.py         # Column name autocomplete for FilterBox
```

## Architecture

### Data flow

```
File → from_file_path() → Polars DataFrame
  → DtBrowser manages three dataframes:
    _original_dt  — full unfiltered data
    _filtered_dt  — after SQL WHERE filter applied
    _display_dt   — filtered + only visible/reordered columns
    _meta_dt      — metadata columns (__index, __color)
  → CustomTable renders visible rows/columns lazily via Polars expressions
```

### Key patterns

- **Message-driven**: Widgets post Textual messages (e.g., `FilterSubmitted`, `CellSelected`, `ColumnSelectionChanged`), handled via `@on` decorators
- **Reactive properties**: Textual `reactive()` triggers watcher methods on change; watchers can be async
- **Exclusive workers**: `@work(exclusive=True)` for filter/search/color operations — prevents races
- **SQL filtering**: Polars `SQLContext` executes user-typed WHERE clauses directly
- **Lazy rendering**: Only visible columns/rows are materialized; column widths cached
- **Metadata separation**: Internal columns (`__index`, `__color`) kept in `_meta_dt`, separate from display data
- **LRU cache**: Color-by computations cached (5 entries) by column tuple

### CustomTable (custom_table.py)

The core rendering component. Extends Textual's `ScrollView` — does NOT use Textual's built-in DataTable. Key reasons: lazy column rendering, custom cursor modes (ROW/CELL/NONE), Rich Segment-level control for coloring/highlighting.

Key methods:
- `set_dt(dt, metadata_dt)` — load data
- `render_header_and_table` property — builds lazy Polars query for visible region
- `_gen_segments()` — converts rows to Rich Segments with styling
- `_measure(arr)` — calculates column width for a Series
- `can_draw(arr)` — checks if a dtype is renderable

### DtBrowser (browser.py)

Orchestrates everything. Manages dataframe state, mounts/unmounts child widgets (FilterBox, ColumnSelector, Bookmarks, RowDetail, SaveModal), handles keyboard actions.

### Supported file formats

Read: `.arrow`, `.feather`, `.arrows`, `.arrowstream`, `.json`, `.csv`, `.parquet`
Write: `.arrow`, `.feather`, `.arrows`, `.json`, `.parquet`, `.pqt`, `.csv`

## Key Bindings

| Key | Action |
|-----|--------|
| `f` | Filter (SQL WHERE) |
| `/` | Search (SQL WHERE) |
| `n` / `N` | Next/prev search result |
| `b` | Toggle bookmark |
| `B` | Show bookmarks |
| `c` | Column visibility/reorder |
| `C` | Color by columns |
| `t` | Timestamp conversion |
| `r` | Toggle row detail pane |
| `g` / `G` | Jump to first/last row |
| `Ctrl+S` | Save filtered data |

## Conventions

- Internal/metadata column names start with `__` (e.g., `__index`, `__color`)
- `DISPLAY_IDX_COL = "_display_index"` used for display row numbering
- Filter history persisted at `~/.cache/dtbrowser/filters.txt` (last 100 entries)
- Mouse is disabled (`app.run(mouse=False)`)
- ColumnSelector is reused for three purposes (columns, colors, timestamps) via different widget IDs
