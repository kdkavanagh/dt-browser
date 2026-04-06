"""Performance test: scrolling with active search highlights vs without.

Creates a large dataframe (200k rows, ~160 columns) with varied cardinality,
applies searches with different match distributions at different terminal sizes,
then measures scroll performance against the baseline (no search).
Fails if scrolling is >15% slower.
"""

import random
import time

import polars as pl
import pytest

from dt_browser.browser import DtBrowser, DtBrowserApp
from dt_browser.filter_box import FilterBox

_NUM_ROWS = 200_000
_NUM_SCROLL_OPS = 60
_SLOWDOWN_THRESHOLD = 0.15

# ~160 columns at ~20 chars each ≈ 3200 chars, enough to fill 3000-wide terminals
_STR_COLS_PER_GROUP = 30
_NUM_COLS_PER_GROUP = 30

_SEARCH_PARAMS = [
    pytest.param(
        "sentinel_col = 'NEEDLE'",
        1,
        1,
        id="single_match",
    ),
    pytest.param(
        "sparse_col = 1",
        100,
        300,
        id="sparse_far_apart",
    ),
    pytest.param(
        "adjacent_col = 1",
        40_000,
        60_000,
        id="many_adjacent",
    ),
    pytest.param(
        "int_col_0 > 5",
        100_000,
        _NUM_ROWS,
        id="dense_most_rows",
    ),
]

_SCREEN_SIZES = [
    pytest.param((120, 30), id="120x30"),
    pytest.param((800, 200), id="800x200"),
    pytest.param((3000, 1000), id="3000x1000"),
]


def _make_large_df() -> pl.DataFrame:
    random.seed(42)
    data: dict[str, list] = {}

    # 30 low-cardinality string columns (5-20 unique values)
    for i in range(_STR_COLS_PER_GROUP):
        card = random.randint(5, 20)
        choices = [f"cat{i}_{j}" for j in range(card)]
        data[f"low_card_str_{i}"] = [random.choice(choices) for _ in range(_NUM_ROWS)]

    # 30 medium-cardinality string columns (100-500 unique values)
    for i in range(_STR_COLS_PER_GROUP):
        card = random.randint(100, 500)
        choices = [f"med{i}_{j}" for j in range(card)]
        data[f"med_card_str_{i}"] = [random.choice(choices) for _ in range(_NUM_ROWS)]

    # 30 high-cardinality string columns (mostly unique)
    for i in range(_STR_COLS_PER_GROUP):
        data[f"high_card_str_{i}"] = [f"unique{i}_{j}" for j in range(_NUM_ROWS)]

    # 30 integer columns with varied ranges
    for i in range(_NUM_COLS_PER_GROUP):
        upper = random.choice([10, 100, 1000, 100_000])
        data[f"int_col_{i}"] = [random.randint(0, upper) for _ in range(_NUM_ROWS)]

    # 30 float columns
    for i in range(_NUM_COLS_PER_GROUP):
        data[f"float_col_{i}"] = [random.random() * 1000 for _ in range(_NUM_ROWS)]

    # --- Columns designed for specific search distributions ---

    # Single match: exactly one row has the sentinel value
    data["sentinel_col"] = ["NEEDLE" if i == _NUM_ROWS // 2 else "haystack" for i in range(_NUM_ROWS)]

    # Sparse matches: every ~1000th row matches (spread far apart)
    data["sparse_col"] = [1 if i % 1000 == 0 else 0 for i in range(_NUM_ROWS)]

    # Adjacent matches: a contiguous block of 50k rows in the middle
    block_start = _NUM_ROWS // 4
    block_end = block_start + _NUM_ROWS // 4
    data["adjacent_col"] = [1 if block_start <= i < block_end else 0 for i in range(_NUM_ROWS)]

    return pl.DataFrame(data)


# Build the dataframe once at module level so parametrized tests share it.
_LARGE_DF = _make_large_df()


async def _scroll_around(pilot, n_ops: int) -> float:
    """Perform n_ops scroll operations using only up/down arrow keys."""
    # Scroll down for 3/4 of the ops, then back up for the rest
    down_count = (n_ops * 3) // 4
    up_count = n_ops - down_count

    # Warm up the rendering pipeline
    for _ in range(3):
        await pilot.press("down")
        await pilot.pause()
    for _ in range(3):
        await pilot.press("up")
        await pilot.pause()

    start = time.perf_counter()
    for _ in range(down_count):
        await pilot.press("down")
        await pilot.pause()
    for _ in range(up_count):
        await pilot.press("up")
        await pilot.pause()
    elapsed = time.perf_counter() - start
    return elapsed


async def _apply_search(pilot, query: str):
    """Open search box, type query, submit, and wait for results."""
    await pilot.press("/")
    await pilot.pause()
    await pilot.press(*list(query))
    await pilot.press("enter")
    await pilot.pause()
    for _ in range(5):
        await pilot.pause()


@pytest.mark.parametrize("screen_size", _SCREEN_SIZES)
@pytest.mark.parametrize("search_query, min_hits, max_hits", _SEARCH_PARAMS)
async def test_search_scroll_performance(search_query, min_hits, max_hits, screen_size):
    """Scrolling with search highlights must be within 15% of baseline scroll speed."""
    app = DtBrowserApp("perf_test", _LARGE_DF)

    async with app.run_test(size=screen_size) as pilot:
        await pilot.pause()
        browser = app.query_one(DtBrowser)

        # --- Baseline: scroll without search ---
        baseline_time = await _scroll_around(pilot, _NUM_SCROLL_OPS)

        # Reset cursor to top
        await pilot.press("g")
        await pilot.pause()

        # --- Apply search ---
        await _apply_search(pilot, search_query)

        assert browser.active_search_queue is not None, "Search should have produced results"
        hit_count = len(browser.active_search_queue)
        assert min_hits <= hit_count <= max_hits, (
            f"Expected {min_hits}-{max_hits} hits, got {hit_count}"
        )

        # Reset cursor to top for fair comparison
        await pilot.press("g")
        await pilot.pause()

        # --- Measure: scroll with search active ---
        search_time = await _scroll_around(pilot, _NUM_SCROLL_OPS)

        slowdown = (search_time - baseline_time) / baseline_time
        w, h = screen_size
        label = f"{w}x{h}"
        print(f"\n[{label} | {search_query}]")
        print(f"  Baseline scroll time: {baseline_time:.3f}s")
        print(f"  Search scroll time:  {search_time:.3f}s")
        print(f"  Slowdown: {slowdown:.1%}")
        print(f"  Search hits: {hit_count:,}")

        assert slowdown <= _SLOWDOWN_THRESHOLD, (
            f"[{label}] Search scrolling is {slowdown:.1%} slower than baseline "
            f"(threshold: {_SLOWDOWN_THRESHOLD:.0%}). "
            f"Baseline: {baseline_time:.3f}s, Search: {search_time:.3f}s, "
            f"Hits: {hit_count:,}"
        )
