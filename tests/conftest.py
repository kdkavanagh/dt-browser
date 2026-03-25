import pathlib

import pytest


@pytest.fixture(autouse=True)
def _isolated_history(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """Give every test its own empty filter-history file.

    Prevents filter/search history written by one test from leaking
    into another when test execution order changes.
    """
    monkeypatch.setenv("DT_BROWSER_HISTORY_FILE", str(tmp_path / "filters.txt"))
