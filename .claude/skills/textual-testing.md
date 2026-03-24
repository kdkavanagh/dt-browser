---
name: textual-testing
description: Guide for testing and debugging Textual TUI applications
---

# Testing & Debugging Textual TUI Apps

## Testing with pytest

### Setup
- Use **pytest** with **pytest-asyncio**. Set `asyncio_mode = auto` in `pyproject.toml`.
- All tests using `run_test()` must be `async def`.

### Running Apps in Tests: `App.run_test()`

Async context manager that runs the app headless (no terminal). Yields a `Pilot` object:

```python
async def test_example():
    app = MyApp()
    async with app.run_test() as pilot:
        await pilot.press("r")
        assert app.screen.styles.background == Color.parse("red")
```

Default terminal size is (80, 24). Override: `app.run_test(size=(100, 50))`.

### Pilot API — Simulating User Input

All methods are async.

**Key presses** — `pilot.press(*keys)`:
```python
await pilot.press("h", "e", "l", "l", "o")   # type text
await pilot.press("enter")                     # special keys
await pilot.press("ctrl+c")                    # modifier combos
```

**Mouse clicks** — `pilot.click(selector, offset, shift, meta, control, times)`:
```python
await pilot.click("#red")                      # CSS selector by ID
await pilot.click(Button)                      # by widget class
await pilot.click(Button, offset=(0, -1))      # with offset relative to widget
await pilot.click(Button, times=2)             # double-click
```

**Other mouse**: `pilot.double_click()`, `pilot.triple_click()`, `pilot.hover()`, `pilot.mouse_down()`, `pilot.mouse_up()` — same signature as click.

**Resize**: `await pilot.resize_terminal(width=120, height=40)`

**Exit**: `await pilot.exit(result=some_value)`

**Wait for animations**:
```python
await pilot.wait_for_animation()
await pilot.wait_for_scheduled_animations()
```

### Managing Async Timing

Messages are processed asynchronously. Use `pilot.pause()` to wait for pending messages before asserting:

```python
await pilot.pause()             # wait for all pending messages
await pilot.pause(delay=0.5)    # delay then wait
```

This is critical — without it, assertions may run before handlers complete.

### Asserting on State

```python
assert app.screen.styles.background == Color.parse("red")
assert app.query_one("#my-input", Input).value == "hello"
assert app.focused is app.query_one("#my-button")
```

Use `app.query()` or `app.query_one()` with CSS selectors to find widgets.

### Snapshot / Visual Regression Testing

Install `pytest-textual-snapshot`. Use the `snap_compare` fixture (sync, not async):

```python
def test_calculator(snap_compare):
    assert snap_compare("path/to/calculator.py")
```

First run always fails (no baseline). Run `pytest --snapshot-update` after visual verification.

Options: `press=["1", "2"]`, `terminal_size=(50, 100)`, `run_before=async_func`.

### Testing Best Practices

- Click targeting is realistic: overlaying widgets receive the click.
- Use `pilot.pause()` liberally to avoid race conditions.
- Snapshot tests are sync functions; `run_test()` tests are async.
- Check Textual's own `tests/` directory on GitHub for advanced patterns.

## Debugging

### Dev Console (two terminals)

Terminal 1: `textual console`
Terminal 2: `textual run --dev my_app.py`

The console shows `print()` output and internal logs. Verbosity: `-v` for verbose, `-x GROUP` to exclude groups (EVENT, DEBUG, INFO, WARNING, ERROR, PRINT, SYSTEM, LOGGING, WORKER).

### Logging

```python
from textual import log

log("Hello")                    # simple string
log(locals())                   # variables
log(children=self.children)     # keyword args
```

Or use `self.log()` on App/Widget instances. For stdlib logging:

```python
import logging
from textual.logging import TextualHandler
logging.basicConfig(level="NOTSET", handlers=[TextualHandler()])
```

### CSS Hot-Reload

`textual run --dev` enables live CSS reloading — edit `.tcss` files and see changes instantly.

### Browser-based

`textual serve my_app.py` converts the TUI into a web app viewable in a browser.
