SHELL=/bin/bash

.VENV:=.venv/bin/activate
ACTIVATE=source $(.VENV)
SRCS=$(shell find src/ -name '*.py')


$(.VENV):
	@python3 -m venv .venv
	@$(ACTIVATE) && pip install uv

.make.package_installed: pyproject.toml | $(.VENV)
	@$(ACTIVATE) && uv pip install -e .[dev]

.make.formatted: $(SRCS) | $(.VENV)
	@$(ACTIVATE) && ruff check --select I --fix src/
	@$(ACTIVATE) && ruff format src/

.make.linted: $(SRCS) .make.formatted | $(.VENV)
	@$(ACTIVATE) && ruff check src/
	@$(ACTIVATE) && pylint src/

.make.typed: $(SRCS) .make.formatted | $(.VENV)
	@$(ACTIVATE) && mypy src/

activate: .make.package_installed
	@bash --init-file <(echo "$(ACTIVATE)")

test: .make.package_installed | $(.VENV)
	@$(ACTIVATE) && pytest tests/ -m "not snapshot"

test-snapshot: .make.package_installed | $(.VENV)
	@$(ACTIVATE) && pytest tests/ -m snapshot

check: .make.formatted .make.linted .make.typed

clean:
	@rm -rf .make.*
	@rm -rf .venv
