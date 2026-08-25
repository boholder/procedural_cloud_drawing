.PHONY: c

c:
	uvx ruff format
	uvx ruff check --fix --unsafe-fixes --show-fixes
	uv run pytest tests
