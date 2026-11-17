# timezone_experiments

Explore alternative global timezone assignments based on solar noon optimization rules.

## Development setup

### Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/)

### Install dependencies & create virtualenv

```bash
uv sync
```

### Install pre-commit hooks

```bash
uv run pre-commit autoupdate
uv run pre-commit install --install-hooks
```

### Run the CLI

```bash
uv run timezone_experiments greet World
uv run timezone_experiments greet World --shout
```

### Run tests

```bash
uv run pytest
```

### Lint & format

```bash
# Check
uv run ruff check .

# Fix
uv run ruff check --fix .

# Format
uv run ruff format .
```

### Type check

```bash
uv run ty check src tests
```

---

## Tools

| Tool | Purpose |
|------|---------|
| [uv](https://docs.astral.sh/uv/) | Package manager & virtualenv |
| [ruff](https://docs.astral.sh/ruff/) | Linting + formatting |
| [ty](https://github.com/astral-sh/ty) | Type checker |
| [pytest](https://pytest.org) | Test framework |
| [pytest-cov](https://pytest-cov.readthedocs.io) | Coverage |
| [pre-commit](https://pre-commit.com) | Git hooks |
| [commitizen](https://commitizen-tools.github.io/commitizen/) | Conventional commits |
