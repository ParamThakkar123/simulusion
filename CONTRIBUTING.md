# Contributing to Simulusion

We welcome contributions to Simulusion! This document outlines the guidelines for contributing to the project.

## Ways to Contribute

- **Report Bugs**: Open an issue on GitHub with a clear description of the bug, steps to reproduce, and expected behavior.
- **Suggest Features**: Propose new features or improvements via GitHub issues.
- **Submit Pull Requests**: Implement fixes or enhancements and submit a PR.

## Development Setup

1. Fork the repository and clone your fork:
   ```bash
   git clone https://github.com/your-username/simulusion.git
   cd simulusion
   ```

2. Install dependencies with development extras:
   ```bash
   uv sync --dev
   ```

3. Set up pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

If you are not using `uv`, install the project in editable mode with development extras instead:

```bash
python -m pip install -e ".[dev]"
pre-commit install
```

## Coding Standards

- Follow PEP 8 style guidelines.
- Use type hints where possible.
- Format code with Ruff: `uv run ruff format .`
- Lint code with Ruff: `uv run ruff check .`
- Type-check code with MyPy where practical: `uv run mypy .`
- Run all configured hooks with: `uv run pre-commit run --all-files`

## Testing

- Run tests with: `uv run pytest`
- Run a fast subset with: `uv run pytest -m "not slow and not gpu and not integration"`
- The repository pins Python bytecode caches outside the source tree with `PYTHONPYCACHEPREFIX=build/__pycache__`; keep this set when running ad-hoc Python commands so `__pycache__` directories are not created beside source files.
- Ensure all tests pass before submitting a PR.
- Add tests for new features or bug fixes.

### Running the full suite with every backend installed

With `simulusion[all]` installed, a single `pytest` process ends up holding torch,
jax, brax, and the Unity SDK resident at the same time. On a machine with
limited RAM the run dies partway through with allocation errors — or, worse,
with no summary line at all, because the process was killed rather than a test
failing.

If you hit that, run the suite in file-sized chunks so memory is reclaimed
between files instead of accumulating for the whole session:

```bash
# POSIX: a fresh process per test, memory reclaimed each time
uv run pytest tests/ --forked

# Any platform: one process per test file
for f in tests/**/test_*.py; do uv run pytest "$f" -q || break; done
```

`-n`/`--dist loadfile` alone does *not* help here — xdist workers persist
across files, and each additional worker is another full copy of torch.

## Pull Request Process

1. Create a feature branch from `main`: `git checkout -b feature/your-feature`
2. Make your changes and commit them.
3. Run checks and tests: `uv run pre-commit run --all-files` and `uv run pytest`
4. Push to your fork and create a PR.
5. PRs must pass CI checks and have at least one approval.

## Commit Message Conventions

Use structured commit messages to keep the history readable:

- `feat: ...` — new feature
- `fix: ...` — bug fix
- `docs: ...` — documentation only
- `refactor: ...` — code restructuring
- `test: ...` — adding or updating tests
- `ci: ...` — CI/CD changes
- `chore: ...` — maintenance tasks (deps, config, etc.)

Write the body in imperative mood ("fix bug" not "fixed bug" or "fixes bug"). If the commit closes an issue, include `Closes #N` in the body.

## Code of Conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md). Please be respectful and inclusive; harassment or discriminatory behavior will not be tolerated.

For questions, reach out via GitHub issues.
