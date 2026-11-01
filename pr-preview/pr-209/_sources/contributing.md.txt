# Contributing

We welcome contributions to Simulusion! This guide covers how to get started.

## Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/ParamThakkar123/simulusion.git
   cd simulusion
   ```

2. **Install in development mode**
   ```bash
   uv sync --dev
   ```

   If you are not using `uv`, use the equivalent editable install:

   ```bash
   python -m pip install -e ".[dev]"
   ```

3. **Install pre-commit hooks**
   ```bash
   uv run pre-commit install
   ```

## Code Style

We use:
- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking
- **Pre-commit** for automated checks

Run checks:
```bash
# Format code
uv run black .

# Lint
uv run ruff check .

# Type check
uv run mypy .

# All checks
uv run pre-commit run --all-files
```

## Testing

Run the test suite:
```bash
# All tests
pytest

# Fast subset (skip slow, GPU, and integration tests)
pytest -m "not slow and not gpu and not integration"

# Specific package-mirrored test area
pytest tests/masks/test_masks.py

# With coverage
pytest --cov=simulusion --cov-report=html
```

## Documentation

Build docs locally:
```bash
cd docs
sphinx-build -b html source build/html
```

Open `docs/build/html/index.html` in your browser.

## Adding New Features

### 1. New Models

1. Create model class in `simulusion/models/`
2. Add config class in `simulusion/configs/`
3. Export both from `simulusion/__init__.py`
4. Update training scripts
5. Add documentation and tests

### 2. New Environments

1. Implement environment wrapper in `simulusion/envs/`
2. Add to environment registry
3. Update documentation

## Pull Request Process

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-feature
   ```
3. **Make your changes**
4. **Run tests and checks**
   ```bash
   uv run pre-commit run --all-files
   uv run pytest
   ```
5. **Update documentation** if needed
6. **Commit your changes**
   ```bash
   git commit -m "Add feature: my feature"
   ```
7. **Push to your fork**
   ```bash
   git push origin feature/my-feature
   ```
8. **Create a Pull Request**

## Commit Messages

Use conventional commit format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation
- `test:` Testing
- `refactor:` Code refactoring
- `chore:` Maintenance

Examples:
- `feat: add I-JEPA linear evaluation`
- `fix: resolve memory leak in dreamer training`
- `docs: update inference guide`

## Issue Reporting

- **Bug reports**: Use the bug report template
- **Feature requests**: Use the feature request template
- **Questions**: Use discussions

## Code of Conduct

Please follow our code of conduct in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.