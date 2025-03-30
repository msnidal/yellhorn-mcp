# CLAUDE.md - Guidelines for AI Assistants

## Testing

- **Unit Tests**: All code must be covered by unit tests. Use `pytest` for writing and running tests.

## Code Style Guidelines

- **Python Version**: 3.10+ (use modern typing with `|` operator)
- **Formatting**: black with default settings
- **Linting**: Use black for code formatting and isort for import ordering (no flake8)
- **Imports**: Use isort to organize imports automatically with black-compatible settings
- **Types**: Use modern type hints for all functions and class attributes, ie. prefer `list[str]` over `List[str]` and `sometype | None` over `Optional[sometype]`.
- **Documentation**: Standard triple-quote docstrings with parameter descriptions for all public methods and classes. Use Google-style docstrings for clarity.
