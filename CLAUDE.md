# CLAUDE.md - Guidelines for AI Assistants

## Build/Test Commands

- Install package: `make install`
- Dev dependencies: `make dev-install`
- Run all tests: `make test`
- Test with coverage: `make test-cov`
- Run single test: `python -m pytest tests/test_app.py::test_function_name -v`
- Lint code: `make lint`
- Format code: `make format`
- Run server: `make run` or `python -m yellhorn_mcp.cli`
- Docker build: `make docker-build`
- Docker run: `make docker-run` (requires environment variables)

## Code Style Guidelines

- **Python Version**: 3.10+ (use modern typing with `|` operator)
- **Formatting**: black with default settings
- **Linting**: flake8 for code quality checks
- **Imports**: Group in order: std lib, third-party, local (alphabetical within groups)
- **Types**: Use type hints for all functions and class attributes, prefer `list[str]` over `List[str]`
- **Naming**: snake_case for functions/variables, PascalCase for classes
- **Documentation**: Triple-quote docstrings with parameter descriptions
- **Error Handling**: Try/except with specific exceptions, use HTTPException for API errors
- **Testing**: Pytest with TestClient for API tests
- **Dependencies**: FastAPI, Pydantic, Gemini API