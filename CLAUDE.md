# CLAUDE.md - Guidelines for AI Assistants

## Testing

- **Unit Tests**: All code must be covered by unit tests. Use `pytest` for writing and running tests.
- **Test Coverage**: Maintain minimum 70% test coverage for all new code
- **Integration Tests**: Include integration tests for LLM Manager and API interactions
- **Mock Testing**: Use proper mocking for external API calls (OpenAI, Gemini, GitHub)

## Code Style Guidelines

- **Python Version**: 3.10+ (use modern typing with `|` operator)
- **Formatting**: black with default settings
- **Linting**: Use black for code formatting and isort for import ordering (no flake8)
- **Imports**: Use isort to organize imports automatically with black-compatible settings
- **Types**: Use modern type hints for all functions and class attributes, ie. prefer `list[str]` over `List[str]` and `sometype | None` over `Optional[sometype]`.
- **Documentation**: Standard triple-quote docstrings with parameter descriptions for all public methods and classes. Use Google-style docstrings for clarity.

## Architecture Guidelines

- **LLM Manager**: All LLM interactions should go through the unified `LLMManager` class
- **Error Handling**: Use proper exception handling with retry logic for external API calls
- **Token Management**: Always check token limits before making API calls
- **Cost Tracking**: Include usage metadata and cost tracking for all LLM calls
- **Chunking**: Implement intelligent chunking for large prompts that exceed context limits

## Formatting Commands

Before committing code, always format with:

```bash
# Format code with black
python -m black yellhorn_mcp tests

# Sort imports with isort
python -m isort yellhorn_mcp tests
```

Remember to run these commands automatically when making changes to ensure consistent code style.

## Retry Logic

Use the built-in retry decorator for external API calls:

```python
from yellhorn_mcp.llm.retry import api_retry

@api_retry
async def my_api_call():
    # Your API call here
    pass
```

## Preparing for Pull Requests

Before submitting a PR to the yellhorn-mcp repository, ensure your code meets all quality standards:

### Pre-submission Checklist

1. **Code Formatting** (Required)

   ```bash
   # Format code with black
   python -m black yellhorn_mcp tests
   
   # Sort imports with isort
   python -m isort yellhorn_mcp tests
   ```

2. **Run Tests Locally**

   ```bash
   # Run tests with coverage report
   pytest --cov=yellhorn_mcp --cov-report term-missing
   
   # Ensure coverage meets minimum threshold (≥70%)
   ```

3. **Verify All Tests Pass**
   - Unit tests must pass for Python 3.10 and 3.11
   - Coverage must be ≥70% (CI will fail if below threshold)
   - All new code must have appropriate test coverage

### CI/CD Pipeline

When you submit a PR, GitHub Actions will automatically verify:

1. **Code formatting** with black (must pass)
2. **Import sorting** with isort (must pass)
3. **Unit tests** on Python 3.10 and 3.11
4. **Coverage threshold** - Must maintain ≥70% line coverage

### Submission Process

1. Fork the repository from `https://github.com/msnidal/yellhorn-mcp`
2. Create a feature branch for your changes
3. Implement your changes following all guidelines in this document
4. Run the pre-submission checklist commands
5. Push to your fork
6. Create a PR to the main repository (`msnidal/yellhorn-mcp`)
7. Ensure all CI checks pass

### Repository Information

- **Main Repository**: <https://github.com/msnidal/yellhorn-mcp>
- **Repository Owner**: msnidal
- **Required Coverage**: ≥70% line coverage
- **Python Versions**: 3.10, 3.11

All PRs will be reviewed by the repository maintainers once CI checks pass.
