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

## LLM Manager Usage

When working with LLM calls, always use the unified `LLMManager`:

```python
# Good - using LLM Manager
llm_manager = LLMManager(openai_client=openai_client, gemini_client=gemini_client)
response = await llm_manager.call_llm(prompt, model, temperature=0.7)

# Good - with usage tracking
result = await llm_manager.call_llm_with_usage(prompt, model)
content = result["content"]
usage = result["usage_metadata"]

# Bad - direct client calls
response = await openai_client.chat.completions.create(...)
```

## Retry Logic

Use the built-in retry decorator for external API calls:

```python
from yellhorn_mcp.llm_manager import api_retry

@api_retry
async def my_api_call():
    # Your API call here
    pass
```
