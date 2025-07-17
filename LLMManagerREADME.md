# LLM Manager Implementation Summary

This document summarizes the exact changes made to the yellhorn-mcp codebase to implement the LLM Manager functionality, which adds intelligent token counting, automatic chunking, usage metrics tracking, and retry mechanisms for both OpenAI and Gemini models.

## Overview

The LLM Manager provides a unified interface for making LLM calls with:
- Automatic token counting and context window management
- Intelligent chunking for prompts that exceed model limits
- Comprehensive usage metrics tracking
- Exponential backoff retry for rate limits
- Support for both OpenAI and Gemini models including GPT-4.1 and deep research models

## Files Created

### 1. `yellhorn_mcp/llm_manager.py`
A comprehensive LLM management system with the following key components:

**Core Classes:**
- `UsageMetadata`: Unified usage tracking that handles both OpenAI and Gemini formats
- `ChunkingStrategy`: Intelligent text splitting with paragraph and sentence-based strategies
- `LLMManager`: Main manager class for all LLM operations

**Key Features:**
- Automatic chunking when prompts exceed model context windows
- Configurable safety margins and overlap ratios for chunks
- Deep research model support (o3, o4) with web search and code interpreter tools
- Citation support for Gemini models with search grounding
- Retry decorator with exponential backoff for rate limit handling

### 2. `yellhorn_mcp/token_counter.py`
Token counting utility using tiktoken for accurate token estimation:

**Features:**
- Model-specific token limits (including GPT-4.1 with 1M tokens)
- Proper encoding mappings (o200k_base for GPT-4o family, cl100k_base for others)
- Token estimation for responses
- Context window fitting checks
- Encoding caching for performance

### 3. `examples/mock_context.py`
Mock context implementation for testing yellhorn MCP tools:

**Components:**
- `mock_github_command`: Simulates GitHub CLI responses
- `BackgroundTaskManager`: Tracks and waits for async tasks
- `MockContext`, `MockLifespanContext`, `MockRequestContext`: Test infrastructure
- Helper functions for running MCP tools in isolation

### 4. `tests/test_llm_manager.py`
Comprehensive test suite with 100+ tests covering:
- Initialization with various configurations
- Model detection (OpenAI vs Gemini)
- Simple and chunked LLM calls
- JSON response handling
- Deep research model tool activation
- Citation and grounding metadata
- Retry logic and error handling
- Usage metadata tracking

### 5. `tests/test_token_counter.py`
Test suite for token counting functionality:
- Model limit retrieval
- Token counting across different models
- Response token estimation
- Context window fitting checks
- Special character handling
- Configuration overrides

### 6. `yellhorn_mcp/metadata_models.py`
Data models for tracking completion and submission metadata:
- `CompletionMetadata`: Tracks generation metrics, timing, and status
- `SubmissionMetadata`: Records workplan submission details
- Used for GitHub comment formatting and status tracking

### 7. `yellhorn_mcp/search_grounding.py`
Search grounding functionality for Gemini models:
- `_get_gemini_search_tools`: Configures Google Search tools for supported models
- `add_citations`: Adds inline citations to generated content
- `add_citations_from_metadata`: Processes grounding metadata into citations
- Enables web search integration for enhanced responses

## Changes to Existing Files

### `yellhorn_mcp/server.py`
**Import Addition:**
```python
from yellhorn_mcp.llm_manager import LLMManager, UsageMetadata
```

**In `app_lifespan` function:**
- Added LLMManager initialization with configuration:
```python
llm_manager = LLMManager(
    openai_client=openai_client,
    gemini_client=gemini_client,
    config={
        "safety_margin_tokens": 2000,
        "overlap_ratio": 0.1,
        "chunk_strategy": "paragraph",
        "aggregation_strategy": "concatenate"
    }
)
```

**In `MODEL_PRICING` dictionary:**
- Added GPT-4.1 pricing:
```python
"gpt-4.1": {
    "input": {"default": 10.0},
    "output": {"default": 30.0},
}
```

**In `format_metrics_section` function:**
- Updated to use `UsageMetadata` for consistent interface:
```python
usage = UsageMetadata(usage_metadata)
```

**In `process_workplan_async` and `process_judgement_async` functions:**
- Replaced direct API calls with LLMManager calls
- Added support for both `call_llm_with_usage` and `call_llm_with_citations`
- Improved error handling and usage metadata extraction

### `pyproject.toml`
**Updated Dependencies:**
```toml
# Core dependencies for LLM Manager functionality
"tiktoken~=0.8.0",           # For accurate token counting across models
"tenacity~=9.1.2",           # For retry with exponential backoff (updated)

# Google API dependencies (updated versions)
"google-genai~=1.16.1",      # Updated from 1.8.0 for latest Gemini features
"google-api-core~=2.25.1",   # Updated from 2.24.2 for better stability
```

These dependency updates ensure:
- Compatibility with the latest Gemini API features
- Improved retry logic with tenacity 9.x
- Better error handling with updated google-api-core
- Consistent API interfaces across all dependencies

## Key Implementation Details

### Token Counting and Chunking
The system automatically detects when a prompt exceeds the model's context window and splits it into manageable chunks. It uses:
- Binary search to find optimal split points
- Natural boundaries (paragraphs, sentences, words) for splits
- Configurable overlap between chunks for context preservation

### Retry Mechanism
Implements intelligent retry logic:
```python
@retry(
    retry=retry_if_exception(is_retryable_error),
    wait=wait_exponential(multiplier=1, min=4, max=60, exp_base=2),
    stop=stop_after_attempt(5),
    before_sleep=log_retry_attempt
)
```

### Usage Metrics Tracking
Provides unified tracking across both OpenAI and Gemini:
- Automatic aggregation for chunked calls
- Cost calculation based on model pricing
- Support for both token count formats

### Deep Research Models
Special handling for o3 and o4 models:
- Automatic enabling of web search and code interpreter tools
- Temperature forced to 1.0 for reasoning models
- Proper tool configuration in API calls

## Testing

The implementation includes comprehensive test coverage:
- **Unit Tests**: Cover all major components and edge cases
- **Integration Tests**: Via mock_context.py for testing in notebook environments
- **Error Handling**: Tests for rate limits, API errors, and edge cases

## Benefits

1. **Reliability**: Automatic retry handling reduces failures from rate limits
2. **Scalability**: Chunking enables processing of arbitrarily large prompts
3. **Cost Tracking**: Built-in usage metrics and cost calculation
4. **Flexibility**: Configurable strategies for different use cases
5. **Compatibility**: Unified interface works with both OpenAI and Gemini models

This implementation significantly enhances the robustness and capabilities of the yellhorn-mcp server while maintaining backward compatibility.

## Integration Details

### How the LLM Manager Integrates with the Server

The LLM Manager is deeply integrated into the yellhorn-mcp server's core workflow, replacing direct API calls with a unified, robust interface. Here's how each component connects:

#### 1. **Initialization Flow in `app_lifespan`**
```python
# The LLM Manager is initialized during server startup
llm_manager = LLMManager(
    openai_client=openai_client,
    gemini_client=gemini_client,
    config={...}
)
```
- Created once per server instance in the lifespan context
- Shared across all MCP tool invocations
- Configuration is centralized for consistent behavior

#### 2. **Usage in Async Processing Functions**

**`process_workplan_async`** - The core workplan generation function:
- Receives `llm_manager` as a parameter from the lifespan context
- Builds prompts that can exceed model limits (includes entire codebase)
- Uses `call_llm_with_usage` for OpenAI models (no citation support)
- Uses `call_llm_with_citations` for Gemini models (with search grounding)
- Automatically handles chunking when codebase + prompt exceeds context window
- Extracts and formats usage metadata for cost tracking

**`process_judgement_async`** - The diff judgement function:
- Similar integration pattern as workplan generation
- Can handle very large diffs that exceed model limits
- Maintains consistent error handling and retry logic

**`curate_context`** - The context curation tool:
- Uses LLM Manager for analyzing directory structures
- Benefits from automatic chunking for large codebases
- Consistent usage tracking across all operations

#### 3. **Token Counting Integration**

The `TokenCounter` is used internally by `LLMManager` but also influences how the server operates:
- **Context Window Awareness**: The server can now make intelligent decisions about what to include in prompts
- **Safety Margins**: The 2000-token safety margin ensures room for system prompts and responses
- **Model Detection**: Proper encoding selection (o200k_base vs cl100k_base) ensures accurate counting

#### 4. **Usage Metadata Flow**

The `UsageMetadata` class provides a unified interface that's used throughout:
```python
# In format_metrics_section
usage = UsageMetadata(usage_metadata)  # Handles any format
input_tokens = usage.prompt_tokens
output_tokens = usage.completion_tokens
```
- Seamlessly handles OpenAI's format (`prompt_tokens`, `completion_tokens`)
- Seamlessly handles Gemini's format (`prompt_token_count`, `candidates_token_count`)
- Used by `calculate_cost()` for accurate pricing
- Aggregated across chunks for total usage

#### 5. **Error Handling and Retry Logic**

The retry decorator with exponential backoff is crucial for production reliability:
```python
@api_retry  # Applied to _call_openai and _call_gemini
async def _call_openai(...):
    # Automatic retry on rate limits
```
- Detects various rate limit error formats
- Exponential backoff: 4s, 8s, 16s, 32s, 60s (max)
- Logs retry attempts for debugging
- Handles both OpenAI and Google-specific exceptions

#### 6. **Mock Context for Testing**

The `mock_context.py` enables standalone testing and notebook usage:
- **BackgroundTaskManager**: Tracks async tasks created by `asyncio.create_task`
- **Mock GitHub Commands**: Simulates GitHub CLI without actual API calls
- **Direct Function Access**: Allows calling internal functions like `process_workplan_async` directly
- **Integration Testing**: Validates the entire flow including chunking and retries

#### 7. **Citation and Search Grounding Support**

For Gemini models with search grounding enabled:
```python
# In process_workplan_async
if "grounding_metadata" in response_data:
    workplan_content = add_citations_from_metadata(
        workplan_content, 
        response_data["grounding_metadata"]
    )
```
- LLM Manager preserves grounding metadata from Gemini responses
- Server processes citations and adds them to the output
- Maintains full search result attribution

#### 8. **Deep Research Model Support**

The LLM Manager automatically configures tools for deep research models:
```python
if self._is_deep_research_model(model):
    params["tools"] = [
        {"type": "web_search_preview"},
        {"type": "code_interpreter", ...}
    ]
```
- Server doesn't need model-specific logic
- Tools are automatically enabled for o3, o4 models
- Temperature is forced to 1.0 for reasoning models

#### 9. **Chunking Strategy Integration**

The paragraph-based chunking strategy aligns with the server's prompt structure:
- Preserves markdown formatting in workplans
- Maintains code block integrity
- Overlapping chunks preserve context across boundaries
- Aggregation strategy concatenates with clear separators

#### 10. **Metrics and Cost Tracking**

The integration provides comprehensive metrics:
```python
# Automatic cost calculation
estimated_cost = calculate_cost(model, input_tokens, output_tokens)

# Completion metadata tracking
completion_metadata = CompletionMetadata(
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    estimated_cost=estimated_cost,
    ...
)
```
- Every LLM call is tracked and reported
- Costs are calculated based on current model pricing
- Metrics are posted as GitHub comments for transparency

### Data Flow Example

Here's how a typical workplan creation flows through the system:

1. **User invokes `create_workplan` MCP tool**
2. **Server creates placeholder GitHub issue**
3. **Launches `process_workplan_async` as background task**
4. **Builds prompt with codebase context** (can be 100K+ tokens)
5. **LLM Manager detects prompt exceeds model limit**
6. **Automatically chunks into smaller pieces**
7. **Makes multiple API calls with retry on rate limits**
8. **Aggregates responses maintaining context**
9. **Returns combined result with usage metadata**
10. **Server updates GitHub issue with workplan**
11. **Posts completion comment with metrics**

This seamless integration ensures reliability, scalability, and cost transparency while handling the complexities of large-scale LLM operations.