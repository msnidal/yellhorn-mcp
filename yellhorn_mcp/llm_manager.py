"""Compatibility layer for legacy import path.

This module re-exports the refactored LLM package interfaces so existing
code and tests importing `yellhorn_mcp.llm_manager` continue to work.
"""

import warnings

from yellhorn_mcp.llm.manager import (
    ChunkingStrategy,
    LLMManager,
    api_retry,
    is_retryable_error,
    log_retry_attempt,
)
from yellhorn_mcp.models.metadata_models import UsageMetadata

warnings.warn(
    "yellhorn_mcp.llm_manager is deprecated; use yellhorn_mcp.llm.* modules.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "LLMManager",
    "ChunkingStrategy",
    "api_retry",
    "is_retryable_error",
    "log_retry_attempt",
    "UsageMetadata",
]
