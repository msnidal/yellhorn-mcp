"""Usage models for LLM package.

Currently re-exports UsageMetadata from models to avoid broad import churn.
If desired, the class can be relocated here in a follow-up without breaking
call sites that import from yellhorn_mcp.llm.usage.
"""

from yellhorn_mcp.models.metadata_models import UsageMetadata

__all__ = ["UsageMetadata"]
