"""Base interfaces and data contracts for LLM clients.

Defines structural Protocols so provider implementations (OpenAI, Gemini, etc.)
can be swapped behind a consistent API.
"""

from typing import Dict, Optional, Protocol, TypedDict, Union, runtime_checkable

from yellhorn_mcp.models.metadata_models import UsageMetadata


@runtime_checkable
class LoggerContext(Protocol):
    async def log(self, *args, **kwargs) -> None: ...


class GenerateResult(TypedDict, total=False):
    # Provider-agnostic content; string for text or JSON-like dict
    content: Union[str, Dict[str, object]]
    # Unified usage tracking
    usage_metadata: UsageMetadata
    # Provider-specific extras (e.g., Gemini grounding metadata)
    extras: Dict[str, object]


class CitationResult(TypedDict, total=False):
    content: Union[str, Dict[str, object]]
    usage_metadata: UsageMetadata
    # Optional provider-specific citation/grounding metadata
    grounding_metadata: object


class LLMClient(Protocol):
    async def generate(
        self,
        *,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        response_format: Optional[str] = None,
        ctx: Optional[LoggerContext] = None,
        **kwargs,
    ) -> GenerateResult:
        """Generate a completion for the given prompt.

        Returns a dict with provider-agnostic `content`, unified `usage_metadata`,
        and optional provider-specific `extras`.
        """
        ...

