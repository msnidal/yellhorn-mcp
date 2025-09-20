"""Base interfaces and data contracts for LLM clients.

Defines structural Protocols and type guards so provider implementations
can be swapped behind a consistent, type-checked API with minimal runtime
introspection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import (
    Dict,
    Optional,
    Protocol,
    Sequence,
    TypedDict,
    TypeGuard,
    Union,
    runtime_checkable,
)

from google.genai.types import GenerateContentConfig

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


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"


class ReasoningEffort(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True, slots=True)
class LLMRequest:
    """Provider-agnostic request payload shared across LLM clients."""

    prompt: str
    model: str
    temperature: float = 0.7
    system_message: Optional[str] = None
    response_format: Optional[ResponseFormat] = None
    generation_config: Optional[GenerateContentConfig] = None
    reasoning_effort: Optional[ReasoningEffort] = None


class LLMClient(Protocol):
    async def generate(
        self,
        request: LLMRequest,
        *,
        ctx: Optional[LoggerContext] = None,
    ) -> GenerateResult:
        """Generate a completion for the given prompt.

        Returns a dict with provider-agnostic `content`, unified `usage_metadata`,
        and optional provider-specific `extras`.
        """
        ...


class UsageResult(TypedDict):
    content: Union[str, Dict[str, object]]
    usage_metadata: UsageMetadata
    reasoning_effort: Optional[ReasoningEffort]


# ----------------------------
# Response typing + type guards
# ----------------------------


# OpenAI Responses API shapes (structural)
@runtime_checkable
class _OAOutputPart(Protocol):
    text: str


@runtime_checkable
class _OAOutputItem(Protocol):
    content: Sequence[_OAOutputPart]


@runtime_checkable
class OpenAIWithOutputList(Protocol):
    output: Sequence[_OAOutputItem]


@runtime_checkable
class WithOutputText(Protocol):
    output_text: str


@runtime_checkable
class WithText(Protocol):
    text: str


def has_openai_output_list(obj: object) -> TypeGuard[OpenAIWithOutputList]:
    try:
        out = getattr(obj, "output")
        return isinstance(out, Sequence) and len(out) > 0 and hasattr(out[0], "content")
    except Exception:
        return False


def has_output_text(obj: object) -> TypeGuard[WithOutputText]:
    return isinstance(getattr(obj, "output_text", None), str)


def has_text(obj: object) -> TypeGuard[WithText]:
    return isinstance(getattr(obj, "text", None), str)


# Gemini-like shapes (minimal for our needs)
@runtime_checkable
class GeminiWithCandidates(Protocol):
    candidates: Sequence[object]


@runtime_checkable
class HasGroundingMetadata(Protocol):
    grounding_metadata: object


def has_candidates(obj: object) -> TypeGuard[GeminiWithCandidates]:
    return isinstance(getattr(obj, "candidates", None), Sequence)


def has_grounding_metadata(obj: object) -> bool:
    return getattr(obj, "grounding_metadata", None) is not None
