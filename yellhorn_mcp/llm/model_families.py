"""Utilities for classifying supported LLM model families.

Centralizes model prefix handling so provider detection stays consistent
across CLI, server, and manager integrations.
"""

from typing import Final, Literal

ModelFamily = Literal["gemini", "openai", "xai"]

_GEMINI_PREFIXES: Final[tuple[str, ...]] = ("gemini-", "mock-")
_OPENAI_PREFIXES: Final[tuple[str, ...]] = ("gpt-", "o")
_XAI_PREFIXES: Final[tuple[str, ...]] = ("grok-",)


def _normalize(model: str) -> str:
    return model.strip().lower()


def is_gemini_model(model: str) -> bool:
    normalized = _normalize(model)
    return any(normalized.startswith(prefix) for prefix in _GEMINI_PREFIXES)


def is_openai_model(model: str) -> bool:
    normalized = _normalize(model)
    return any(normalized.startswith(prefix) for prefix in _OPENAI_PREFIXES)


def is_xai_model(model: str) -> bool:
    normalized = _normalize(model)
    return any(normalized.startswith(prefix) for prefix in _XAI_PREFIXES)


def detect_model_family(model: str) -> ModelFamily:
    if is_gemini_model(model):
        return "gemini"
    if is_openai_model(model):
        return "openai"
    if is_xai_model(model):
        return "xai"
    raise ValueError(f"Unsupported model '{model}'")


def supports_reasoning_effort(model: str) -> bool:
    normalized = _normalize(model)
    if normalized == "gpt-5-nano":
        return False
    return normalized.startswith("gpt-5")
