"""Configuration model for LLMManager using Pydantic.

Adds typed strategies (Enums) and normalizes synonyms.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ChunkStrategy(str, Enum):
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"

    @classmethod
    def normalize(cls, value: str) -> "ChunkStrategy":
        normalized = value.strip().lower()
        if normalized in {"paragraph", "paragraphs"}:
            return cls.PARAGRAPHS
        return cls.SENTENCES


class AggregationStrategy(str, Enum):
    CONCATENATE = "concatenate"
    SUMMARIZE = "summarize"

    @classmethod
    def normalize(cls, value: str) -> "AggregationStrategy":
        normalized = value.strip().lower()
        if normalized in {"summarize", "summary"}:
            return cls.SUMMARIZE
        if normalized in {"concatenate", "concat", "join"}:
            return cls.CONCATENATE
        return cls.CONCATENATE


class LLMManagerConfig(BaseModel):
    safety_margin_tokens: int | None = Field(
        default=None, description="Legacy: fixed token safety margin (prefer ratio)."
    )
    safety_margin_ratio: float = Field(
        default=0.2, description="Fraction of model limit reserved for responses/system."
    )
    overlap_ratio: float = Field(default=0.1, ge=0.0, le=0.5, description="Chunk overlap ratio.")
    aggregation_strategy: AggregationStrategy = Field(
        default=AggregationStrategy.CONCATENATE,
        description="Aggregation strategy for multi-chunk responses.",
    )
    chunk_strategy: ChunkStrategy = Field(
        default=ChunkStrategy.SENTENCES,
        description="Chunking algorithm: 'sentences' or 'paragraphs'.",
    )

    @field_validator("chunk_strategy", mode="before")
    @classmethod
    def _normalize_chunk_strategy(cls, value: Any) -> ChunkStrategy:
        if isinstance(value, ChunkStrategy):
            return value
        if isinstance(value, str) and value.strip():
            return ChunkStrategy.normalize(value)
        # fallback to default when invalid
        return ChunkStrategy.SENTENCES

    @field_validator("aggregation_strategy", mode="before")
    @classmethod
    def _normalize_agg_strategy(cls, value: Any) -> AggregationStrategy:
        if isinstance(value, AggregationStrategy):
            return value
        if isinstance(value, str) and value.strip():
            return AggregationStrategy.normalize(value)
        return AggregationStrategy.CONCATENATE
