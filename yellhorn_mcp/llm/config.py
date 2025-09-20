"""Configuration model for LLMManager using Pydantic.

Adds typed strategies (Literals) and normalizes synonyms.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


ChunkStrategy = Literal["sentences", "paragraph", "paragraphs"]
AggregationStrategy = Literal["concatenate", "summarize"]


class LLMManagerConfig(BaseModel):
    safety_margin_tokens: int | None = Field(
        default=None, description="Legacy: fixed token safety margin (prefer ratio)."
    )
    safety_margin_ratio: float = Field(
        default=0.2, description="Fraction of model limit reserved for responses/system."
    )
    overlap_ratio: float = Field(default=0.1, ge=0.0, le=0.5, description="Chunk overlap ratio.")
    aggregation_strategy: AggregationStrategy = Field(
        default="concatenate", description="Aggregation strategy for multi-chunk responses."
    )
    chunk_strategy: ChunkStrategy = Field(
        default="sentences", description="Chunking algorithm: 'sentences' or 'paragraphs'."
    )

    @field_validator("chunk_strategy", mode="before")
    @classmethod
    def _normalize_chunk_strategy(cls, v: object) -> ChunkStrategy:
        if isinstance(v, str):
            val = v.strip().lower()
            if val in ("paragraph", "paragraphs"):
                return "paragraphs"
            if val == "sentences":
                return "sentences"
        # fallback to default when invalid
        return "sentences"

    @field_validator("aggregation_strategy", mode="before")
    @classmethod
    def _normalize_agg_strategy(cls, v: object) -> AggregationStrategy:
        if isinstance(v, str):
            val = v.strip().lower()
            if val in ("concatenate", "concat", "join"):
                return "concatenate"
            if val in ("summarize", "summary"):
                return "summarize"
        return "concatenate"

