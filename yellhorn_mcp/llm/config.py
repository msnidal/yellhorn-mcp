"""Configuration model for LLMManager using Pydantic."""

from pydantic import BaseModel, Field


class LLMManagerConfig(BaseModel):
    safety_margin_tokens: int | None = Field(
        default=None, description="Legacy: fixed token safety margin (prefer ratio)."
    )
    safety_margin_ratio: float = Field(
        default=0.2, description="Fraction of model limit reserved for responses/system."
    )
    overlap_ratio: float = Field(default=0.1, ge=0.0, le=0.5, description="Chunk overlap ratio.")
    aggregation_strategy: str = Field(
        default="concatenate", description="Aggregation strategy for multi-chunk responses."
    )
    chunk_strategy: str = Field(
        default="sentences", description="Chunking algorithm: 'sentences' or 'paragraphs'."
    )

