"""Configuration management for KG filter parameters."""

from typing import Optional

from pydantic import BaseModel, Field


class FilterConfig(BaseModel):
    """Filter configuration for co-occurrence relationships.

    These parameters control which co-occurrence relationships are retained
    during relation filtering.
    """

    docs_count_min: int = Field(2, ge=1, description="Minimum document count")
    sent_count_min: int = Field(2, ge=1, description="Minimum sentence count")
    stopwords_enabled: bool = Field(
        True, description="Enable biomedical stopword filtering"
    )
    allowed_entity_type_pairs: Optional[list[tuple[str, str]]] = Field(
        None, description="Whitelist of entity type pairs (e.g., [('Gene', 'Disease')])"
    )
    stopwords: Optional[set[str]] = Field(
        None, description="Custom stopword set (uses BIOMEDICAL_STOPWORDS if None)"
    )
