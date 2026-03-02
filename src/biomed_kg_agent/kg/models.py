"""
Knowledge graph models for entity storage and relationships.

This module provides SQLModel-based data models for storing normalized entities,
mentions, and co-occurrences in the knowledge graph construction pipeline.
"""

from typing import Optional

from sqlalchemy import Index, PrimaryKeyConstraint
from sqlmodel import Field, SQLModel


class Entity(SQLModel, table=True):  # type: ignore[call-arg]
    """Normalized entity for knowledge graph construction."""

    id: str = Field(primary_key=True)  # Use resolve_entity_id() for stability
    name: str  # Colloquial search term (first occurrence, normalized)
    entity_type: str  # "Drug", "Gene", "Disease", "Metabolite", etc.
    umls_cui: Optional[str] = None
    chebi_id: Optional[str] = None
    go_id: Optional[str] = None
    mesh_id: Optional[str] = None
    umls_preferred_name: Optional[str] = None  # Canonical UMLS preferred term


class Mention(SQLModel, table=True):  # type: ignore[call-arg]
    """Lightweight entity mention for co-occurrence extraction."""

    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: str
    entity_id: str = Field(index=True)  # FK to Entity.id (indexed for joins)
    text: str  # Original mention text
    sentence_id: int  # For sentence-level co-occurrence
    sentence_text: str  # Full sentence text containing this mention
    start_pos: int
    end_pos: int
    source_label: str  # Original spaCy label: "CHEMICAL", "DISEASE", etc.

    # Composite index to accelerate sentence-scoped lookups.
    # Hot path: relation extraction groups mentions by (doc_id, sentence_id),
    # and queries often filter WHERE doc_id=? AND sentence_id=?.
    # This avoids full scans as Mention grows.
    __table_args__ = (Index("idx_doc_sentence", "doc_id", "sentence_id"),)


class Cooccurrence(SQLModel, table=True):  # type: ignore[call-arg]
    """Pre-computed entity co-occurrence statistics."""

    entity_a_id: str  # Ordered pair (a < b)
    entity_b_id: str
    sent_count: int  # Co-occurrences at sentence level
    docs_count: int  # Number of documents containing both
    doc_ids_sample: Optional[str] = (
        None  # JSON list of doc IDs parallel to evidence_sentences
    )
    evidence_sentences: Optional[str] = (
        None  # JSON list of diverse evidence sentences (default: 5)
    )

    __table_args__ = (
        PrimaryKeyConstraint("entity_a_id", "entity_b_id"),
        Index("idx_docs_count", "docs_count"),  # Fast "top relations" queries
    )
