"""
Data models for biomedical NLP and entity extraction.

This module contains unified SQLModel-based data models for both business logic
and persistence of extracted biomedical entities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import JSON, Column
from sqlmodel import Field, SQLModel


class ExtractedEntity(SQLModel, table=True):  # type: ignore[call-arg]
    """Standardized entity representation with ontology linking and persistence.

    Unified model serving both business logic and database storage.
    """

    __tablename__ = "extracted_entities"

    # Primary key and foreign key
    id: Optional[int] = Field(default=None, primary_key=True)
    doc_id: str | None = Field(
        default=None,
        index=True,
        description="Document ID for error analysis and provenance",
    )

    # Core entity fields
    text: str = Field(index=True, description="The entity text")
    start_pos: int = Field(ge=0, description="Start character position")
    end_pos: int = Field(gt=0, description="End character position")
    source_model: str | None = Field(
        default=None, description="Model that extracted this entity"
    )
    entity_type: str = Field(
        description="Universal normalized entity type"
    )  # "chemical", "disease", "gene", etc.
    source_label: str = Field(
        description="Original spaCy entity label (e.g., 'CHEMICAL', 'DISEASE')"
    )  # spaCy's ent.label_

    # Sentence context
    sentence_id: int = Field(
        ge=0, description="Sentence-level granularity for relations"
    )
    sentence_text: str = Field(
        description="Full text of the sentence containing this entity"
    )

    # Confidence scores
    ner_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="NER model confidence in entity detection (None for scispaCy models)",
    )
    linking_confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
        description="Entity linking confidence score from UMLS/ontology mapping",
    )

    # External ontology IDs for knowledge graph linking
    umls_cui: Optional[str] = Field(
        default=None, description="UMLS Concept Unique Identifier"
    )
    umls_preferred_name: Optional[str] = Field(
        default=None, description="UMLS preferred term from knowledge base"
    )
    umls_semantic_types: Optional[list[str]] = Field(
        default=None,
        description="UMLS semantic type codes (TUIs)",
        sa_column=Column(JSON),
    )
    chebi_id: Optional[str] = Field(
        default=None, description="ChEBI identifier for chemicals"
    )
    go_id: Optional[str] = Field(default=None, description="Gene Ontology identifier")
    mesh_id: Optional[str] = Field(default=None, description="MeSH identifier")

    # Cross-ontology parent classes
    parent_classes: Optional[list[str]] = Field(
        default=None,
        description="Parent ontology classes (e.g., CHEBI parent terms)",
        sa_column=Column(JSON),
    )


class ProcessedDocument(SQLModel, table=True):  # type: ignore[call-arg]
    """Intermediate storage for NLP-processed documents."""

    __tablename__ = "processed_documents"

    # Core DocumentInternal fields
    id: str = Field(primary_key=True, description="PMID, PMC ID, etc.")
    title: str = Field(description="Document title")
    text: str = Field(description="Full content (abstract + full text)")
    source: str = Field(description="Data source: pubmed, pmc, clinical_trials")
    pub_year: Optional[int] = Field(default=None, description="Publication year")

    # Academic publication metadata (available across most sources)
    journal: Optional[str] = Field(default=None, description="Journal name")
    authors: Optional[str] = Field(
        default=None, description="Comma-separated author list"
    )
    document_type: Optional[str] = Field(default=None, description="Document type")
    doi: Optional[str] = Field(default=None, description="Digital Object Identifier")
    keywords: Optional[str] = Field(
        default=None, description="Comma-separated keywords"
    )
    mesh_terms: Optional[str] = Field(
        default=None, description="MeSH terms (for MEDLINE-indexed publications)"
    )

    # Overflow for future sources (preserve flexibility)
    extras_json: Optional[str] = Field(
        default=None, description="Additional source-specific metadata as JSON"
    )

    # Processing metadata
    processed_at: datetime = Field(description="When NLP processing completed")
    config_json: str = Field(description="NER/linking configuration (JSON)")
    entity_count: int = Field(description="Total entities extracted")
