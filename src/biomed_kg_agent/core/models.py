"""
Core data models for the biomedical document processing pipeline.

This module contains the canonical DocumentInternal model that serves as the
universal document representation across all data sources and processing stages.
"""

from typing import Any, Optional

from pydantic import BaseModel, Field

from biomed_kg_agent.nlp.models import ExtractedEntity


class DocumentInternal(BaseModel):
    """
    Canonical internal document representation for pipeline processing.

    This model serves as the universal document format that all data sources
    (PubMed, PMC, clinical trials, etc.) are converted to before processing.
    This enables source-agnostic NER, relation extraction, and KG construction.

    Design Principles:
    - Required fields: Essential for all processing stages
    - Optional fields: Common metadata that may not be available from all sources
    - Extras dict: Source-specific metadata that doesn't fit standard fields
    - Pipeline results: Populated during processing stages

    Example:
        # From PubMed
        doc = DocumentInternal(
            id="12345678",
            title="Cancer metabolism study",
            text="Abstract text here...",
            source="pubmed",
            pub_year=2024,
            extras={"journal": "Nature", "authors": "Smith J, Doe A"}
        )

        # After NER processing
        doc.entities = [ExtractedEntity(...), ...]
    """

    # Required fields - essential for all processing
    id: str = Field(..., description="Universal document ID (PMID, PMC ID, etc.)")
    title: str = Field(..., description="Document title")
    text: str = Field(
        ..., description="Main document text (abstract + full text for PMC)"
    )
    source: str = Field(
        ..., description="Data source identifier (pubmed, pmc, clinical_trials)"
    )

    # Optional common metadata
    pub_year: Optional[int] = Field(None, description="Publication year")

    # Source-specific metadata storage
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Source-specific metadata (journal, authors, DOI, etc.)",
    )

    # Pipeline processing results - populated during processing stages
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Extracted biomedical entities from NER processing",
    )

    # Future pipeline stages (ready for implementation)
    relations: list[Any] = Field(
        default_factory=list,
        description="Extracted relations between entities (future implementation)",
    )

    kg_nodes: list[Any] = Field(
        default_factory=list,
        description="Knowledge graph nodes created from this document (future)",
    )

    kg_edges: list[Any] = Field(
        default_factory=list,
        description="Knowledge graph edges created from this document (future)",
    )

    @property
    def entity_count(self) -> int:
        """Get total number of extracted entities."""
        return len(self.entities)

    @property
    def entities_by_entity_type(self) -> dict[str, list[ExtractedEntity]]:
        """Group entities by their universal entity type."""
        entity_types: dict[str, list[ExtractedEntity]] = {}
        for entity in self.entities:
            if entity.entity_type not in entity_types:
                entity_types[entity.entity_type] = []
            entity_types[entity.entity_type].append(entity)
        return entity_types

    @property
    def source_metadata(self) -> dict[str, Any]:
        """Get source-specific metadata for this document."""
        return {"source": self.source, "pub_year": self.pub_year, **self.extras}

    def __repr__(self) -> str:
        return f"<DocumentInternal id={self.id} source={self.source} entities={self.entity_count}>"
