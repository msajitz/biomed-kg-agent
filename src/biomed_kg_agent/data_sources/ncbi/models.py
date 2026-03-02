"""
Data models for NCBI data sources (PubMed, PMC).

This module contains models specific to NCBI data sources including
PubMed abstracts and PMC full-text articles.
"""

from typing import Optional

from sqlmodel import Field, SQLModel


class PubmedDocument(SQLModel, table=True):  # type: ignore[call-arg]
    """
    Model for PubMed document data from NCBI E-utilities API.

    This model represents the raw data structure from PubMed before
    conversion to the canonical DocumentInternal format for pipeline processing.

    Used for:
    - Ingestion from PubMed API
    - Storage in SQLite database
    - Conversion to DocumentInternal via pubmed_to_internal()
    """

    pmid: str = Field(primary_key=True, description="PubMed ID")
    title: str
    abstract: str
    document_type: Optional[str] = Field(
        default="Article",
        description=(
            "Type of the document, e.g., Article, BookChapter. " "'Unknown' if neither."
        ),
    )
    journal: Optional[str] = None
    year: Optional[int] = None
    authors: Optional[str] = None  # Comma-separated list of authors

    # Enhanced metadata fields for biomedical research
    mesh_terms: Optional[str] = Field(
        default=None,
        description="Comma-separated list of MeSH (Medical Subject Headings) terms",
    )
    doi: Optional[str] = Field(
        default=None,
        description="Digital Object Identifier for permanent paper linking",
    )
    keywords: Optional[str] = Field(
        default=None, description="Comma-separated list of author-provided keywords"
    )

    def __repr__(self) -> str:
        return f"<PubmedDocument pmid={self.pmid} title={self.title[:30]}...>"
