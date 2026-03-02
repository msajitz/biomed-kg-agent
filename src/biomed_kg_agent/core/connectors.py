"""
Data source connector adapters for pipeline processing.

This module provides functions to convert source-specific document models
(PubmedDocument, PMC documents, etc.) to the canonical DocumentInternal format
for pipeline processing.

Design Principles:
- Each data source has a dedicated to_internal() function
- Source-specific metadata goes into the extras dict
- Required fields are mapped consistently
- Handle missing/optional fields gracefully
"""

import logging

from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument

logger = logging.getLogger(__name__)


def pubmed_to_internal(doc: PubmedDocument) -> DocumentInternal:
    """
    Convert a PubmedDocument to DocumentInternal format.

    Maps PubMed-specific fields to the canonical format, storing
    PubMed-specific metadata in the extras dict.

    Args:
        doc: PubmedDocument object from PubMed API

    Returns:
        DocumentInternal object ready for pipeline processing

    Example:
        pubmed_doc = PubmedDocument(pmid="12345", title="...", abstract="...")
        internal_doc = pubmed_to_internal(pubmed_doc)
        result = run_pipeline(internal_doc)
    """
    # Build extras dict with PubMed-specific metadata
    extras = {}

    # Add available metadata fields
    if doc.journal:
        extras["journal"] = doc.journal
    if doc.authors:
        extras["authors"] = doc.authors
    if doc.mesh_terms:
        extras["mesh_terms"] = doc.mesh_terms
    if doc.doi:
        extras["doi"] = doc.doi
    if doc.keywords:
        extras["keywords"] = doc.keywords
    if doc.document_type:
        extras["document_type"] = doc.document_type

    return DocumentInternal(
        id=doc.pmid,
        title=doc.title or "",  # Handle potential None values
        text=doc.abstract or "",  # Handle potential None values
        source="pubmed",
        pub_year=doc.year,
        extras=extras,
    )


def pmc_to_internal(doc: PubmedDocument) -> DocumentInternal:
    """
    Convert a PMC document (stored as PubmedDocument) to DocumentInternal format.

    PMC documents reuse the PubmedDocument model but contain full text
    in the abstract field. This function handles the PMC-specific mapping.

    Args:
        doc: PubmedDocument object from PMC API (contains full text)

    Returns:
        DocumentInternal object ready for pipeline processing

    Note:
        PMC documents are currently stored as PubmedDocument objects,
        but the abstract field contains the full text content.
    """
    # Build extras dict with PMC-specific metadata
    extras = {}

    # Add available metadata fields
    if doc.journal:
        extras["journal"] = doc.journal
    if doc.authors:
        extras["authors"] = doc.authors
    if doc.document_type:
        extras["document_type"] = doc.document_type

    # PMC documents may have PMC ID in extras
    if hasattr(doc, "pmc_id") and doc.pmc_id:
        extras["pmc_id"] = doc.pmc_id

    return DocumentInternal(
        id=doc.pmid,  # PMC documents use PMID as primary identifier
        title=doc.title or "",
        text=doc.abstract or "",  # For PMC, this contains full text
        source="pmc",
        pub_year=doc.year,
        extras=extras,
    )
