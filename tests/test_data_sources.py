"""Tests for data sources conversion module.

This module tests ONLY data conversion from source-specific formats
to the canonical DocumentInternal format. It does NOT test NER, linking,
or any pipeline processing.
"""

from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument


def test_document_internal_properties(sample_pubmed_doc: PubmedDocument) -> None:
    """Test DocumentInternal properties and clean API."""
    # Convert to canonical format using the connector
    from biomed_kg_agent.core.connectors import pubmed_to_internal

    result = pubmed_to_internal(sample_pubmed_doc)

    # Should have direct property access to canonical fields
    assert result.id == "123456"
    assert result.title == "Sample PubMed Article"
    assert result.text == "This is a sample abstract."
    assert result.source == "pubmed"
    assert result.pub_year == 2024

    # Should have source metadata in extras
    assert result.extras["journal"] == "Sample Journal"
    assert result.extras["authors"] == "Doe, John; Smith, Jane"

    # Should have value-added properties
    assert hasattr(result, "entity_count")
    assert hasattr(result, "entities_by_entity_type")
    assert hasattr(result, "source_metadata")

    # Should be empty initially (no processing done)
    assert result.entity_count == 0
    assert len(result.entities) == 0
