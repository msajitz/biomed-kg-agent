"""Tests for two-pass processing module.

This module tests the two-pass processing orchestration functionality.
Uses mocks to avoid loading heavy models for fast testing.
"""

from unittest.mock import MagicMock

import pytest

from biomed_kg_agent.nlp.models import ExtractedEntity
from biomed_kg_agent.nlp.two_pass_processor import process_documents_two_pass


def test_process_documents_two_pass_basic(
    mock_two_pass_deps: tuple[MagicMock, MagicMock],
    temp_db_path: str,
) -> None:
    """Test basic two-pass processing functionality."""
    mock_ner_class, mock_linker_class = mock_two_pass_deps

    texts = [
        "Glucose metabolism in cancer cells",
        "Aspirin reduces inflammation",
        "p53 mutations cause tumors",
    ]
    doc_ids = ["doc1", "doc2", "doc3"]

    # Setup mock NER instance with side_effect to return correct doc_id per call
    mock_ner = MagicMock()

    def mock_extract_entities(
        text: str, doc_id: str
    ) -> dict[str, list[ExtractedEntity]]:
        """Return entities with correct doc_id for each document."""
        return {
            "chemical": [
                ExtractedEntity(
                    text="glucose",
                    start_pos=0,
                    end_pos=7,
                    source_model="mock",
                    entity_type="chemical",
                    source_label="CHEMICAL",
                    doc_id=doc_id,
                    sentence_id=0,
                    sentence_text=text,
                    ner_confidence=0.9,
                )
            ]
        }

    mock_ner.extract_entities.side_effect = mock_extract_entities
    mock_ner_class.return_value = mock_ner

    # Setup mock EntityLinker instance
    mock_linker = MagicMock()
    mock_linker.link_entities_with_cache.return_value = {
        "glucose": {"umls_cui": "C0017725", "linking_confidence": 0.95}
    }
    mock_linker_class.return_value = mock_linker

    entities_by_doc, umls_mappings = process_documents_two_pass(
        texts, doc_ids, config_path=None, output_db=temp_db_path
    )

    # Verify results
    assert len(entities_by_doc) == 3
    assert isinstance(umls_mappings, dict)
    assert "glucose" in umls_mappings

    # Verify NER was called for each document
    assert mock_ner.extract_entities.call_count == 3


def test_process_documents_two_pass_no_doc_ids(
    mock_two_pass_deps: tuple[MagicMock, MagicMock],
    temp_db_path: str,
) -> None:
    """Test two-pass processing without providing doc_ids."""
    mock_ner_class, mock_linker_class = mock_two_pass_deps

    texts = ["Glucose metabolism", "Aspirin treatment"]

    # Setup mocks
    mock_ner = MagicMock()
    mock_ner.extract_entities.return_value = {"chemical": []}
    mock_ner_class.return_value = mock_ner

    mock_linker = MagicMock()
    mock_linker.link_entities_with_cache.return_value = {}
    mock_linker_class.return_value = mock_linker

    entities_by_doc, umls_mappings = process_documents_two_pass(
        texts, output_db=temp_db_path
    )

    # Should generate doc_ids automatically
    assert len(entities_by_doc) == 2
    assert isinstance(umls_mappings, dict)


def test_process_documents_two_pass_doc_ids_mismatch() -> None:
    """Test two-pass processing with mismatched doc_ids length."""
    texts = ["Text 1", "Text 2"]
    doc_ids = ["doc1"]  # Length mismatch

    with pytest.raises(ValueError, match="doc_ids length .* must match texts length"):
        process_documents_two_pass(texts, doc_ids)


def test_process_documents_two_pass_empty_input(
    mock_two_pass_deps: tuple[MagicMock, MagicMock],
) -> None:
    """Test two-pass processing with empty input."""
    mock_ner_class, mock_linker_class = mock_two_pass_deps

    # Setup mocks
    mock_ner = MagicMock()
    mock_ner_class.return_value = mock_ner

    mock_linker = MagicMock()
    mock_linker.link_entities_with_cache.return_value = {}
    mock_linker_class.return_value = mock_linker

    entities_by_doc, umls_mappings = process_documents_two_pass(
        [], output_db=":memory:"
    )

    assert entities_by_doc == []
    assert umls_mappings == {}
