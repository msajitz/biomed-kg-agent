"""Tests for knowledge graph extraction orchestration function."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.nlp.models import ExtractedEntity
from biomed_kg_agent.orchestrators import build_knowledge_graph


@pytest.mark.unit
@patch("biomed_kg_agent.orchestrators.save_kg_data")
@patch("biomed_kg_agent.orchestrators.load_processed_documents")
def test_build_knowledge_graph_success(
    mock_load: MagicMock,
    mock_save: MagicMock,
    tmp_path: Path,
) -> None:
    """Test successful knowledge graph construction."""
    input_db = str(tmp_path / "nlp.db")
    output_db = str(tmp_path / "kg.db")

    doc = DocumentInternal(
        id="doc1",
        title="Test Document",
        text="Cancer treatment involves chemotherapy and radiation.",
        source="pubmed",
        entities=[
            ExtractedEntity(
                text="Cancer",
                entity_type="Disease",
                start_pos=0,
                end_pos=6,
                sentence_id=0,
                sentence_text="Cancer treatment involves chemotherapy and radiation.",
            ),
            ExtractedEntity(
                text="chemotherapy",
                entity_type="Treatment",
                start_pos=25,
                end_pos=37,
                sentence_id=0,
                sentence_text="Cancer treatment involves chemotherapy and radiation.",
            ),
            ExtractedEntity(
                text="radiation",
                entity_type="Treatment",
                start_pos=42,
                end_pos=51,
                sentence_id=0,
                sentence_text="Cancer treatment involves chemotherapy and radiation.",
            ),
        ],
    )

    mock_load.return_value = [doc]

    result = build_knowledge_graph(input_db, output_db)

    assert result["entities"] > 0
    assert result["mentions"] > 0
    assert result["cooccurrences"] > 0
    assert result["output_path"] == output_db
    assert result["method"] == "cooccurrence"

    # Verify save_kg_data was called with entities, mentions, and cooccurrences
    mock_save.assert_called_once()
    call_args = mock_save.call_args[0]
    assert len(call_args) == 4  # entities, mentions, cooccurrences, output_path


@pytest.mark.unit
@patch("biomed_kg_agent.orchestrators.load_processed_documents")
def test_build_knowledge_graph_empty_database(
    mock_load: MagicMock, tmp_path: Path
) -> None:
    """Test KG construction with empty database."""
    input_db = str(tmp_path / "empty.db")
    output_db = str(tmp_path / "kg.db")

    mock_load.return_value = []

    result = build_knowledge_graph(input_db, output_db)

    assert result["entities"] == 0
    assert result["mentions"] == 0
    assert result["cooccurrences"] == 0
    assert result["output_path"] == output_db


@pytest.mark.unit
@patch("biomed_kg_agent.orchestrators.save_kg_data")
@patch("biomed_kg_agent.orchestrators.load_processed_documents")
def test_build_knowledge_graph_multiple_documents(
    mock_load: MagicMock,
    _mock_save: MagicMock,
    tmp_path: Path,
) -> None:
    """Test KG construction with multiple documents."""
    input_db = str(tmp_path / "nlp.db")
    output_db = str(tmp_path / "kg.db")

    doc1 = DocumentInternal(
        id="doc1",
        title="Doc 1",
        text="Cancer and diabetes are diseases.",
        source="pubmed",
        entities=[
            ExtractedEntity(
                text="Cancer",
                entity_type="Disease",
                start_pos=0,
                end_pos=6,
                sentence_id=0,
                sentence_text="Cancer and diabetes are diseases.",
            ),
            ExtractedEntity(
                text="diabetes",
                entity_type="Disease",
                start_pos=11,
                end_pos=19,
                sentence_id=0,
                sentence_text="Cancer and diabetes are diseases.",
            ),
        ],
    )

    doc2 = DocumentInternal(
        id="doc2",
        title="Doc 2",
        text="Cancer treatment uses chemotherapy.",
        source="pubmed",
        entities=[
            ExtractedEntity(
                text="Cancer",
                entity_type="Disease",
                start_pos=0,
                end_pos=6,
                sentence_id=0,
                sentence_text="Cancer treatment uses chemotherapy.",
            ),
            ExtractedEntity(
                text="chemotherapy",
                entity_type="Treatment",
                start_pos=22,
                end_pos=34,
                sentence_id=0,
                sentence_text="Cancer treatment uses chemotherapy.",
            ),
        ],
    )

    mock_load.return_value = [doc1, doc2]

    result = build_knowledge_graph(input_db, output_db)

    # Should have entities from both documents
    assert result["entities"] > 0
    assert result["mentions"] > 0
    # Should have co-occurrences (Cancer-diabetes in doc1, Cancer-chemotherapy in doc2)
    assert result["cooccurrences"] > 0


@pytest.mark.unit
@patch("biomed_kg_agent.orchestrators.save_kg_data")
@patch("biomed_kg_agent.orchestrators.load_processed_documents")
def test_build_knowledge_graph_no_cooccurrences(
    mock_load: MagicMock,
    _mock_save: MagicMock,
    tmp_path: Path,
) -> None:
    """Test KG construction when entities don't co-occur."""
    input_db = str(tmp_path / "nlp.db")
    output_db = str(tmp_path / "kg.db")

    # Document with entities in different sentences (no co-occurrence)
    doc = DocumentInternal(
        id="doc1",
        title="Test",
        text="First sentence. Second sentence.",
        source="pubmed",
        entities=[
            ExtractedEntity(
                text="First",
                entity_type="Other",
                start_pos=0,
                end_pos=5,
                sentence_id=0,
                sentence_text="First sentence.",
            ),
            ExtractedEntity(
                text="Second",
                entity_type="Other",
                start_pos=17,
                end_pos=23,
                sentence_id=1,
                sentence_text="Second sentence.",
            ),
        ],
    )

    mock_load.return_value = [doc]

    result = build_knowledge_graph(input_db, output_db)

    # Should have entities and mentions but no co-occurrences
    assert result["entities"] > 0
    assert result["mentions"] > 0
    assert result["cooccurrences"] == 0
