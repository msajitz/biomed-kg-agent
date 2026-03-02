"""Tests for core pipeline functionality.

This module tests ONLY the pipeline orchestration logic using mocks.
It does NOT load real models and runs very fast.
"""

from unittest.mock import MagicMock, patch

from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.core.pipeline import process_documents
from biomed_kg_agent.nlp.models import ExtractedEntity


def test_process_documents_empty_list() -> None:
    """Test processing empty document list."""
    result = process_documents([], output_db=":memory:")
    assert result == []


@patch("biomed_kg_agent.core.pipeline.process_documents_two_pass")
def test_process_documents_logic(mock_two_pass: MagicMock) -> None:
    """Test the core pipeline logic with mocked dependencies.

    process_documents_two_pass now returns entities with UMLS already applied.
    The pipeline just flattens and assigns them to DocumentInternal objects.
    """
    # Setup test documents
    docs = [
        DocumentInternal(
            id="doc1", title="Test Doc 1", text="Glucose metabolism", source="test"
        ),
        DocumentInternal(
            id="doc2", title="Test Doc 2", text="Aspirin effects", source="test"
        ),
    ]

    # Mock two-pass processor output (entities already have UMLS data applied)
    mock_entities_by_doc = [
        {
            "chemical": [
                ExtractedEntity(
                    text="glucose",
                    start_pos=0,
                    end_pos=7,
                    source_model="mock_model",
                    entity_type="chemical",
                    source_label="CHEMICAL",
                    doc_id="doc1",
                    sentence_id=0,
                    sentence_text="Glucose metabolism",
                    umls_cui="C0017725",
                    linking_confidence=0.95,
                )
            ]
        },
        {
            "chemical": [
                ExtractedEntity(
                    text="aspirin",
                    start_pos=0,
                    end_pos=7,
                    source_model="mock_model",
                    entity_type="chemical",
                    source_label="CHEMICAL",
                    doc_id="doc2",
                    sentence_id=0,
                    sentence_text="Aspirin effects",
                    umls_cui="C0004057",
                    linking_confidence=0.90,
                )
            ]
        },
    ]
    mock_umls_mappings = {
        "glucose": {"umls_cui": "C0017725", "linking_confidence": 0.95},
        "aspirin": {"umls_cui": "C0004057", "linking_confidence": 0.90},
    }
    mock_two_pass.return_value = (mock_entities_by_doc, mock_umls_mappings)

    # Test the pipeline
    result = process_documents(docs, config_path=None, output_db=":memory:")

    # Verify two-pass processor was called correctly
    expected_metadata: dict[str, dict] = {
        "doc1": {
            "title": "Test Doc 1",
            "source": "test",
            "pub_year": None,
            "extras": {},
        },
        "doc2": {
            "title": "Test Doc 2",
            "source": "test",
            "pub_year": None,
            "extras": {},
        },
    }
    mock_two_pass.assert_called_once_with(
        ["Glucose metabolism", "Aspirin effects"],
        ["doc1", "doc2"],
        None,  # config_path
        ":memory:",  # output_db
        expected_metadata,
    )

    # Verify results
    assert len(result) == 2
    assert result[0].id == "doc1"
    assert result[1].id == "doc2"

    # Check entities were flattened and assigned (UMLS already applied by processor)
    assert len(result[0].entities) == 1
    assert result[0].entities[0].text == "glucose"
    assert result[0].entities[0].umls_cui == "C0017725"

    assert len(result[1].entities) == 1
    assert result[1].entities[0].text == "aspirin"
    assert result[1].entities[0].umls_cui == "C0004057"


@patch("biomed_kg_agent.core.pipeline.process_documents_two_pass")
def test_process_documents_handles_no_entities(mock_two_pass: MagicMock) -> None:
    """Test pipeline handles documents with no entities gracefully."""
    docs = [DocumentInternal(id="doc1", title="Empty", text="Just text", source="test")]

    # Mock no entities found
    mock_two_pass.return_value = ([{}], {})

    result = process_documents(docs, output_db=":memory:")

    assert len(result) == 1
    assert len(result[0].entities) == 0


@patch("biomed_kg_agent.core.pipeline.process_documents_two_pass")
def test_process_documents_default_parameters(mock_two_pass: MagicMock) -> None:
    """Test pipeline passes correct parameters to two-pass processor."""
    docs = [DocumentInternal(id="doc1", title="Test", text="Test text", source="test")]

    mock_two_pass.return_value = ([{}], {})

    process_documents(
        docs, output_db=":memory:"
    )  # Specify output_db for test isolation

    # Verify parameters passed to two_pass processor
    expected_metadata: dict[str, dict] = {
        "doc1": {"title": "Test", "source": "test", "pub_year": None, "extras": {}}
    }
    mock_two_pass.assert_called_once_with(
        ["Test text"],
        ["doc1"],
        None,  # config_path
        ":memory:",  # output_db
        expected_metadata,
    )
