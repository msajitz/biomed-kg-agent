"""Tests for database-driven functionality in two-pass processing module.

This module tests database-driven incremental processing and auto-resume
functionality for memory optimization and fault tolerance.

Uses mocks to avoid loading heavy models for fast testing.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from biomed_kg_agent.nlp.models import ExtractedEntity


@patch("biomed_kg_agent.nlp.two_pass_processor.create_engine")
@patch("biomed_kg_agent.nlp.two_pass_processor.Session")
def test_get_processed_doc_ids_fresh_start(
    mock_session_class: Any, mock_create_engine: Any, temp_db_path: Any
) -> None:
    """Test _get_processed_doc_ids with no existing data (fresh start)."""
    from biomed_kg_agent.nlp.config import load_ner_config
    from biomed_kg_agent.nlp.two_pass_processor import _get_processed_doc_ids

    mock_session = MagicMock()
    mock_config_result = MagicMock()
    mock_config_result.first.return_value = None
    mock_ids_result = MagicMock()
    mock_ids_result.all.return_value = []
    mock_session.exec.side_effect = [mock_config_result, mock_ids_result]
    mock_session_class.return_value.__enter__.return_value = mock_session

    config = load_ner_config()
    result = _get_processed_doc_ids(config, temp_db_path)

    assert result == set()
    mock_create_engine.assert_called_once_with(f"sqlite:///{temp_db_path}")


@patch("biomed_kg_agent.nlp.two_pass_processor.create_engine")
@patch("biomed_kg_agent.nlp.two_pass_processor.Session")
def test_get_processed_doc_ids_partial(
    mock_session_class: Any, mock_create_engine: Any, temp_db_path: Any
) -> None:
    """Test _get_processed_doc_ids returns correct set of processed IDs."""
    from biomed_kg_agent.nlp.config import load_ner_config
    from biomed_kg_agent.nlp.two_pass_processor import _get_processed_doc_ids

    config = load_ner_config()
    processed = [f"doc_{i}" for i in range(150)]

    mock_session = MagicMock()
    mock_config_result = MagicMock()
    mock_config_result.first.return_value = None
    mock_ids_result = MagicMock()
    mock_ids_result.all.return_value = processed
    mock_session.exec.side_effect = [mock_config_result, mock_ids_result]
    mock_session_class.return_value.__enter__.return_value = mock_session

    result = _get_processed_doc_ids(config, temp_db_path)

    assert result == set(processed)
    assert len(result) == 150


@patch("biomed_kg_agent.nlp.two_pass_processor.create_engine")
def test_get_processed_doc_ids_db_error(
    mock_create_engine: Any, temp_db_path: Any
) -> None:
    """Test _get_processed_doc_ids returns empty set on DB errors
    (treats all docs as unprocessed)."""
    from biomed_kg_agent.nlp.config import load_ner_config
    from biomed_kg_agent.nlp.two_pass_processor import _get_processed_doc_ids

    mock_create_engine.side_effect = Exception("Database connection failed")

    config = load_ner_config()
    result = _get_processed_doc_ids(config, temp_db_path)

    assert result == set()


@patch("biomed_kg_agent.nlp.two_pass_processor.create_engine")
@patch("biomed_kg_agent.nlp.two_pass_processor.Session")
def test_get_processed_doc_ids_config_mismatch(
    mock_session_class: Any, mock_create_engine: Any, temp_db_path: Any
) -> None:
    """Test _get_processed_doc_ids raises ValueError on config mismatch."""
    from biomed_kg_agent.nlp.config import LinkerConfig, NerConfig
    from biomed_kg_agent.nlp.two_pass_processor import _get_processed_doc_ids

    current_config = NerConfig(
        model_priorities={"bionlp": 3, "craft": 2, "bc5cdr": 1},
        linker=LinkerConfig(
            enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
        ),
    )

    stored_config = NerConfig(
        model_priorities={"bionlp": 1, "craft": 1, "bc5cdr": 1},
        linker=LinkerConfig(
            enabled=True,
            core_model="en_core_sci_sm",
            confidence_threshold=0.5,
        ),
    )

    mock_doc = MagicMock()
    mock_doc.config_json = stored_config.model_dump_json()

    mock_session = MagicMock()
    mock_session.exec.return_value.first.return_value = mock_doc
    mock_session_class.return_value.__enter__.return_value = mock_session

    with pytest.raises(ValueError, match="Config mismatch"):
        _get_processed_doc_ids(current_config, temp_db_path)


@patch("biomed_kg_agent.nlp.two_pass_processor._load_results_in_order")
@patch("biomed_kg_agent.nlp.two_pass_processor._update_entities_in_database")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_unique_entity_texts_from_db")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_processed_doc_ids")
def test_process_documents_auto_resume_completed(
    mock_get_processed: Any,
    mock_get_unique: Any,
    _mock_update_db: Any,
    mock_load_results: Any,
    mock_two_pass_deps: Any,
) -> None:
    """Test auto-resume when all documents are already processed in Pass 1."""
    from biomed_kg_agent.nlp.two_pass_processor import process_documents_two_pass

    mock_ner_class, mock_linker_class = mock_two_pass_deps

    texts = ["Text 1", "Text 2"]
    doc_ids = ["doc1", "doc2"]

    mock_get_processed.return_value = {"doc1", "doc2"}
    mock_get_unique.return_value = {"entity_text"}
    mock_load_results.return_value = [{}, {}]

    mock_linker = MagicMock()
    mock_linker.link_entities_with_cache.return_value = {
        "entity_text": {"umls_cui": None}
    }
    mock_linker_class.return_value = mock_linker

    result = process_documents_two_pass(texts, doc_ids, output_db=":memory:")

    mock_ner_class.assert_not_called()
    mock_get_unique.assert_called_once()

    entities_by_doc, umls_mappings = result
    assert len(entities_by_doc) == 2


@patch("biomed_kg_agent.nlp.two_pass_processor._load_results_in_order")
@patch("biomed_kg_agent.nlp.two_pass_processor._update_entities_in_database")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_unique_entity_texts_from_db")
@patch("biomed_kg_agent.nlp.two_pass_processor._extract_entities_to_db")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_processed_doc_ids")
def test_process_documents_auto_resume_partial(
    mock_get_processed: Any,
    mock_extract: Any,
    mock_get_unique: Any,
    _mock_update_db: Any,
    mock_load_results: Any,
    mock_two_pass_deps: Any,
) -> None:
    """Test auto-resume when some documents are already processed."""
    from biomed_kg_agent.nlp.two_pass_processor import process_documents_two_pass

    mock_ner_class, mock_linker_class = mock_two_pass_deps

    texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
    doc_ids = ["doc1", "doc2", "doc3", "doc4"]

    mock_get_processed.return_value = {"doc1", "doc2"}
    mock_get_unique.return_value = {"new_entity"}
    mock_load_results.return_value = [{}, {}, {}, {}]

    mock_linker = MagicMock()
    mock_linker.link_entities_with_cache.return_value = {
        "new_entity": {"umls_cui": "C123"}
    }
    mock_linker_class.return_value = mock_linker

    result = process_documents_two_pass(texts, doc_ids, output_db=":memory:")

    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[0] == ["Text 3", "Text 4"]
    assert call_args[1] == ["doc3", "doc4"]
    assert call_args[2] == 2  # already_done count

    entities_by_doc, umls_mappings = result
    assert len(entities_by_doc) == 4


@patch("biomed_kg_agent.nlp.two_pass_processor._load_results_in_order")
@patch("biomed_kg_agent.nlp.two_pass_processor._update_entities_in_database")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_unique_entity_texts_from_db")
@patch("biomed_kg_agent.nlp.two_pass_processor._extract_entities_to_db")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_processed_doc_ids")
def test_resume_retries_failed_docs(
    mock_get_processed: Any,
    mock_extract: Any,
    mock_get_unique: Any,
    _mock_update_db: Any,
    mock_load_results: Any,
    mock_two_pass_deps: Any,
) -> None:
    """Test that docs missing from DB (e.g. NER failure) are retried on resume."""
    from biomed_kg_agent.nlp.two_pass_processor import process_documents_two_pass

    _mock_ner_class, mock_linker_class = mock_two_pass_deps

    texts = ["Text 1", "Text 2", "Text 3"]
    doc_ids = ["doc1", "doc2", "doc3"]

    # doc2 failed previously: only doc1 and doc3 are in the DB
    mock_get_processed.return_value = {"doc1", "doc3"}
    mock_get_unique.return_value = set()
    mock_load_results.return_value = [{}, {}, {}]

    mock_linker = MagicMock()
    mock_linker.link_entities_with_cache.return_value = {}
    mock_linker_class.return_value = mock_linker

    process_documents_two_pass(texts, doc_ids, output_db=":memory:")

    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[0] == ["Text 2"]
    assert call_args[1] == ["doc2"]
    assert call_args[2] == 2  # already_done count


@patch("biomed_kg_agent.nlp.two_pass_processor._load_results_in_order")
@patch("biomed_kg_agent.nlp.two_pass_processor._update_entities_in_database")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_unique_entity_texts_from_db")
@patch("biomed_kg_agent.nlp.two_pass_processor._extract_entities_to_db")
@patch("biomed_kg_agent.nlp.two_pass_processor._get_processed_doc_ids")
def test_resume_with_corpus_growth(
    mock_get_processed: Any,
    mock_extract: Any,
    mock_get_unique: Any,
    _mock_update_db: Any,
    mock_load_results: Any,
    mock_two_pass_deps: Any,
) -> None:
    """Test that new docs added to corpus are processed, existing ones skipped."""
    from biomed_kg_agent.nlp.two_pass_processor import process_documents_two_pass

    _mock_ner_class, mock_linker_class = mock_two_pass_deps

    texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
    doc_ids = ["doc1", "doc2", "doc3", "doc4"]

    # Previous run processed doc1 and doc2; doc3/doc4 are new additions
    mock_get_processed.return_value = {"doc1", "doc2"}
    mock_get_unique.return_value = set()
    mock_load_results.return_value = [{}, {}, {}, {}]

    mock_linker = MagicMock()
    mock_linker.link_entities_with_cache.return_value = {}
    mock_linker_class.return_value = mock_linker

    process_documents_two_pass(texts, doc_ids, output_db=":memory:")

    mock_extract.assert_called_once()
    call_args = mock_extract.call_args[0]
    assert call_args[0] == ["Text 3", "Text 4"]
    assert call_args[1] == ["doc3", "doc4"]


@patch("biomed_kg_agent.nlp.two_pass_processor.save_nlp_results")
@patch("biomed_kg_agent.nlp.two_pass_processor.BiomedicalNER")
def test_extract_entities_to_db_incremental_saves(
    mock_ner_class: Any, mock_save_to_db: Any, temp_db_path: Any
) -> None:
    """Test that _extract_entities_to_db saves to database every 100 documents."""
    from biomed_kg_agent.nlp.two_pass_processor import _extract_entities_to_db

    # Create 250 test documents (should trigger 3 saves: 100, 100, 50)
    texts = [f"Test document {i}" for i in range(250)]
    doc_ids = [f"doc_{i}" for i in range(250)]

    # Mock NER to return simple entities
    mock_ner = MagicMock()

    def mock_extract_entities(text: str, doc_id: str) -> dict:
        return {
            "chemical": [
                ExtractedEntity(
                    text="test_entity",
                    start_pos=0,
                    end_pos=11,
                    source_model="mock",
                    entity_type="chemical",
                    source_label="CHEMICAL",
                    doc_id=doc_id,
                    sentence_id=0,
                    sentence_text=text,
                )
            ]
        }

    mock_ner.extract_entities.side_effect = mock_extract_entities
    mock_ner_class.return_value = mock_ner

    from biomed_kg_agent.nlp.config import load_ner_config

    config = load_ner_config()
    _extract_entities_to_db(texts, doc_ids, 0, config, temp_db_path)

    # Should save 3 times during Pass 1: at 100, 200, and final batch of 50
    assert mock_save_to_db.call_count == 3

    # Verify batch sizes
    call_args_list = mock_save_to_db.call_args_list
    assert len(call_args_list[0][0][0]) == 100  # First batch
    assert len(call_args_list[1][0][0]) == 100  # Second batch
    assert len(call_args_list[2][0][0]) == 50  # Final batch

    # Verify that all documents were processed
    assert mock_ner.extract_entities.call_count == 250


@patch("biomed_kg_agent.nlp.two_pass_processor.save_nlp_results")
@patch("biomed_kg_agent.nlp.two_pass_processor.BiomedicalNER")
def test_extract_entities_to_db_error_handling(
    mock_ner_class: Any, mock_save_to_db: Any, temp_db_path: Any
) -> None:
    """Test that _extract_entities_to_db handles individual document errors gracefully."""
    from biomed_kg_agent.nlp.two_pass_processor import _extract_entities_to_db

    texts = ["Good text", "Bad text", "Another good text"]
    doc_ids = ["doc1", "doc2", "doc3"]

    # Mock NER to fail on second document
    mock_ner = MagicMock()

    def side_effect(text: str, doc_id: str) -> dict:
        if "Bad text" in text:
            raise Exception("Processing failed")
        return {"chemical": []}

    mock_ner.extract_entities.side_effect = side_effect
    mock_ner_class.return_value = mock_ner

    from biomed_kg_agent.nlp.config import load_ner_config

    config = load_ner_config()

    # Should not raise - errors are logged and processing continues
    _extract_entities_to_db(texts, doc_ids, 0, config, temp_db_path)

    # All 3 documents should be attempted
    assert mock_ner.extract_entities.call_count == 3

    # Only 2 good docs saved in final batch (failed doc skipped)
    assert mock_save_to_db.call_count == 1
    assert len(mock_save_to_db.call_args_list[0][0][0]) == 2


@patch("biomed_kg_agent.nlp.two_pass_processor.load_entities")
def test_load_results_in_order_error_handling(mock_load_entities: Any) -> None:
    """Test RuntimeError on database error to prevent silent data loss."""
    from biomed_kg_agent.nlp.two_pass_processor import _load_results_in_order

    mock_load_entities.side_effect = Exception("Database read failed")

    with pytest.raises(RuntimeError, match="Failed to load results"):
        _load_results_in_order(["doc1", "doc2"], "test.db")

    mock_load_entities.assert_called_once_with(db_path="test.db")


@patch("biomed_kg_agent.nlp.two_pass_processor.load_entities")
def test_load_results_in_order_preserves_doc_order(mock_load_entities: Any) -> None:
    """Test _load_results_in_order returns entities in the order of doc_ids."""
    from biomed_kg_agent.nlp.two_pass_processor import _load_results_in_order

    # Mock DB returns entities keyed by doc_id (arbitrary order)
    mock_load_entities.return_value = {
        "doc2": [
            ExtractedEntity(
                text="p53",
                start_pos=0,
                end_pos=3,
                source_model="mock",
                entity_type="gene",
                source_label="GENE",
                doc_id="doc2",
                sentence_id=0,
                sentence_text="p53 test",
            ),
        ],
        "doc1": [
            ExtractedEntity(
                text="glucose",
                start_pos=0,
                end_pos=7,
                source_model="mock",
                entity_type="chemical",
                source_label="CHEMICAL",
                doc_id="doc1",
                sentence_id=0,
                sentence_text="Glucose test",
            ),
            ExtractedEntity(
                text="aspirin",
                start_pos=0,
                end_pos=7,
                source_model="mock",
                entity_type="chemical",
                source_label="CHEMICAL",
                doc_id="doc1",
                sentence_id=0,
                sentence_text="Aspirin test",
            ),
        ],
        "doc3": [],
    }

    # Request in specific order: doc1, doc2, doc3
    result = _load_results_in_order(["doc1", "doc2", "doc3"], "test.db")

    assert len(result) == 3

    # First: doc1 with 2 chemicals
    assert "chemical" in result[0]
    assert len(result[0]["chemical"]) == 2

    # Second: doc2 with 1 gene
    assert "gene" in result[1]
    assert len(result[1]["gene"]) == 1

    # Third: doc3 empty
    assert result[2] == {}

    mock_load_entities.assert_called_once_with(db_path="test.db")
