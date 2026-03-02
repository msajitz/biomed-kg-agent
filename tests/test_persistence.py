"""Tests for NLP results persistence (models and save/load logic).

Tests cover JSON serialization, database round-trips, and filtering.
Uses temporary databases for isolation.
"""

import tempfile
from datetime import datetime
from pathlib import Path

from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.nlp.config import get_default_config
from biomed_kg_agent.nlp.models import ExtractedEntity
from biomed_kg_agent.nlp.persistence import (
    build_processed_document,
    get_document_by_id,
    get_processing_info,
    load_entities,
    load_processed_documents,
    save_nlp_results,
)

# ============================================================================
# build_processed_document() Tests
# ============================================================================
# Note: JSON field persistence (umls_semantic_types, parent_classes) is tested
# via round-trip save/load tests below, which exercise SQLAlchemy's JSON column handling.


def test_build_processed_document_extracts_whitelisted_fields() -> None:
    """Test whitelisted extras become dedicated fields."""
    doc = DocumentInternal(
        id="PMID123",
        title="Test Article",
        text="Abstract text",
        source="pubmed",
        pub_year=2023,
        extras={
            "journal": "Nature",
            "authors": "Smith J, Doe A",
            "doi": "10.1234/test",
            "document_type": "Journal Article",
            "keywords": "biology, chemistry",
            "mesh_terms": "D001234, D005678",
            "custom_field": "should_go_to_json",
        },
        entities=[],
    )

    config = get_default_config()
    processed = build_processed_document(doc, config)

    # Verify whitelisted fields
    assert processed.journal == "Nature"
    assert processed.authors == "Smith J, Doe A"
    assert processed.doi == "10.1234/test"
    assert processed.document_type == "Journal Article"
    assert processed.keywords == "biology, chemistry"
    assert processed.mesh_terms == "D001234, D005678"

    # Verify remaining extras in JSON
    assert processed.extras_json == '{"custom_field": "should_go_to_json"}'

    # Verify config is stored
    assert processed.config_json is not None


def test_build_processed_document_handles_empty_extras() -> None:
    """Test document with no extras."""
    doc = DocumentInternal(
        id="PMID456",
        title="Test",
        text="Text",
        source="pubmed",
        pub_year=2023,
        extras={},
        entities=[],
    )

    config = get_default_config()
    processed = build_processed_document(doc, config)

    assert processed.journal is None
    assert processed.authors is None
    assert processed.extras_json is None
    assert processed.config_json is not None


# ============================================================================
# save_nlp_results() + load_processed_documents() Round-trip Tests
# ============================================================================


def test_save_and_load_documents_round_trip_complete() -> None:
    """Test full round-trip with all fields populated."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        # Create test documents
        entities = [
            ExtractedEntity(
                text="glucose",
                start_pos=0,
                end_pos=7,
                source_model="scispacy",
                entity_type="chemical",
                source_label="CHEMICAL",
                doc_id="PMID123",
                sentence_id=0,
                sentence_text="glucose is mentioned",
                ner_confidence=0.95,
                linking_confidence=0.88,
                umls_cui="C0017725",
                umls_preferred_name="Glucose",
                umls_semantic_types=["T109", "T121"],
                chebi_id="CHEBI:17234",
                mesh_id="D005947",
                parent_classes=["CHEBI:24431"],
            ),
            ExtractedEntity(
                text="diabetes",
                start_pos=20,
                end_pos=28,
                source_model="scispacy",
                entity_type="disease",
                source_label="DISEASE",
                doc_id="PMID123",
                sentence_id=1,
                sentence_text="diabetes mentioned",
                linking_confidence=0.92,
                umls_cui="C0011847",
            ),
        ]

        docs = [
            DocumentInternal(
                id="PMID123",
                title="Glucose and Diabetes Study",
                text="Full text here",
                source="pubmed",
                pub_year=2023,
                extras={
                    "journal": "Nature Medicine",
                    "authors": "Smith J",
                    "doi": "10.1234/test",
                },
                entities=entities,
            )
        ]

        # Save
        config = get_default_config()
        save_nlp_results(docs, config, db_path=db_path)

        # Load
        loaded_docs = load_processed_documents(db_path=db_path)

        assert len(loaded_docs) == 1
        loaded = loaded_docs[0]

        # Verify document fields
        assert loaded.id == "PMID123"
        assert loaded.title == "Glucose and Diabetes Study"
        assert loaded.text == "Full text here"
        assert loaded.source == "pubmed"
        assert loaded.pub_year == 2023

        # Verify entities
        assert len(loaded.entities) == 2

        glucose = next(e for e in loaded.entities if e.text == "glucose")
        assert glucose.umls_cui == "C0017725"
        assert glucose.umls_preferred_name == "Glucose"
        assert glucose.umls_semantic_types == ["T109", "T121"]
        assert glucose.chebi_id == "CHEBI:17234"
        assert glucose.parent_classes == ["CHEBI:24431"]
        assert glucose.ner_confidence == 0.95
        assert glucose.linking_confidence == 0.88

        diabetes = next(e for e in loaded.entities if e.text == "diabetes")
        assert diabetes.umls_cui == "C0011847"
        assert diabetes.linking_confidence == 0.92
        assert diabetes.ner_confidence is None  # Not set


def test_save_and_load_documents_with_minimal_fields() -> None:
    """Test round-trip with only required fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        entities = [
            ExtractedEntity(
                text="protein",
                start_pos=0,
                end_pos=7,
                source_model="test",
                entity_type="protein",
                source_label="PROTEIN",
                doc_id="DOC001",
                sentence_id=0,
                sentence_text="protein mentioned",
                # No optional fields
            )
        ]

        docs = [
            DocumentInternal(
                id="DOC001",
                title="Minimal Doc",
                text="Text",
                source="test_source",
                pub_year=None,
                extras={},
                entities=entities,
            )
        ]

        config = get_default_config()
        save_nlp_results(docs, config, db_path=db_path)
        loaded_docs = load_processed_documents(db_path=db_path)

        assert len(loaded_docs) == 1
        assert loaded_docs[0].id == "DOC001"
        assert len(loaded_docs[0].entities) == 1
        assert loaded_docs[0].entities[0].umls_cui is None
        assert loaded_docs[0].entities[0].linking_confidence is None


def test_save_empty_document_list() -> None:
    """Test save_nlp_results with empty list."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        # Should log warning but not crash
        config = get_default_config()
        save_nlp_results([], config, db_path=db_path)

        # Database should exist but be empty
        loaded = load_processed_documents(db_path=db_path)
        assert len(loaded) == 0


def test_save_document_with_no_entities() -> None:
    """Test document with zero entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        docs = [
            DocumentInternal(
                id="EMPTY001",
                title="No Entities",
                text="Text with no biomedical terms",
                source="test",
                pub_year=2023,
                extras={},
                entities=[],
            )
        ]

        config = get_default_config()
        save_nlp_results(docs, config, db_path=db_path)
        loaded_docs = load_processed_documents(db_path=db_path)

        assert len(loaded_docs) == 1
        assert loaded_docs[0].id == "EMPTY001"
        assert len(loaded_docs[0].entities) == 0


# ============================================================================
# load_entities() Filtering Tests
# ============================================================================


def test_load_entities_returns_dict_grouped_by_doc_id() -> None:
    """Test load_entities groups entities by document."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        docs = [
            DocumentInternal(
                id="DOC1",
                title="First",
                text="Text",
                source="test",
                entities=[
                    ExtractedEntity(
                        text="entity1",
                        start_pos=0,
                        end_pos=7,
                        source_model="test",
                        entity_type="chemical",
                        source_label="CHEMICAL",
                        doc_id="DOC1",
                        sentence_id=0,
                        sentence_text="entity1 here",
                    )
                ],
            ),
            DocumentInternal(
                id="DOC2",
                title="Second",
                text="Text",
                source="test",
                entities=[
                    ExtractedEntity(
                        text="entity2",
                        start_pos=0,
                        end_pos=7,
                        source_model="test",
                        entity_type="protein",
                        source_label="PROTEIN",
                        doc_id="DOC2",
                        sentence_id=0,
                        sentence_text="entity2 here",
                    ),
                    ExtractedEntity(
                        text="entity3",
                        start_pos=10,
                        end_pos=17,
                        source_model="test",
                        entity_type="disease",
                        source_label="DISEASE",
                        doc_id="DOC2",
                        sentence_id=1,
                        sentence_text="entity3 here",
                    ),
                ],
            ),
        ]

        config = get_default_config()
        save_nlp_results(docs, config, db_path=db_path)

        entities_by_doc = load_entities(db_path=db_path)

        assert len(entities_by_doc) == 2
        assert len(entities_by_doc["DOC1"]) == 1
        assert len(entities_by_doc["DOC2"]) == 2
        assert entities_by_doc["DOC1"][0].text == "entity1"


# Tests for use_case and linking_profile filtering removed -
# config is now enforced at DB level via strict validation


# ============================================================================
# get_processing_info() Tests
# ============================================================================


def test_get_processing_info_returns_metadata() -> None:
    """Test get_processing_info returns correct metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        docs = [
            DocumentInternal(
                id=f"DOC{i}",
                title=f"Document {i}",
                text="Text",
                source="test",
                entities=[
                    ExtractedEntity(
                        text=f"entity{i}",
                        start_pos=0,
                        end_pos=7,
                        source_model="test",
                        entity_type="chemical",
                        source_label="CHEMICAL",
                        doc_id=f"DOC{i}",
                        sentence_id=0,
                        sentence_text=f"entity{i}",
                    )
                ],
            )
            for i in range(3)
        ]

        config = get_default_config()
        save_nlp_results(docs, config, db_path=db_path)

        info = get_processing_info(db_path=db_path)

        assert info is not None
        assert info["document_count"] == 3
        assert info["total_entities"] == 3
        assert "config" in info
        assert isinstance(info["config"], dict)
        assert "model_priorities" in info["config"]
        assert "linker" in info["config"]
        assert isinstance(info["processed_at"], datetime)


def test_get_processing_info_returns_none_for_empty_db() -> None:
    """Test get_processing_info with no data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "empty.db")

        # Create empty database
        config = get_default_config()
        save_nlp_results([], config, db_path=db_path)

        info = get_processing_info(db_path=db_path)
        assert info is None


# ============================================================================
# get_document_by_id() Tests
# ============================================================================


def test_get_document_by_id_returns_dict() -> None:
    """Test fetching an existing document returns a dict with expected fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        docs = [
            DocumentInternal(
                id="PMID999",
                title="",
                text="Abstract text",
                source="pubmed",
                pub_year=2024,
                extras={"journal": "Science"},
                entities=[],
            )
        ]

        config = get_default_config()
        save_nlp_results(docs, config, db_path=db_path)

        result = get_document_by_id("PMID999", db_path=db_path)
        assert result is not None
        assert result["id"] == "PMID999"
        assert result["text"] == "Abstract text"
        assert result["title"] == ""
        assert result["journal"] == "Science"


def test_get_document_by_id_returns_none_for_missing() -> None:
    """Test fetching a nonexistent ID returns None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "test.db")

        config = get_default_config()
        save_nlp_results([], config, db_path=db_path)

        assert get_document_by_id("NONEXISTENT", db_path=db_path) is None
