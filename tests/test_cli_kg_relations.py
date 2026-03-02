"""Tests for CLI build-kg command."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner
from sqlmodel import Session, create_engine

from biomed_kg_agent.cli import cli
from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument
from biomed_kg_agent.kg.models import Cooccurrence, Entity, Mention, SQLModel
from biomed_kg_agent.nlp.models import ExtractedEntity


def create_test_pubmed_db(db_path: str) -> None:
    """Create a test database with sample PubMed documents."""
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Add sample PubMed documents
        doc1 = PubmedDocument(
            pmid="12345",
            title="Glucose and insulin interactions",
            abstract="Glucose levels affect insulin response. "
            "Diabetes involves glucose metabolism.",
            authors="Smith J, Doe A",
            journal="Test Journal",
            pub_date="2024-01-01",
            doi="10.1234/test",
        )
        doc2 = PubmedDocument(
            pmid="67890",
            title="Cancer and p53 mutations",
            abstract="p53 mutations are common in cancer. TP53 gene regulates cell death.",
            authors="Brown B",
            journal="Cancer Research",
            pub_date="2024-01-02",
            doi="10.1234/cancer",
        )
        session.add(doc1)
        session.add(doc2)
        session.commit()


def create_mock_processed_docs() -> list[DocumentInternal]:
    """Create mock processed documents with entities."""
    return [
        DocumentInternal(
            id="12345",
            title="Glucose and insulin interactions",
            text="Glucose levels affect insulin response. Diabetes involves glucose metabolism.",
            source="pubmed",
            pub_year=2024,
            entities=[
                ExtractedEntity(
                    text="glucose",
                    start_pos=0,
                    end_pos=7,
                    source_model="mock",
                    entity_type="chemical",
                    source_label="CHEMICAL",
                    doc_id="12345",
                    sentence_id=0,
                    sentence_text="Glucose levels affect insulin response.",
                    umls_cui="C0017725",
                    ner_confidence=None,
                    linking_confidence=None,
                    chebi_id=None,
                    go_id=None,
                    mesh_id=None,
                ),
                ExtractedEntity(
                    text="insulin",
                    start_pos=20,
                    end_pos=27,
                    source_model="mock",
                    entity_type="chemical",
                    source_label="CHEMICAL",
                    doc_id="12345",
                    sentence_id=0,
                    sentence_text="Glucose levels affect insulin response.",
                    umls_cui="C0021641",
                    ner_confidence=None,
                    linking_confidence=None,
                    chebi_id=None,
                    go_id=None,
                    mesh_id=None,
                ),
                ExtractedEntity(
                    text="diabetes",
                    start_pos=40,
                    end_pos=48,
                    source_model="mock",
                    entity_type="disease",
                    source_label="DISEASE",
                    doc_id="12345",
                    sentence_id=1,
                    sentence_text="Diabetes involves glucose metabolism.",
                    umls_cui="C0011849",
                    ner_confidence=None,
                    linking_confidence=None,
                    chebi_id=None,
                    go_id=None,
                    mesh_id=None,
                ),
            ],
        ),
        DocumentInternal(
            id="67890",
            title="Cancer and p53 mutations",
            text="p53 mutations are common in cancer. TP53 gene regulates cell death.",
            source="pubmed",
            pub_year=2024,
            entities=[
                ExtractedEntity(
                    text="p53",
                    start_pos=0,
                    end_pos=3,
                    source_model="mock",
                    entity_type="gene",
                    source_label="GENE",
                    doc_id="67890",
                    sentence_id=0,
                    sentence_text="p53 mutations are common in cancer.",
                    umls_cui="C0043227",
                    ner_confidence=None,
                    linking_confidence=None,
                    chebi_id=None,
                    go_id=None,
                    mesh_id=None,
                ),
                ExtractedEntity(
                    text="cancer",
                    start_pos=25,
                    end_pos=31,
                    source_model="mock",
                    entity_type="disease",
                    source_label="DISEASE",
                    doc_id="67890",
                    sentence_id=0,
                    sentence_text="p53 mutations are common in cancer.",
                    umls_cui="C0006826",
                    ner_confidence=None,
                    linking_confidence=None,
                    chebi_id=None,
                    go_id=None,
                    mesh_id=None,
                ),
            ],
        ),
    ]


@pytest.mark.unit
def test_build_kg_command_help() -> None:
    """Test build-kg command help text."""
    runner = CliRunner()
    result = runner.invoke(cli, ["build-kg", "--help"])

    assert result.exit_code == 0
    assert "Build knowledge graph" in result.output


@pytest.mark.unit
@patch("biomed_kg_agent.orchestrators.load_processed_documents")
def test_build_kg_full_workflow(mock_load: MagicMock) -> None:
    """Test the complete build-kg workflow with mocked NLP."""
    runner = CliRunner()

    # Mock the NLP processing to return our test entities
    mock_processed_docs = create_mock_processed_docs()
    mock_load.return_value = mock_processed_docs
    print(mock_processed_docs)
    with tempfile.TemporaryDirectory() as temp_dir:
        input_db = Path(temp_dir) / "input.db"
        output_db = Path(temp_dir) / "kg.db"

        # Create test input database
        create_test_pubmed_db(str(input_db))

        # Run the command
        result = runner.invoke(
            cli,
            [
                "build-kg",
                "--input",
                str(input_db),
                "--output",
                str(output_db),
            ],
        )
        print(result.output)
        # Verify command succeeded
        assert result.exit_code == 0
        assert "Knowledge graph built" in result.output
        assert "Entities: 5" in result.output  # All entities are unique by UMLS CUI
        assert "Mentions: 5" in result.output  # 5 total mentions
        assert "Relationships: 2" in result.output  # glucose+insulin, p53+cancer

        # Verify the output database was created and contains expected data
        engine = create_engine(f"sqlite:///{output_db}")
        with Session(engine) as session:
            from sqlmodel import select

            entities = session.exec(select(Entity)).all()
            mentions = session.exec(select(Mention)).all()
            cooccurrences = session.exec(select(Cooccurrence)).all()

            # Check we have the expected entities (deduplicated by UMLS CUI)
            entity_ids = {e.id for e in entities}
            expected_ids = {"C0017725", "C0021641", "C0011849", "C0043227", "C0006826"}
            assert entity_ids == expected_ids

            # Check mentions preserve original text
            mention_texts = {m.text for m in mentions}
            assert "glucose" in mention_texts
            assert "insulin" in mention_texts
            assert "diabetes" in mention_texts
            assert "p53" in mention_texts
            assert "cancer" in mention_texts

            # Check co-occurrences (glucose+insulin, p53+cancer)
            assert len(cooccurrences) == 2
            # Verify doc_ids_sample is populated
            for cooc in cooccurrences:
                assert cooc.sent_count == 1
                assert cooc.docs_count == 1
                assert cooc.doc_ids_sample is not None
                doc_ids = json.loads(cooc.doc_ids_sample)
                assert len(doc_ids) == 1  # Each pair appears in only one document


@pytest.mark.unit
@patch("biomed_kg_agent.nlp.persistence.load_processed_documents")
def test_build_kg_no_processed_docs(mock_load: MagicMock) -> None:
    """Test build-kg handles empty processed documents gracefully."""
    runner = CliRunner()
    mock_load.return_value = []

    with tempfile.TemporaryDirectory() as temp_dir:
        input_db = Path(temp_dir) / "input.db"

        # Create empty test database
        create_test_pubmed_db(str(input_db))

        result = runner.invoke(
            cli,
            [
                "build-kg",
                "--input",
                str(input_db),
            ],
        )

        assert result.exit_code == 0
        assert "No entities found" in result.output


@pytest.mark.unit
def test_build_kg_missing_input() -> None:
    """Test build-kg handles missing input file."""
    runner = CliRunner()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_db = Path(temp_dir) / "nonexistent.db"

        result = runner.invoke(
            cli,
            [
                "build-kg",
                "--input",
                str(input_db),
            ],
        )

        # Should fail when trying to load from non-existent database
        assert result.exit_code != 0
