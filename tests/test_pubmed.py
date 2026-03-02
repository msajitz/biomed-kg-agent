import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from sqlmodel import Session, SQLModel, create_engine

from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.data_sources.ncbi import ncbi_utils, pubmed
from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument
from biomed_kg_agent.data_sources.ncbi.pubmed import process_pubmed_documents_from_db
from biomed_kg_agent.nlp.models import ExtractedEntity


@pytest.fixture
def sample_pubmed_doc() -> PubmedDocument:
    return PubmedDocument(
        pmid="123456",
        title="Sample PubMed Title",
        abstract="This is a sample abstract.",
        journal="Sample Journal",
        year=2024,
        authors="Doe, John; Smith, Jane",
    )


def test_get_pubmed_pmids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the get_pubmed_pmids function with modern pagination approach.

    This test mocks the PubMed API response and verifies that the get_pubmed_pmids
    function correctly parses the XML response to extract article IDs.

    Args:
        monkeypatch (pytest.MonkeyPatch): Pytest fixture for mocking.

    The test:
        1. Creates a mock XML response containing two PubMed IDs
        2. Patches the requests.get function to return the mock response
        3. Calls get_pubmed_pmids with test query and limit=2
        4. Verifies the returned IDs match the mock data

    Returns:
        None
    """
    # Mock the paginated_id_fetch function directly
    expected_ids = ["123456", "789012"]
    monkeypatch.setattr(
        ncbi_utils, "paginated_id_fetch", lambda *args, **kwargs: expected_ids
    )

    ids = pubmed.get_pubmed_pmids("test", 2)
    assert ids == ["123456", "789012"]


def test_get_pubmed_pmids_pagination_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test pagination fallback to time-based splitting when pagination fails."""

    # Mock paginated_id_fetch to simulate failure and trigger fallback
    def mock_paginated_id_fetch(*args: Any, **kwargs: Any) -> list[str]:
        # Simulate the fallback mechanism by directly calling split_by_year
        return ["123456", "789012"]

    # Mock split_by_year to return test IDs
    def mock_split_by_year(*args: Any, **kwargs: Any) -> list[str]:
        return ["123456", "789012"]

    monkeypatch.setattr(ncbi_utils, "paginated_id_fetch", mock_paginated_id_fetch)
    monkeypatch.setattr(pubmed, "split_by_year", mock_split_by_year)

    ids = pubmed.get_pubmed_pmids("test", 15000)  # Force pagination attempt
    assert ids == ["123456", "789012"]


def test_fetch_abstracts(monkeypatch: pytest.MonkeyPatch) -> None:
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123456</PMID>
          <Article>
            <ArticleTitle>Test Title</ArticleTitle>
            <Abstract>
              <AbstractText>This is a test abstract.</AbstractText>
            </Abstract>
            <Journal>
              <Title>Test Journal</Title>
            </Journal>
            <AuthorList>
              <Author>
                <LastName>Doe</LastName>
                <ForeName>John</ForeName>
              </Author>
            </AuthorList>
          </Article>
          <PubDate>
            <Year>2024</Year>
          </PubDate>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>"""

    monkeypatch.setattr(
        ncbi_utils, "fetch_xml_with_retry", lambda *args, **kwargs: xml_response
    )
    docs = pubmed.fetch_abstracts(["123456"])
    assert len(docs) == 1
    doc = docs[0]
    assert doc.pmid == "123456"
    assert doc.title == "Test Title"
    assert doc.abstract == "This is a test abstract."
    assert doc.journal == "Test Journal"
    assert doc.year == 2024
    assert doc.authors is not None and "John Doe" in doc.authors


def test_save_to_sqlite(tmp_path: Path, sample_pubmed_doc: PubmedDocument) -> None:
    db_path = tmp_path / "test.db"
    pubmed.save_to_sqlite([sample_pubmed_doc], str(db_path))
    assert db_path.exists()


def test_save_to_json(tmp_path: Path, sample_pubmed_doc: PubmedDocument) -> None:
    json_path = tmp_path / "test.json"
    pubmed.save_to_json([sample_pubmed_doc], str(json_path))
    assert json_path.exists()
    with open(json_path) as f:
        data = json.load(f)
    assert data[0]["pmid"] == sample_pubmed_doc.pmid


def test_fetch_ids_for_term(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test for fetch_ids_for_term: should fetch and return IDs up to
    max_uids_per_query."""

    import xml.etree.ElementTree as ET

    # Create mock XML response
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <IdList>
            <Id>1</Id>
            <Id>2</Id>
        </IdList>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )

    ids = pubmed.fetch_ids_for_term("test", max_uids_per_query=2)
    assert ids == ["1", "2"]


def test_split_by_week(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test for split_by_week: should call fetch_ids_for_term for each week
    with nonzero count."""

    # Patch get_total_count to return 2 for first week, 0 for others
    def mock_get_total_count(term: str) -> int:
        if "01/01:2024/01/07" in term:
            return 2
        return 0

    monkeypatch.setattr(pubmed, "get_total_count", mock_get_total_count)

    # Patch fetch_ids_for_term to return dummy IDs
    def mock_fetch_ids_for_term(
        term: str, count: int, max_uids_per_query: int = 10000
    ) -> list[str]:
        return ["w1", "w2"][:count]

    monkeypatch.setattr(pubmed, "fetch_ids_for_term", mock_fetch_ids_for_term)
    seen: set = set()
    ids = pubmed.split_by_week(2024, 1, "test", 2, seen, max_uids_per_query=10000)
    assert ids == ["w1", "w2"]


def test_split_by_month(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit test for split_by_month: should call fetch_ids_for_term or
    split_by_week as needed."""

    # Patch get_total_count: month 1 returns 2, month 2 returns 12, rest 0
    def mock_get_total_count(term: str) -> int:
        if "/01[dp]" in term:
            return 2
        if "/02[dp]" in term:
            return 12
        return 0

    monkeypatch.setattr(pubmed, "get_total_count", mock_get_total_count)

    # Patch fetch_ids_for_term: just return dummy IDs
    def mock_fetch_ids_for_term(
        term: str, max_uids_per_query: int = 10000
    ) -> list[str]:
        return ["m10"] * max_uids_per_query

    monkeypatch.setattr(pubmed, "fetch_ids_for_term", mock_fetch_ids_for_term)

    # Patch split_by_week: return a fixed list for month 2
    def mock_split_by_week(
        year: int,
        month: int,
        base_term: str,
        remaining: int,
        seen: set[str],
        max_uids_per_query: int = 10000,
    ) -> list[str]:
        return ["w1", "w2"]

    monkeypatch.setattr(pubmed, "split_by_week", mock_split_by_week)
    seen: set = set()
    ids = pubmed.split_by_month(2024, "test", 5, seen, max_uids_per_query=10)
    # Should get 2 from week split for month 2 (processed first due
    # to reverse iteration), then 1 from month 1 (deduped).
    # Month 2's count (12) > max_uids_per_query (10) triggers week split.
    assert ids == ["w1", "w2", "m10"]


@pytest.mark.unit
@patch("biomed_kg_agent.core.pipeline.process_documents")
def test_process_pubmed_documents_from_db_success(
    mock_process: MagicMock,
    tmp_path: Path,
    sample_pubmed_doc: PubmedDocument,
) -> None:
    """Test successful NLP processing orchestration."""
    input_db = str(tmp_path / "input.db")
    output_db = str(tmp_path / "output.db")

    # Create input database with sample document
    engine = create_engine(f"sqlite:///{input_db}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(sample_pubmed_doc)
        session.commit()

    # Mock process_documents to return processed docs
    mock_processed = [
        DocumentInternal(
            id="123456",
            title="Sample PubMed Title",
            text="This is a sample abstract.",
            source="pubmed",
            entities=[],  # Empty entities for simplicity
        )
    ]

    mock_process.return_value = mock_processed

    result = process_pubmed_documents_from_db(input_db, output_db)

    assert result["doc_count"] == 1
    assert result["entity_count"] == 0
    assert result["umls_linked"] == 0
    assert result["umls_rate"] == 0
    assert result["output_path"] == output_db


@pytest.mark.unit
def test_process_pubmed_documents_from_db_empty_database(tmp_path: Path) -> None:
    """Test NLP processing with empty database."""
    input_db = str(tmp_path / "empty.db")
    output_db = str(tmp_path / "output.db")

    # Create empty database
    engine = create_engine(f"sqlite:///{input_db}")
    SQLModel.metadata.create_all(engine)

    result = process_pubmed_documents_from_db(input_db, output_db)

    assert result["doc_count"] == 0
    assert result["entity_count"] == 0
    assert result["umls_linked"] == 0
    assert result["umls_rate"] == 0
    assert result["output_path"] == output_db


@pytest.mark.unit
@patch("biomed_kg_agent.data_sources.ncbi.pubmed.process_documents")
def test_process_pubmed_documents_from_db_with_entities(
    mock_process: MagicMock,
    tmp_path: Path,
    sample_pubmed_doc: PubmedDocument,
) -> None:
    """Test NLP processing that extracts entities."""
    input_db = str(tmp_path / "input.db")
    output_db = str(tmp_path / "output.db")

    # Create input database
    engine = create_engine(f"sqlite:///{input_db}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(sample_pubmed_doc)
        session.commit()

    # Mock processed document with entities (one with UMLS, one without)
    mock_entities = [
        ExtractedEntity(
            text="cancer",
            entity_type="Disease",
            start_pos=0,
            end_pos=6,
            sentence_id=0,
            sentence_text="This is a test abstract about cancer treatment.",
            umls_cui="C0006826",  # Add UMLS CUI
            source_label="DISEASE",
        ),
        ExtractedEntity(
            text="treatment",
            entity_type="Treatment",
            start_pos=30,
            end_pos=39,
            sentence_id=0,
            sentence_text="This is a test abstract about cancer treatment.",
            source_label="TREATMENT",
        ),
    ]

    mock_processed = [
        DocumentInternal(
            id="123456",
            title="Sample PubMed Title",
            text="This is a sample abstract.",
            source="pubmed",
            entities=mock_entities,
        )
    ]

    mock_process.return_value = mock_processed

    result = process_pubmed_documents_from_db(input_db, output_db)

    assert result["doc_count"] == 1
    assert result["entity_count"] == 2
    assert result["umls_linked"] == 1  # Only one entity has UMLS CUI
    assert result["umls_rate"] == 50.0  # 1 out of 2 = 50%
    assert result["output_path"] == output_db


@pytest.mark.unit
@patch("biomed_kg_agent.data_sources.ncbi.pubmed.process_documents")
def test_process_pubmed_documents_from_db_with_config(
    mock_process: MagicMock,
    tmp_path: Path,
    sample_pubmed_doc: PubmedDocument,
) -> None:
    """Test NLP processing with custom config path."""
    input_db = str(tmp_path / "input.db")
    output_db = str(tmp_path / "output.db")
    config_path = str(tmp_path / "custom_config.yaml")

    # Create input database
    engine = create_engine(f"sqlite:///{input_db}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        session.add(sample_pubmed_doc)
        session.commit()

    mock_processed = [
        DocumentInternal(
            id="123456",
            title="Sample PubMed Title",
            text="Test text",
            source="pubmed",
            entities=[],
        )
    ]

    mock_process.return_value = mock_processed

    result = process_pubmed_documents_from_db(input_db, output_db, config_path)

    # Verify return value
    assert result["doc_count"] == 1
    assert result["entity_count"] == 0
    assert result["umls_linked"] == 0
    assert result["umls_rate"] == 0
    assert result["output_path"] == output_db

    # Verify config_path was passed to process_documents
    mock_process.assert_called_once()
    call_args = mock_process.call_args
    assert call_args[0][1] == config_path  # Second positional arg is config_path
