"""Tests for PMC (PubMed Central) module."""

from collections.abc import Callable
from typing import Any

import pytest

from biomed_kg_agent.data_sources.ncbi import ncbi_utils, pmc
from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument
from tests.mocks.ncbi import MockResp


def make_mock_get(xml_response: str) -> Callable[[Any, Any], MockResp]:
    """Create a mock `requests.get` function for testing.

    Args:
        xml_response (str): The XML response to return when the mock function is called.

    Returns:
        Callable[[Any, Any], MockResp]: A mock function that simulates `requests.get`.
    """

    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        return MockResp(xml_response)

    return mock_get


def test_get_pmc_total_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test getting total count of PMC articles."""
    import xml.etree.ElementTree as ET

    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <Count>12345</Count>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )
    count = pmc.get_pmc_total_count("cancer")
    assert count == 12345


def test_get_pmc_total_count_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PMC total count with API error."""

    def mock_make_ncbi_request(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("NCBI API error: Database is not supported: pmc")

    monkeypatch.setattr(ncbi_utils, "make_ncbi_request", mock_make_ncbi_request)

    with pytest.raises(RuntimeError, match="NCBI API error"):
        pmc.get_pmc_total_count("cancer")


def test_fetch_pmc_ids_for_term(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching PMC IDs for a search term."""
    import xml.etree.ElementTree as ET

    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <IdList>
            <Id>PMC123456</Id>
            <Id>PMC789012</Id>
        </IdList>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )
    ids = pmc.fetch_pmc_ids_for_term("cancer metabolism", 10)
    assert ids == ["PMC123456", "PMC789012"]


def test_get_pmc_pmids_small_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test getting PMC PMIDs with small limit (direct request)."""
    expected_ids = ["PMC123456", "PMC789012"]

    monkeypatch.setattr(pmc, "paginated_id_fetch", lambda *args, **kwargs: expected_ids)

    ids = pmc.get_pmc_pmids("cancer", 2)
    assert ids == expected_ids


def test_get_pmc_pmids_large_limit_pagination(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test getting PMC PMIDs with large limit (pagination)."""
    expected_ids = ["PMC123456", "PMC789012"]

    monkeypatch.setattr(pmc, "paginated_id_fetch", lambda *args, **kwargs: expected_ids)

    ids = pmc.get_pmc_pmids("cancer", 15000)
    assert len(ids) >= 2
    assert "PMC123456" in ids
    assert "PMC789012" in ids


def test_parse_pmc_xml() -> None:
    """Test parsing PMC XML content."""
    xml_content = """<?xml version="1.0" ?>
    <pmc-articleset>
        <article>
            <front>
                <journal-meta>
                    <journal-title-group>
                        <journal-title>Test Journal</journal-title>
                    </journal-title-group>
                </journal-meta>
                <article-meta>
                    <article-id pub-id-type="pmid">123456</article-id>
                    <article-id pub-id-type="pmc">PMC123456</article-id>
                    <title-group>
                        <article-title>Test Article Title</article-title>
                    </title-group>
                    <contrib-group>
                        <contrib contrib-type="author">
                            <name>
                                <surname>Doe</surname>
                                <given-names>John</given-names>
                            </name>
                        </contrib>
                    </contrib-group>
                    <pub-date>
                        <year>2025</year>
                    </pub-date>
                    <abstract>
                        <p>This is a test abstract for the PMC article.</p>
                    </abstract>
                </article-meta>
            </front>
            <body>
                <sec>
                    <p>This is the full text content of the article.</p>
                </sec>
            </body>
        </article>
    </pmc-articleset>"""

    docs = pmc._parse_pmc_xml(xml_content)
    assert len(docs) == 1

    doc = docs[0]
    assert doc.pmid == "123456"
    assert doc.title == "Test Article Title"
    assert "This is a test abstract" in doc.abstract
    assert "This is the full text content" in doc.abstract
    assert doc.journal == "Test Journal"
    assert doc.year == 2025
    assert doc.authors == ["John Doe"]


def test_parse_pmc_xml_minimal() -> None:
    """Test parsing PMC XML with minimal required fields."""
    xml_content = """<?xml version="1.0" ?>
    <pmc-articleset>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmc">PMC123456</article-id>
                    <title-group>
                        <article-title>Minimal Test Article</article-title>
                    </title-group>
                </article-meta>
            </front>
        </article>
    </pmc-articleset>"""

    docs = pmc._parse_pmc_xml(xml_content)
    assert len(docs) == 1

    doc = docs[0]
    assert doc.pmid == "PMC123456"
    assert doc.title == "Minimal Test Article"
    assert doc.journal == "Unknown journal"
    assert doc.year is None
    assert doc.authors == []


def test_parse_pmc_xml_invalid() -> None:
    """Test parsing invalid PMC XML."""
    xml_content = "invalid xml content"

    docs = pmc._parse_pmc_xml(xml_content)
    assert len(docs) == 0


def test_fetch_pmc_articles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching PMC articles."""
    xml_response = """<?xml version="1.0" ?>
    <pmc-articleset>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmid">123456</article-id>
                    <title-group>
                        <article-title>Test PMC Article</article-title>
                    </title-group>
                </article-meta>
            </front>
        </article>
    </pmc-articleset>"""

    monkeypatch.setattr(
        pmc, "fetch_xml_with_retry", lambda *args, **kwargs: xml_response
    )

    articles = pmc.fetch_pmc_articles(["PMC123456"])
    assert len(articles) == 1
    assert articles[0].pmid == "123456"
    assert articles[0].title == "Test PMC Article"


def test_fetch_pmc_articles_with_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PMC articles fetching with retry logic."""
    xml_response = """<?xml version="1.0" ?>
    <pmc-articleset>
        <article>
            <front>
                <article-meta>
                    <article-id pub-id-type="pmid">123456</article-id>
                    <title-group>
                        <article-title>Test PMC Article</article-title>
                    </title-group>
                </article-meta>
            </front>
        </article>
    </pmc-articleset>"""

    call_count = [0]

    def mock_fetch_xml_with_retry(*args: Any, **kwargs: Any) -> str:
        call_count[0] += 1
        # Always return the successful response, we just want to test that the function gets called
        return xml_response

    monkeypatch.setattr(pmc, "fetch_xml_with_retry", mock_fetch_xml_with_retry)

    articles = pmc.fetch_pmc_articles(["PMC123456"])
    assert len(articles) == 1
    assert call_count[0] == 1  # Verify function was called


def test_get_articles_from_working_source(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test getting articles from working source (PMC fallback)."""

    # Mock successful PMC response
    def mock_get_pmc_pmids(term: str, limit: int) -> list[str]:
        return ["PMC123456", "PMC789012"]

    def mock_fetch_pmc_articles(pmc_ids: list[str]) -> list[PubmedDocument]:
        return [
            PubmedDocument(
                pmid="123456",
                title="Test PMC Article",
                abstract="Test abstract from PMC",
                journal="Test Journal",
                year=2025,
            )
        ]

    monkeypatch.setattr(pmc, "get_pmc_pmids", mock_get_pmc_pmids)
    monkeypatch.setattr(pmc, "fetch_pmc_articles", mock_fetch_pmc_articles)

    docs = pmc.get_articles_from_working_source("cancer", 2)
    assert len(docs) == 1
    assert docs[0].pmid == "123456"
    assert docs[0].title == "Test PMC Article"


def test_get_articles_from_working_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test working source fallback when PMC fails."""

    def mock_get_pmc_pmids(term: str, limit: int) -> None:
        raise Exception("PMC API error")

    monkeypatch.setattr(pmc, "get_pmc_pmids", mock_get_pmc_pmids)

    docs = pmc.get_articles_from_working_source("cancer", 2)
    assert len(docs) == 0


def test_extract_article_id() -> None:
    """Test article ID extraction from different sources."""
    import xml.etree.ElementTree as ET

    # Test with PMID
    xml_with_pmid = """
    <article-meta>
        <article-id pub-id-type="pmid">123456</article-id>
        <article-id pub-id-type="pmc">PMC123456</article-id>
    </article-meta>
    """
    elem = ET.fromstring(xml_with_pmid)
    article_id = pmc._extract_article_id(elem)
    assert article_id == "123456"

    # Test with only PMC ID
    xml_with_pmc = """
    <article-meta>
        <article-id pub-id-type="pmc">PMC123456</article-id>
    </article-meta>
    """
    elem = ET.fromstring(xml_with_pmc)
    article_id = pmc._extract_article_id(elem)
    assert article_id == "PMC123456"

    # Test with no ID
    xml_no_id = "<article-meta></article-meta>"
    elem = ET.fromstring(xml_no_id)
    article_id = pmc._extract_article_id(elem)
    assert article_id == ""


def test_extract_authors() -> None:
    """Test author extraction from PMC XML."""
    import xml.etree.ElementTree as ET

    xml_with_authors = """
    <article-meta>
        <contrib-group>
            <contrib contrib-type="author">
                <name>
                    <surname>Doe</surname>
                    <given-names>John</given-names>
                </name>
            </contrib>
            <contrib contrib-type="author">
                <name>
                    <surname>Smith</surname>
                    <given-names>Jane</given-names>
                </name>
            </contrib>
        </contrib-group>
    </article-meta>
    """

    elem = ET.fromstring(xml_with_authors)
    authors = pmc._extract_authors(elem)
    assert authors == ["John Doe", "Jane Smith"]

    # Test with no authors
    xml_no_authors = "<article-meta></article-meta>"
    elem = ET.fromstring(xml_no_authors)
    authors = pmc._extract_authors(elem)
    assert authors == []


def test_extract_year() -> None:
    """Test year extraction from PMC XML."""
    import xml.etree.ElementTree as ET

    xml_with_year = """
    <article-meta>
        <pub-date>
            <year>2025</year>
        </pub-date>
    </article-meta>
    """

    elem = ET.fromstring(xml_with_year)
    year = pmc._extract_year(elem)
    assert year == 2025

    # Test with invalid year
    xml_invalid_year = """
    <article-meta>
        <pub-date>
            <year>invalid</year>
        </pub-date>
    </article-meta>
    """

    elem = ET.fromstring(xml_invalid_year)
    year = pmc._extract_year(elem)
    assert year is None
