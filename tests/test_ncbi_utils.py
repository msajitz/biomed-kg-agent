"""Tests for ncbi_utils module."""

import xml.etree.ElementTree as ET
from typing import Any

import pytest
import requests  # type: ignore

from biomed_kg_agent.data_sources.ncbi import ncbi_utils
from tests.mocks.ncbi import MockResp


def test_make_ncbi_request_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful NCBI API request."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <Count>5</Count>
    </eSearchResult>"""

    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        return MockResp(xml_response)

    monkeypatch.setattr(ncbi_utils.requests, "get", mock_get)

    result = ncbi_utils.make_ncbi_request(
        "esearch.fcgi", {"db": "pubmed", "term": "test"}
    )
    assert result is not None
    assert result.tag == "eSearchResult"
    count_elem = result.find(".//Count")
    assert count_elem is not None
    assert count_elem.text == "5"


def test_make_ncbi_request_api_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NCBI API request with API error."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <ERROR>Database is not supported: invalid</ERROR>
    </eSearchResult>"""

    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        return MockResp(xml_response)

    monkeypatch.setattr(ncbi_utils.requests, "get", mock_get)

    with pytest.raises(RuntimeError, match="NCBI API error"):
        ncbi_utils.make_ncbi_request("esearch.fcgi", {"db": "invalid", "term": "test"})


def test_make_ncbi_request_network_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test NCBI API request with network error."""

    def mock_get(*args: Any, **kwargs: Any) -> None:
        raise requests.RequestException("Network error")

    monkeypatch.setattr(ncbi_utils.requests, "get", mock_get)

    with pytest.raises(RuntimeError, match="NCBI request failed"):
        ncbi_utils.make_ncbi_request("esearch.fcgi", {"db": "pubmed", "term": "test"})


def test_get_total_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test getting total count."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <Count>12345</Count>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )

    count = ncbi_utils.get_total_count("pubmed", "cancer")
    assert count == 12345


def test_get_total_count_no_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test getting total count when Count element is missing."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )

    count = ncbi_utils.get_total_count("pubmed", "nonexistent")
    assert count == 0


def test_fetch_ids_for_term(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching IDs for a search term."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <IdList>
            <Id>123456</Id>
            <Id>789012</Id>
            <Id>345678</Id>
        </IdList>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )

    ids = ncbi_utils.fetch_ids_for_term("pubmed", "cancer", max_uids=10)
    assert ids == ["123456", "789012", "345678"]


def test_fetch_ids_for_term_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching IDs when no results found."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <IdList>
        </IdList>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )

    ids = ncbi_utils.fetch_ids_for_term("pubmed", "nonexistent")
    assert ids == []


def test_paginated_id_fetch_small_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test paginated fetch with small limit (direct request)."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <Count>2</Count>
        <IdList>
            <Id>123456</Id>
            <Id>789012</Id>
        </IdList>
    </eSearchResult>"""

    mock_root = ET.fromstring(xml_response)
    monkeypatch.setattr(
        ncbi_utils, "make_ncbi_request", lambda *args, **kwargs: mock_root
    )

    ids = ncbi_utils.paginated_id_fetch("pubmed", "test", 2)
    assert ids == ["123456", "789012"]


def test_paginated_id_fetch_large_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that large limit with large total_count returns empty (needs time splitting)."""
    count_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <Count>15000</Count>
    </eSearchResult>"""

    def mock_make_ncbi_request(*args: Any, **kwargs: Any) -> ET.Element:
        # Only the count check happens; no ID fetching should occur
        return ET.fromstring(count_response)

    monkeypatch.setattr(ncbi_utils, "make_ncbi_request", mock_make_ncbi_request)

    # When both limit and total_count exceed MAX_UIDS_PER_QUERY (9999),
    # we need to fetch more than the cap allows, so return empty
    ids = ncbi_utils.paginated_id_fetch("pubmed", "test", 15000)
    assert len(ids) == 0


def test_paginated_id_fetch_large_limit_small_corpus(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that large limit with small total_count still works."""
    count_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <Count>5000</Count>
    </eSearchResult>"""

    ids_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <IdList>
            <Id>111111</Id>
            <Id>222222</Id>
        </IdList>
    </eSearchResult>"""

    call_count = [0]

    def mock_make_ncbi_request(*args: Any, **kwargs: Any) -> ET.Element:
        call_count[0] += 1
        if call_count[0] == 1:  # First call: get total count
            return ET.fromstring(count_response)
        else:  # Subsequent calls: get IDs
            return ET.fromstring(ids_response)

    monkeypatch.setattr(ncbi_utils, "make_ncbi_request", mock_make_ncbi_request)

    # Even though limit=15000 exceeds cap, total_count=5000 is within cap,
    # so to_fetch=min(15000, 5000)=5000 which is valid
    ids = ncbi_utils.paginated_id_fetch("pubmed", "test", 15000)
    assert len(ids) == 2
    assert "111111" in ids
    assert "222222" in ids


def test_paginated_id_fetch_large_corpus_small_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that large corpus (>9999) still returns IDs when limit is small."""
    count_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <Count>15000</Count>
    </eSearchResult>"""

    ids_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <eSearchResult>
        <IdList>
            <Id>123456</Id>
            <Id>789012</Id>
        </IdList>
    </eSearchResult>"""

    call_count = [0]

    def mock_make_ncbi_request(*args: Any, **kwargs: Any) -> ET.Element:
        call_count[0] += 1
        if call_count[0] == 1:  # First call: get total count
            return ET.fromstring(count_response)
        else:  # Subsequent calls: get IDs
            return ET.fromstring(ids_response)

    monkeypatch.setattr(ncbi_utils, "make_ncbi_request", mock_make_ncbi_request)

    # Even though total_count exceeds MAX_UIDS_PER_QUERY (9999),
    # we can still fetch a small sample (first N by relevance)
    ids = ncbi_utils.paginated_id_fetch("pubmed", "test", 100)
    assert len(ids) == 2
    assert "123456" in ids
    assert "789012" in ids


def test_fetch_xml_with_retry_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful XML fetch with retry."""
    xml_response = "<?xml version='1.0'?><root>test</root>"

    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        return MockResp(xml_response)

    monkeypatch.setattr(ncbi_utils.requests, "get", mock_get)

    result = ncbi_utils.fetch_xml_with_retry("pubmed", ["123", "456"])
    assert result == xml_response


def test_fetch_xml_with_retry_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test XML fetch with retry that ultimately fails."""

    def mock_get(*args: Any, **kwargs: Any) -> None:
        raise requests.RequestException("Network error")

    # Mock sleep to speed up test
    monkeypatch.setattr(ncbi_utils.time, "sleep", lambda x: None)
    monkeypatch.setattr(ncbi_utils.requests, "get", mock_get)

    result = ncbi_utils.fetch_xml_with_retry("pubmed", ["123", "456"])
    assert result == ""


def test_fetch_xml_with_retry_eventual_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test XML fetch that succeeds after retries."""

    xml_response = "<?xml version='1.0'?><root>test</root>"

    call_count = [0]

    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        call_count[0] += 1
        if call_count[0] < 2:  # Fail first attempt
            raise requests.RequestException("Network error")
        return MockResp(xml_response)

    # Mock sleep to speed up test
    monkeypatch.setattr(ncbi_utils.time, "sleep", lambda x: None)
    monkeypatch.setattr(ncbi_utils.requests, "get", mock_get)

    result = ncbi_utils.fetch_xml_with_retry("pubmed", ["123", "456"])
    assert result == xml_response
    assert call_count[0] == 2


def test_deduplicate_and_limit_ids() -> None:
    """Test ID deduplication and limiting."""
    seen: set[str] = {"123", "456"}
    ids = ["123", "789", "456", "101112", "789"]  # Contains duplicates and already seen

    result = ncbi_utils.deduplicate_and_limit_ids(ids, 10, seen)

    # Should get only new, unique IDs
    assert result == ["789", "101112"]
    # seen set should be updated
    assert "789" in seen
    assert "101112" in seen


def test_deduplicate_and_limit_ids_with_limit() -> None:
    """Test ID deduplication with strict limit."""
    seen: set[str] = set()
    ids = ["123", "456", "789", "101112", "131415"]

    result = ncbi_utils.deduplicate_and_limit_ids(ids, 3, seen)

    # Should get only first 3 unique IDs
    assert len(result) == 3
    assert result == ["123", "456", "789"]
    assert len(seen) == 3
