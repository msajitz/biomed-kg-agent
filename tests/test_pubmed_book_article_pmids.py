"""Tests for specific problematic book article PMIDs."""

from typing import Any, Callable

import pytest

from biomed_kg_agent.data_sources.ncbi import ncbi_utils, pubmed
from tests.mocks.ncbi import MockResp


def make_mock_get(xml_response: str) -> Callable[..., MockResp]:
    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        return MockResp(xml_response)

    return mock_get


def test_problematic_book_article_pmids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that previously problematic book article PMIDs are correctly processed.

    Specifically testing PMIDs that were causing issues:
    37816105, 25905162, 34097369, 33411450, 25905240, 33905196
    """
    # Using a simplified mock response for testing
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <PubmedArticleSet>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">37816105</PMID>
          <ArticleTitle>Book Article 1</ArticleTitle>
          <Book><BookTitle>Book Title 1</BookTitle></Book>
        </BookDocument>
      </PubmedBookArticle>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">25905162</PMID>
          <ArticleTitle>Book Article 2</ArticleTitle>
          <Book><BookTitle>Book Title 2</BookTitle></Book>
        </BookDocument>
      </PubmedBookArticle>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">34097369</PMID>
          <ArticleTitle>Book Article 3</ArticleTitle>
          <Book><BookTitle>Book Title 3</BookTitle></Book>
        </BookDocument>
      </PubmedBookArticle>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">33411450</PMID>
          <ArticleTitle>Book Article 4</ArticleTitle>
          <Book><BookTitle>Book Title 4</BookTitle></Book>
        </BookDocument>
      </PubmedBookArticle>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">25905240</PMID>
          <ArticleTitle>Book Article 5</ArticleTitle>
          <Book><BookTitle>Book Title 5</BookTitle></Book>
        </BookDocument>
      </PubmedBookArticle>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">33905196</PMID>
          <ArticleTitle>Book Article 6</ArticleTitle>
          <Book><BookTitle>Book Title 6</BookTitle></Book>
        </BookDocument>
      </PubmedBookArticle>
    </PubmedArticleSet>"""

    monkeypatch.setattr(ncbi_utils.requests, "get", make_mock_get(xml_response))

    # Test fetching all problematic PMIDs at once
    problematic_pmids = [
        "37816105",
        "25905162",
        "34097369",
        "33411450",
        "25905240",
        "33905196",
    ]
    docs = pubmed.fetch_abstracts(problematic_pmids)

    # Verify we got all 6 documents back
    assert len(docs) == 6

    # Verify each document has the correct document_type
    for doc in docs:
        assert doc.document_type == "BookChapter"

    # Verify all PMIDs are present
    retrieved_pmids = {doc.pmid for doc in docs}
    assert retrieved_pmids == set(problematic_pmids)
