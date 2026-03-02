from typing import Any, Callable

import pytest

from biomed_kg_agent.data_sources.ncbi import ncbi_utils, pubmed
from tests.mocks.ncbi import MockResp


def make_mock_get(xml_response: str) -> Callable[..., MockResp]:
    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        return MockResp(xml_response)

    return mock_get


def test_fetch_book_article_abstracts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching and parsing book articles from PubMed."""
    # Sample XML based on an actual PubmedBookArticle response
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <PubmedArticleSet>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">37816105</PMID>
          <ArticleTitle>ExWAS Approach to Cancer Risk</ArticleTitle>
          <Book>
            <BookTitle>Methods in Molecular Biology</BookTitle>
            <PubDate>
              <Year>2023</Year>
            </PubDate>
          </Book>
          <Abstract>
            <AbstractText>
              This chapter describes the exposome-wide association study (ExWAS)
              approach for identifying environmental factors associated with cancer
              risk. Environmental exposures play a critical role in cancer etiology,
              and the ExWAS approach allows for comprehensive assessment of multiple
              exposures and their impact on cancer outcomes.
            </AbstractText>
          </Abstract>
          <AuthorList>
            <Author>
              <LastName>Smith</LastName>
              <ForeName>John</ForeName>
            </Author>
            <Author>
              <LastName>Jones</LastName>
              <ForeName>Jane</ForeName>
            </Author>
          </AuthorList>
        </BookDocument>
      </PubmedBookArticle>
    </PubmedArticleSet>"""

    monkeypatch.setattr(ncbi_utils.requests, "get", make_mock_get(xml_response))

    # Fetch the book article
    docs = pubmed.fetch_abstracts(["37816105"])

    # Verify results
    assert len(docs) == 1
    doc = docs[0]
    assert doc.pmid == "37816105"
    assert doc.title == "ExWAS Approach to Cancer Risk"
    assert "ExWAS" in (doc.abstract or "")
    assert doc.document_type == "BookChapter"
    assert (
        doc.journal == "Methods in Molecular Biology"
    )  # Book title is used as journal
    assert doc.year == 2023
    assert "John Smith" in (doc.authors or "")
    assert "Jane Jones" in (doc.authors or "")


def test_parse_pubmed_xml_with_mixed_content() -> None:
    """Test parsing XML with both regular articles and book articles."""
    # Sample XML with both article types
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345678</PMID>
          <Article>
            <ArticleTitle>Regular Article Title</ArticleTitle>
            <Abstract>
              <AbstractText>Regular article abstract.</AbstractText>
            </Abstract>
            <Journal>
              <Title>Journal of Testing</Title>
            </Journal>
            <AuthorList>
              <Author>
                <LastName>Regular</LastName>
                <ForeName>Author</ForeName>
              </Author>
            </AuthorList>
          </Article>
          <PubDate>
            <Year>2022</Year>
          </PubDate>
        </MedlineCitation>
      </PubmedArticle>
      <PubmedBookArticle>
        <BookDocument>
          <PMID>87654321</PMID>
          <ArticleTitle>Book Chapter Title</ArticleTitle>
          <Book>
            <BookTitle>Book of Tests</BookTitle>
            <PubDate>
              <Year>2023</Year>
            </PubDate>
          </Book>
          <Abstract>
            <AbstractText>Book chapter abstract.</AbstractText>
          </Abstract>
          <AuthorList>
            <Author>
              <LastName>Book</LastName>
              <ForeName>Author</ForeName>
            </Author>
          </AuthorList>
        </BookDocument>
      </PubmedBookArticle>
    </PubmedArticleSet>"""

    # Parse the XML directly
    docs = pubmed._parse_pubmed_xml(xml_response)

    # Verify results
    assert len(docs) == 2

    # Check regular article
    article = next((doc for doc in docs if doc.pmid == "12345678"), None)
    assert article is not None
    assert article.document_type == "Article"
    assert article.title == "Regular Article Title"

    # Check book article
    book = next((doc for doc in docs if doc.pmid == "87654321"), None)
    assert book is not None
    assert book.document_type == "BookChapter"
    assert book.title == "Book Chapter Title"
    assert book.journal == "Book of Tests"


def test_unknown_document_type() -> None:
    """Test handling of unknown document types in PubMed XML."""
    # Create XML with a made-up document type
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <PubmedArticleSet>
      <UnknownArticleType>
        <PMID>99999999</PMID>
        <ArticleTitle>Unknown Type Document</ArticleTitle>
      </UnknownArticleType>
    </PubmedArticleSet>"""

    # Parse the XML directly to test handling of unknown types
    docs = pubmed._parse_pubmed_xml(xml_response)

    # There should be no documents (since our parser looks for specific types)
    assert len(docs) == 0
