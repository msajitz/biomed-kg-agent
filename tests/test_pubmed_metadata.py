"""
Tests for the new metadata extraction features in PubMed parsing.

This file tests the extraction of MeSH terms, DOI, and keywords from PubMed XML.
"""

import xml.etree.ElementTree as ET
from typing import Any, Callable

import pytest

from biomed_kg_agent.data_sources.ncbi import ncbi_utils, pubmed
from biomed_kg_agent.data_sources.ncbi.pubmed import (
    _get_doi,
    _get_keywords,
    _get_mesh_terms,
)
from tests.mocks.ncbi import MockResp


def make_mock_get(xml_response: str) -> Callable[..., MockResp]:
    def mock_get(*args: Any, **kwargs: Any) -> MockResp:
        return MockResp(xml_response)

    return mock_get


def test_get_mesh_terms_article() -> None:
    """Test extracting MeSH terms from regular articles."""
    # Test article with MeSH terms
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>123456</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                </Article>
                <MeshHeadingList>
                    <MeshHeading>
                        <DescriptorName>Diabetes Mellitus</DescriptorName>
                    </MeshHeading>
                    <MeshHeading>
                        <DescriptorName>Insulin</DescriptorName>
                    </MeshHeading>
                    <MeshHeading>
                        <DescriptorName>Blood Glucose</DescriptorName>
                    </MeshHeading>
                </MeshHeadingList>
            </MedlineCitation>
        </PubmedArticle>
        """
    )
    mesh_terms = _get_mesh_terms(article_xml, "Article")
    assert mesh_terms == "Diabetes Mellitus, Insulin, Blood Glucose"

    # Test article with no MeSH terms
    article_xml_no_mesh = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>123456</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
    )
    mesh_terms = _get_mesh_terms(article_xml_no_mesh, "Article")
    assert mesh_terms == ""


def test_get_doi_article() -> None:
    """Test extracting DOI from regular articles."""
    # Test article with DOI
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>123456</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                </Article>
            </MedlineCitation>
            <PubmedData>
                <ArticleIdList>
                    <ArticleId IdType="pubmed">123456</ArticleId>
                    <ArticleId IdType="doi">10.1038/nature12345</ArticleId>
                    <ArticleId IdType="pmc">PMC123456</ArticleId>
                </ArticleIdList>
            </PubmedData>
        </PubmedArticle>
        """
    )
    doi = _get_doi(article_xml, "Article")
    assert doi == "10.1038/nature12345"

    # Test article with no DOI
    article_xml_no_doi = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>123456</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                </Article>
            </MedlineCitation>
            <PubmedData>
                <ArticleIdList>
                    <ArticleId IdType="pubmed">123456</ArticleId>
                    <ArticleId IdType="pmc">PMC123456</ArticleId>
                </ArticleIdList>
            </PubmedData>
        </PubmedArticle>
        """
    )
    doi = _get_doi(article_xml_no_doi, "Article")
    assert doi == ""


def test_get_keywords_article() -> None:
    """Test extracting keywords from regular articles."""
    # Test article with keywords
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>123456</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                </Article>
                <KeywordList>
                    <Keyword>cancer metabolism</Keyword>
                    <Keyword>glucose</Keyword>
                    <Keyword>mitochondria</Keyword>
                    <Keyword>biomarkers</Keyword>
                </KeywordList>
            </MedlineCitation>
        </PubmedArticle>
        """
    )
    keywords = _get_keywords(article_xml, "Article")
    assert keywords == "cancer metabolism, glucose, mitochondria, biomarkers"

    # Test article with no keywords
    article_xml_no_keywords = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <PMID>123456</PMID>
                <Article>
                    <ArticleTitle>Test Article</ArticleTitle>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
        """
    )
    keywords = _get_keywords(article_xml_no_keywords, "Article")
    assert keywords == ""


def test_fetch_abstracts_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching and parsing articles with full metadata.

    Includes MeSH terms, DOI, and keywords extraction.
    """
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>123456</PMID>
          <Article>
            <ArticleTitle>Cancer Metabolism and Glucose Utilization</ArticleTitle>
            <Abstract>
              <AbstractText>This study investigates cancer metabolism.</AbstractText>
            </Abstract>
            <Journal>
              <Title>Nature Medicine</Title>
            </Journal>
            <AuthorList>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>John</ForeName>
              </Author>
              <Author>
                <LastName>Johnson</LastName>
                <ForeName>Jane</ForeName>
              </Author>
            </AuthorList>
          </Article>
          <MeshHeadingList>
            <MeshHeading>
              <DescriptorName>Neoplasms</DescriptorName>
            </MeshHeading>
            <MeshHeading>
              <DescriptorName>Glucose</DescriptorName>
            </MeshHeading>
            <MeshHeading>
              <DescriptorName>Cell Metabolism</DescriptorName>
            </MeshHeading>
          </MeshHeadingList>
          <KeywordList>
            <Keyword>cancer metabolism</Keyword>
            <Keyword>glucose utilization</Keyword>
            <Keyword>metabolomics</Keyword>
          </KeywordList>
          <PubDate>
            <Year>2024</Year>
          </PubDate>
        </MedlineCitation>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="pubmed">123456</ArticleId>
            <ArticleId IdType="doi">10.1038/nm.2024.123456</ArticleId>
            <ArticleId IdType="pmc">PMC9876543</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedArticle>
    </PubmedArticleSet>"""

    monkeypatch.setattr(ncbi_utils.requests, "get", make_mock_get(xml_response))

    # Fetch the article
    docs = pubmed.fetch_abstracts(["123456"])

    # Verify results
    assert len(docs) == 1
    doc = docs[0]

    # Test basic fields
    assert doc.pmid == "123456"
    assert doc.title == "Cancer Metabolism and Glucose Utilization"
    assert "cancer metabolism" in (doc.abstract or "")
    assert doc.journal == "Nature Medicine"
    assert doc.year == 2024
    assert "John Smith" in (doc.authors or "")
    assert "Jane Johnson" in (doc.authors or "")

    # Test new metadata fields
    assert doc.mesh_terms == "Neoplasms, Glucose, Cell Metabolism"
    assert doc.doi == "10.1038/nm.2024.123456"
    assert doc.keywords == "cancer metabolism, glucose utilization, metabolomics"


def test_fetch_book_article_with_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test fetching and parsing book articles with metadata."""
    xml_response = """<?xml version="1.0" encoding="UTF-8" ?>
    <PubmedArticleSet>
      <PubmedBookArticle>
        <BookDocument>
          <PMID Version="1">987654</PMID>
          <ArticleTitle>Metabolomics in Cancer Research</ArticleTitle>
          <Book>
            <BookTitle>Methods in Molecular Biology</BookTitle>
            <PubDate>
              <Year>2023</Year>
            </PubDate>
          </Book>
          <Abstract>
            <AbstractText>
              This chapter describes metabolomics techniques for cancer research.
            </AbstractText>
          </Abstract>
          <AuthorList>
            <Author>
              <LastName>Brown</LastName>
              <ForeName>Alice</ForeName>
            </Author>
          </AuthorList>
          <MeshHeadingList>
            <MeshHeading>
              <DescriptorName>Metabolomics</DescriptorName>
            </MeshHeading>
            <MeshHeading>
              <DescriptorName>Neoplasms</DescriptorName>
            </MeshHeading>
          </MeshHeadingList>
          <KeywordList>
            <Keyword>metabolomics</Keyword>
            <Keyword>mass spec</Keyword>
            <Keyword>biomarkers</Keyword>
          </KeywordList>
        </BookDocument>
        <PubmedData>
          <ArticleIdList>
            <ArticleId IdType="pubmed">987654</ArticleId>
            <ArticleId IdType="doi">10.1007/978-1-4939-9236-2_15</ArticleId>
          </ArticleIdList>
        </PubmedData>
      </PubmedBookArticle>
    </PubmedArticleSet>"""

    monkeypatch.setattr(ncbi_utils.requests, "get", make_mock_get(xml_response))

    # Fetch the book article
    docs = pubmed.fetch_abstracts(["987654"])

    # Verify results
    assert len(docs) == 1
    doc = docs[0]

    # Test basic fields
    assert doc.pmid == "987654"
    assert doc.title == "Metabolomics in Cancer Research"
    assert doc.document_type == "BookChapter"
    assert doc.journal == "Methods in Molecular Biology"  # Book title used as journal
    assert doc.year == 2023
    assert "Alice Brown" in (doc.authors or "")

    # Test new metadata fields
    assert doc.mesh_terms == "Metabolomics, Neoplasms"
    assert doc.doi == "10.1007/978-1-4939-9236-2_15"
    assert doc.keywords == "metabolomics, mass spec, biomarkers"
