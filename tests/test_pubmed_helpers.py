import xml.etree.ElementTree as ET

import pytest

from biomed_kg_agent.data_sources.ncbi import pubmed
from biomed_kg_agent.data_sources.ncbi.pubmed import _deduplicate_and_limit_ids


def test_deduplicate_and_limit_ids_empty_input() -> None:
    seen: set[str] = set()
    assert _deduplicate_and_limit_ids([], 5, seen) == []
    assert seen == set()


def test_deduplicate_and_limit_ids_no_duplicates() -> None:
    seen: set[str] = set()
    raw_ids = ["1", "2", "3"]
    assert _deduplicate_and_limit_ids(raw_ids, 5, seen) == ["1", "2", "3"]
    assert seen == {"1", "2", "3"}


def test_deduplicate_and_limit_ids_with_duplicates() -> None:
    seen: set[str] = set()
    raw_ids = ["1", "2", "1", "3", "2"]
    assert _deduplicate_and_limit_ids(raw_ids, 5, seen) == ["1", "2", "3"]
    assert seen == {"1", "2", "3"}


def test_deduplicate_and_limit_ids_limit_respected() -> None:
    seen: set[str] = set()
    raw_ids = ["1", "2", "3", "4", "5"]
    assert _deduplicate_and_limit_ids(raw_ids, 3, seen) == ["1", "2", "3"]
    assert seen == {"1", "2", "3"}


def test_deduplicate_and_limit_ids_pre_existing_seen() -> None:
    seen: set[str] = {"1", "2"}
    raw_ids = ["2", "3", "4"]
    assert _deduplicate_and_limit_ids(raw_ids, 5, seen) == ["3", "4"]
    assert seen == {"1", "2", "3", "4"}


def test_deduplicate_and_limit_ids_limit_with_pre_existing_seen() -> None:
    seen: set[str] = {"1"}
    raw_ids = ["2", "3", "4"]
    # Limit is 2, seen already has 1, so only 1 new ID should be added
    assert _deduplicate_and_limit_ids(raw_ids, 2, seen) == ["2"]
    assert seen == {"1", "2"}


def test_deduplicate_and_limit_ids_limit_already_met_by_seen() -> None:
    seen: set[str] = {"1", "2", "3"}
    raw_ids = ["4", "5"]
    assert _deduplicate_and_limit_ids(raw_ids, 3, seen) == []
    assert seen == {"1", "2", "3"}


def test_deduplicate_and_limit_ids_limit_zero() -> None:
    seen: set[str] = set()
    raw_ids = ["1", "2", "3"]
    assert _deduplicate_and_limit_ids(raw_ids, 0, seen) == []
    assert seen == set()


@pytest.fixture
def book_article_xml() -> ET.Element:
    """Create a sample BookDocument XML element for testing."""
    xml_string = """
    <PubmedBookArticle>
        <BookDocument>
            <PMID Version="1">37816105</PMID>
            <ArticleTitle>Test Book Chapter</ArticleTitle>
            <Book>
                <BookTitle>Test Book Title</BookTitle>
                <PubDate>
                    <Year>2023</Year>
                </PubDate>
                <AuthorList Type="authors">
                    <Author>
                        <LastName>Smith</LastName>
                        <ForeName>John</ForeName>
                    </Author>
                </AuthorList>
            </Book>
            <Abstract>
                <AbstractText>
                    This is a test abstract for the book chapter.
                </AbstractText>
            </Abstract>
            <AuthorList>
                <Author>
                    <LastName>Jones</LastName>
                    <ForeName>Jane</ForeName>
                </Author>
            </AuthorList>
        </BookDocument>
    </PubmedBookArticle>
    """
    return ET.fromstring(xml_string)


@pytest.fixture
def article_xml() -> ET.Element:
    """Create a sample PubmedArticle XML element for testing."""
    xml_string = """
    <PubmedArticle>
        <MedlineCitation>
            <PMID Version="1">12345678</PMID>
            <Article>
                <ArticleTitle>Test Journal Article</ArticleTitle>
                <Abstract>
                    <AbstractText>
                        This is a test abstract for the journal article.
                    </AbstractText>
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
    """
    return ET.fromstring(xml_string)


def test_get_document_type() -> None:
    """Test document type detection."""
    # Test for PubmedArticle
    article = ET.Element("PubmedArticle")
    doc_type, is_book = pubmed._get_document_type(article)
    assert doc_type == "Article"
    assert is_book is False

    # Test for PubmedBookArticle
    book_article = ET.Element("PubmedBookArticle")
    doc_type, is_book = pubmed._get_document_type(book_article)
    assert doc_type == "BookChapter"
    assert is_book is True

    # Test for Unknown type
    unknown = ET.Element("UnknownType")
    doc_type, is_book = pubmed._get_document_type(unknown)
    assert doc_type == "Unknown"
    assert is_book is False


def test_get_document_node(
    book_article_xml: ET.Element, article_xml: ET.Element
) -> None:
    """Test retrieving the appropriate document node."""
    # For book articles, should return BookDocument node
    _, is_book = pubmed._get_document_type(book_article_xml)
    doc_node = pubmed._get_document_node(book_article_xml, is_book)
    assert doc_node is not None
    assert doc_node.tag == "BookDocument"

    # For regular articles, should return the article itself
    _, is_book = pubmed._get_document_type(article_xml)
    doc_node = pubmed._get_document_node(article_xml, is_book)
    assert doc_node is not None
    assert doc_node == article_xml


def test_get_title(book_article_xml: ET.Element, article_xml: ET.Element) -> None:
    """Test title extraction from different document types."""
    # Get document nodes first
    _, is_book_article = pubmed._get_document_type(book_article_xml)
    book_doc_node = pubmed._get_document_node(book_article_xml, is_book_article)
    assert book_doc_node is not None
    _, is_article = pubmed._get_document_type(article_xml)
    article_doc_node = pubmed._get_document_node(article_xml, is_article)
    assert article_doc_node is not None

    # Test title extraction
    book_title = pubmed._get_title(book_doc_node, "BookChapter")
    article_title = pubmed._get_title(article_doc_node, "Article")

    assert book_title == "Test Book Chapter"
    assert article_title == "Test Journal Article"

    # Test fallback to book title if article title is missing
    book_without_article_title = ET.fromstring(
        """
    <BookDocument>
        <Book>
            <BookTitle>Only Book Title</BookTitle>
        </Book>
    </BookDocument>
    """
    )
    title = pubmed._get_title(book_without_article_title, "BookChapter")
    assert title == "Only Book Title"


def test_get_abstract_texts(
    book_article_xml: ET.Element, article_xml: ET.Element
) -> None:
    """Test abstract extraction from different document types."""
    # Get document nodes first
    _, is_book_article = pubmed._get_document_type(book_article_xml)
    book_doc_node = pubmed._get_document_node(book_article_xml, is_book_article)
    assert book_doc_node is not None
    _, is_article = pubmed._get_document_type(article_xml)
    article_doc_node = pubmed._get_document_node(article_xml, is_article)
    assert article_doc_node is not None

    # Test abstract extraction
    book_abstract_texts = pubmed._get_abstract_texts(book_doc_node, "BookChapter")
    article_abstract_texts = pubmed._get_abstract_texts(article_doc_node, "Article")

    assert len(book_abstract_texts) == 1
    assert "test abstract for the book chapter" in book_abstract_texts[0]

    assert len(article_abstract_texts) == 1
    assert "test abstract for the journal article" in article_abstract_texts[0]

    # Test other abstract locations for books
    book_with_other_abstract = ET.fromstring(
        """
    <BookDocument>
        <OtherAbstract>
            <AbstractText>Other abstract location test</AbstractText>
        </OtherAbstract>
    </BookDocument>
    """
    )
    abstract_texts = pubmed._get_abstract_texts(book_with_other_abstract, "BookChapter")
    assert len(abstract_texts) == 1
    assert abstract_texts[0] == "Other abstract location test"


def test_get_journal(book_article_xml: ET.Element, article_xml: ET.Element) -> None:
    """Test journal/book title extraction."""
    # Get document nodes first
    _, is_book_article = pubmed._get_document_type(book_article_xml)
    book_doc_node = pubmed._get_document_node(book_article_xml, is_book_article)
    assert book_doc_node is not None
    _, is_article = pubmed._get_document_type(article_xml)
    article_doc_node = pubmed._get_document_node(article_xml, is_article)
    assert article_doc_node is not None

    # Test journal extraction
    book_journal = pubmed._get_journal(book_doc_node, "BookChapter")
    article_journal = pubmed._get_journal(article_doc_node, "Article")

    assert book_journal == "Test Book Title"
    assert article_journal == "Test Journal"


def test_get_year_text(book_article_xml: ET.Element, article_xml: ET.Element) -> None:
    """Test year extraction from different document types."""
    # Get document nodes first
    _, is_book_article = pubmed._get_document_type(book_article_xml)
    book_doc_node = pubmed._get_document_node(book_article_xml, is_book_article)
    assert book_doc_node is not None
    _, is_article = pubmed._get_document_type(article_xml)
    article_doc_node = pubmed._get_document_node(article_xml, is_article)
    assert article_doc_node is not None

    # Test year extraction
    book_year = pubmed._get_year_text(book_doc_node, "BookChapter")
    article_year = pubmed._get_year_text(article_doc_node, "Article")

    assert book_year == "2023"
    assert article_year == "2024"


def test_get_author_list_node(
    book_article_xml: ET.Element, article_xml: ET.Element
) -> None:
    """Test author list node extraction."""
    # Get document nodes first
    _, is_book_article = pubmed._get_document_type(book_article_xml)
    book_doc_node = pubmed._get_document_node(book_article_xml, is_book_article)
    assert book_doc_node is not None
    _, is_article = pubmed._get_document_type(article_xml)
    article_doc_node = pubmed._get_document_node(article_xml, is_article)
    assert article_doc_node is not None

    # Test author list node extraction
    book_author_list = pubmed._get_author_list_node(book_doc_node, "BookChapter")
    article_author_list = pubmed._get_author_list_node(article_doc_node, "Article")

    # Verify we got author list nodes
    assert book_author_list is not None
    assert article_author_list is not None

    # For books, should prioritize chapter authors over book authors
    author_element = book_author_list.find(".//Author/LastName")
    assert author_element is not None
    assert author_element.text == "Jones"

    # Test fallback to book authors if chapter authors not found
    book_with_only_book_authors = ET.fromstring(
        """
    <BookDocument>
        <Book>
            <AuthorList Type="authors">
                <Author>
                    <LastName>BookAuthor</LastName>
                    <ForeName>Only</ForeName>
                </Author>
            </AuthorList>
        </Book>
    </BookDocument>
    """
    )
    author_list = pubmed._get_author_list_node(
        book_with_only_book_authors, "BookChapter"
    )
    assert author_list is not None
    author_element = author_list.find(".//Author/LastName")
    assert author_element is not None
    assert author_element.text == "BookAuthor"


def test_get_authors() -> None:
    """Test author name extraction and formatting."""
    # Create test author list
    author_list = ET.fromstring(
        """
    <AuthorList>
        <Author>
            <LastName>Smith</LastName>
            <ForeName>John</ForeName>
        </Author>
        <Author>
            <LastName>Doe</LastName>
            <ForeName>Jane</ForeName>
        </Author>
        <Author>  <!-- Author with missing ForeName -->
            <LastName>NoFirstName</LastName>
        </Author>
    </AuthorList>
    """
    )

    authors = pubmed._get_authors(author_list)
    assert "John Smith" in authors
    assert "Jane Doe" in authors
    assert "NoFirstName" in authors

    # Test with empty author list
    empty_list = ET.fromstring("<AuthorList></AuthorList>")
    assert pubmed._get_authors(empty_list) == ""

    # Test with None
    assert pubmed._get_authors(None) == ""
