"""
Tests for the extraction helper functions in the pubmed module.

This file focuses on testing the individual extraction functions that are used
to parse PubMed XML data, with a special focus on book article handling.
"""

import xml.etree.ElementTree as ET

from biomed_kg_agent.data_sources.ncbi.pubmed import (
    _get_abstract_texts,
    _get_author_list_node,
    _get_authors,
    _get_document_node,
    _get_document_type,
    _get_journal,
    _get_title,
    _get_year_text,
)


def test_get_document_type() -> None:
    """Test document type detection for different XML element tags."""
    # Test Article
    article_xml = ET.fromstring("<PubmedArticle><PMID>123</PMID></PubmedArticle>")
    doc_type, is_book = _get_document_type(article_xml)
    assert doc_type == "Article"
    assert is_book is False

    # Test BookChapter
    book_xml = ET.fromstring("<PubmedBookArticle><PMID>456</PMID></PubmedBookArticle>")
    doc_type, is_book = _get_document_type(book_xml)
    assert doc_type == "BookChapter"
    assert is_book is True

    # Test Unknown
    unknown_xml = ET.fromstring("<UnknownType><PMID>789</PMID></UnknownType>")
    doc_type, is_book = _get_document_type(unknown_xml)
    assert doc_type == "Unknown"
    assert is_book is False


def test_get_document_node() -> None:
    """Test retrieval of appropriate document node based on document type."""
    # Test book article node extraction
    book_xml = ET.fromstring(
        """
        <PubmedBookArticle>
            <BookDocument>
                <PMID>456</PMID>
            </BookDocument>
        </PubmedBookArticle>
    """
    )
    node = _get_document_node(book_xml, True)
    assert node is not None
    assert node.tag == "BookDocument"
    assert node.findtext("PMID") == "456"

    # Test non-book article node extraction (should return the item itself)
    article_xml = ET.fromstring("<PubmedArticle><PMID>123</PMID></PubmedArticle>")
    node = _get_document_node(article_xml, False)
    assert node is article_xml


def test_get_title_book_chapter() -> None:
    """Test extracting titles from book chapters with different XML structures."""
    # Test book with both ArticleTitle and BookTitle
    book_xml = ET.fromstring(
        """
        <BookDocument>
            <ArticleTitle>Chapter Title</ArticleTitle>
            <Book>
                <BookTitle>Book Title</BookTitle>
            </Book>
        </BookDocument>
    """
    )
    title = _get_title(book_xml, "BookChapter")
    assert title == "Chapter Title"  # Should prefer ArticleTitle

    # Test book with only BookTitle
    book_xml_no_article_title = ET.fromstring(
        """
        <BookDocument>
            <Book>
                <BookTitle>Book Title Only</BookTitle>
            </Book>
        </BookDocument>
    """
    )
    title = _get_title(book_xml_no_article_title, "BookChapter")
    assert title == "Book Title Only"  # Should fall back to BookTitle

    # Test book with no title
    book_xml_no_title = ET.fromstring("<BookDocument></BookDocument>")
    title = _get_title(book_xml_no_title, "BookChapter")
    assert title == ""  # Should return empty string if no title found


def test_get_title_article() -> None:
    """Test extracting titles from regular articles."""
    # Test article with title
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <ArticleTitle>Article Title</ArticleTitle>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    """
    )
    title = _get_title(article_xml, "Article")
    assert title == "Article Title"

    # Test article with no title
    article_xml_no_title = ET.fromstring("<PubmedArticle></PubmedArticle>")
    title = _get_title(article_xml_no_title, "Article")
    assert title == ""  # Should return empty string if no title found


def test_get_abstract_texts_book_chapter() -> None:
    """Test extracting abstract texts from book chapters with different
    XML structures."""
    # Test book with standard Abstract/AbstractText structure
    book_xml = ET.fromstring(
        """
        <BookDocument>
            <Abstract>
                <AbstractText>Abstract content 1</AbstractText>
                <AbstractText>Abstract content 2</AbstractText>
            </Abstract>
        </BookDocument>
    """
    )
    texts = _get_abstract_texts(book_xml, "BookChapter")
    assert len(texts) == 2
    assert texts[0] == "Abstract content 1"
    assert texts[1] == "Abstract content 2"

    # Test book with OtherAbstract
    book_xml_other = ET.fromstring(
        """
        <BookDocument>
            <OtherAbstract>
                <AbstractText>Other abstract content</AbstractText>
            </OtherAbstract>
        </BookDocument>
    """
    )
    texts = _get_abstract_texts(book_xml_other, "BookChapter")
    assert len(texts) == 1
    assert texts[0] == "Other abstract content"

    # Test book with direct AbstractText
    book_xml_direct = ET.fromstring(
        """
        <BookDocument>
            <AbstractText>Direct abstract</AbstractText>
        </BookDocument>
    """
    )
    texts = _get_abstract_texts(book_xml_direct, "BookChapter")
    assert len(texts) == 1
    assert texts[0] == "Direct abstract"

    # Test book with no abstract
    book_xml_no_abstract = ET.fromstring("<BookDocument></BookDocument>")
    texts = _get_abstract_texts(book_xml_no_abstract, "BookChapter")
    assert len(texts) == 0


def test_get_abstract_texts_article() -> None:
    """Test extracting abstract texts from regular articles."""
    # Test article with standard AbstractText elements
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <Abstract>
                        <AbstractText>Article abstract 1</AbstractText>
                        <AbstractText>Article abstract 2</AbstractText>
                    </Abstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    """
    )
    texts = _get_abstract_texts(article_xml, "Article")
    assert len(texts) == 2
    assert texts[0] == "Article abstract 1"
    assert texts[1] == "Article abstract 2"

    # Test article with OtherAbstract
    article_xml_other = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <OtherAbstract>
                        <AbstractText>Other article abstract</AbstractText>
                    </OtherAbstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    """
    )
    texts = _get_abstract_texts(article_xml_other, "Article")
    assert len(texts) == 1
    assert texts[0] == "Other article abstract"

    # Test article with OtherAbstract with direct text
    article_xml_other_direct = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <OtherAbstract>Some direct text</OtherAbstract>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    """
    )
    texts = _get_abstract_texts(article_xml_other_direct, "Article")
    assert (
        len(texts) == 0
    )  # Should be empty since we check for OtherAbstract.text only if it has
    # no AbstractText children

    # Test article with no abstract
    article_xml_no_abstract = ET.fromstring("<PubmedArticle></PubmedArticle>")
    texts = _get_abstract_texts(article_xml_no_abstract, "Article")
    assert len(texts) == 0


def test_get_journal_book_chapter() -> None:
    """Test extracting journal/book title from book chapters."""
    # Test book with BookTitle
    book_xml = ET.fromstring(
        """
        <BookDocument>
            <Book>
                <BookTitle>Book Series Title</BookTitle>
            </Book>
        </BookDocument>
    """
    )
    journal = _get_journal(book_xml, "BookChapter")
    assert journal == "Book Series Title"

    # Test book with no BookTitle
    book_xml_no_title = ET.fromstring("<BookDocument><Book></Book></BookDocument>")
    journal = _get_journal(book_xml_no_title, "BookChapter")
    assert journal == ""


def test_get_journal_article() -> None:
    """Test extracting journal name from regular articles."""
    # Test article with Journal/Title
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <Journal>
                        <Title>Journal of Testing</Title>
                    </Journal>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    """
    )
    journal = _get_journal(article_xml, "Article")
    assert journal == "Journal of Testing"

    # Test article with no Journal/Title
    article_xml_no_journal = ET.fromstring("<PubmedArticle></PubmedArticle>")
    journal = _get_journal(article_xml_no_journal, "Article")
    assert journal == ""


def test_get_year_text_book_chapter() -> None:
    """Test extracting publication year from book chapters."""
    # Test book with PubDate/Year
    book_xml = ET.fromstring(
        """
        <BookDocument>
            <PubDate>
                <Year>2023</Year>
            </PubDate>
        </BookDocument>
    """
    )
    year = _get_year_text(book_xml, "BookChapter")
    assert year == "2023"

    # Test book with Book/PubDate/Year
    book_xml_book_pubdate = ET.fromstring(
        """
        <BookDocument>
            <Book>
                <PubDate>
                    <Year>2022</Year>
                </PubDate>
            </Book>
        </BookDocument>
    """
    )
    year = _get_year_text(book_xml_book_pubdate, "BookChapter")
    assert year == "2022"

    # Test book with no Year
    book_xml_no_year = ET.fromstring("<BookDocument></BookDocument>")
    year = _get_year_text(book_xml_no_year, "BookChapter")
    assert year == ""


def test_get_year_text_article() -> None:
    """Test extracting publication year from regular articles."""
    # Test article with PubDate/Year
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <PubDate>
                        <Year>2021</Year>
                    </PubDate>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    """
    )
    year = _get_year_text(article_xml, "Article")
    assert year == "2021"

    # Test article with no Year
    article_xml_no_year = ET.fromstring("<PubmedArticle></PubmedArticle>")
    year = _get_year_text(article_xml_no_year, "Article")
    assert year == ""


def test_get_author_list_node_book_chapter() -> None:
    """Test extracting author list node from book chapters."""
    # Test book with chapter authors
    book_xml = ET.fromstring(
        """
        <BookDocument>
            <AuthorList>
                <Author>
                    <LastName>Smith</LastName>
                </Author>
            </AuthorList>
        </BookDocument>
    """
    )
    author_list = _get_author_list_node(book_xml, "BookChapter")
    assert author_list is not None
    author_element = author_list.find(".//LastName")
    assert author_element is not None
    assert author_element.text == "Smith"

    # Test book with book authors
    book_xml_book_authors = ET.fromstring(
        """
        <BookDocument>
            <Book>
                <AuthorList Type="authors">
                    <Author>
                        <LastName>Johnson</LastName>
                    </Author>
                </AuthorList>
            </Book>
        </BookDocument>
    """
    )
    author_list = _get_author_list_node(book_xml_book_authors, "BookChapter")
    assert author_list is not None
    author_element = author_list.find(".//LastName")
    assert author_element is not None
    assert author_element.text == "Johnson"

    # Test book with editors
    book_xml_editors = ET.fromstring(
        """
        <BookDocument>
            <Book>
                <AuthorList Type="editors">
                    <Author>
                        <LastName>Williams</LastName>
                    </Author>
                </AuthorList>
            </Book>
        </BookDocument>
    """
    )
    author_list = _get_author_list_node(book_xml_editors, "BookChapter")
    assert author_list is not None
    author_element = author_list.find(".//LastName")
    assert author_element is not None
    assert author_element.text == "Williams"

    # Test book with no authors
    book_xml_no_authors = ET.fromstring("<BookDocument></BookDocument>")
    author_list = _get_author_list_node(book_xml_no_authors, "BookChapter")
    assert author_list is None


def test_get_author_list_node_article() -> None:
    """Test extracting author list node from regular articles."""
    # Test article with authors
    article_xml = ET.fromstring(
        """
        <PubmedArticle>
            <MedlineCitation>
                <Article>
                    <AuthorList>
                        <Author>
                            <LastName>Davis</LastName>
                        </Author>
                    </AuthorList>
                </Article>
            </MedlineCitation>
        </PubmedArticle>
    """
    )
    author_list = _get_author_list_node(article_xml, "Article")
    assert author_list is not None
    author_element = author_list.find(".//LastName")
    assert author_element is not None
    assert author_element.text == "Davis"

    # Test article with no authors
    article_xml_no_authors = ET.fromstring("<PubmedArticle></PubmedArticle>")
    author_list = _get_author_list_node(article_xml_no_authors, "Article")
    assert author_list is None


def test_get_authors() -> None:
    """Test formatting author names from an author list node."""
    # Test with multiple authors
    authors_xml = ET.fromstring(
        """
        <AuthorList>
            <Author>
                <ForeName>John</ForeName>
                <LastName>Smith</LastName>
            </Author>
            <Author>
                <ForeName>Jane</ForeName>
                <LastName>Doe</LastName>
            </Author>
            <Author>
                <ForeName></ForeName>
                <LastName>Johnson</LastName>
            </Author>
            <Author>
                <ForeName>Alice</ForeName>
            </Author>
        </AuthorList>
    """
    )
    authors_string = _get_authors(authors_xml)
    assert "John Smith" in authors_string
    assert "Jane Doe" in authors_string
    assert "Johnson" in authors_string  # Should include LastName-only authors
    assert "Alice" not in authors_string  # Should not include ForeName-only authors

    # Test with no authors
    assert _get_authors(None) == ""
