"""
PubMed domain module: Complete workflows for PubMed documents.

This module provides end-to-end functionality for PubMed documents, from API
ingestion to NLP processing. It handles PubMed-specific I/O and data conversion
while using generic processing infrastructure (core.pipeline).

Workflows:
    1. Ingestion: Fetch from PubMed E-utilities API -> Save to database
    2. Processing: Load from database -> Process with NLP pipeline

Usage:
    from biomed_kg_agent.data_sources.ncbi.pubmed import (
        ingest_pubmed_abstracts,
        process_pubmed_documents_from_db
    )

    # Ingestion: fetch abstracts with metadata (MeSH, DOI, keywords)
    ingest_pubmed_abstracts(
        term="cancer metabolism", limit=100, output_path="data/corpus.db"
    )

    # Processing: run NER + entity linking pipeline
    process_pubmed_documents_from_db(
        input_db_path="data/corpus.db", output_db_path="data/nlp.db"
    )

API Limits:
    - Maximum 9,999 PMIDs per query (E-utilities limitation)
    - Automatic query splitting by year/month/week for larger datasets
    - Rate limiting with configurable delays and exponential backoff
"""

import json
import logging
import os
import time
from calendar import monthrange
from xml.etree import ElementTree as ET

from sqlmodel import Session, SQLModel, create_engine, select

from biomed_kg_agent.core.connectors import pubmed_to_internal
from biomed_kg_agent.core.database import create_db_engine
from biomed_kg_agent.core.pipeline import process_documents
from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument

from . import ncbi_utils

logger = logging.getLogger("biomed_kg_agent.pubmed")


def _get_document_type(item: ET.Element) -> tuple[str, bool]:
    """
    Determine the document type based on the XML element tag.

    Args:
        item: The XML element representing a PubMed document

    Returns:
        Tuple of (document_type, is_book_article)
    """
    pmid = item.findtext(".//PMID") or "N/A"

    if item.tag == "PubmedArticle":
        logger.debug(f"PMID {pmid}: Identified as standard article")
        return "Article", False
    elif item.tag == "PubmedBookArticle":
        logger.debug(f"PMID {pmid}: Identified as book chapter")
        return "BookChapter", True
    else:
        # This case should ideally not be reached if articles_and_book_articles
        # is built from specific findall calls for known tags.
        logger.warning(
            f"Encountered an unexpected item tag: {item.tag} for PMID: {pmid}. "
            f"Setting doc_type to 'Unknown'."
        )
        return "Unknown", False


def _get_document_node(item: ET.Element, is_book_article: bool) -> ET.Element | None:
    """
    Get the document node based on document type.

    Args:
        item: The XML element representing a PubMed document
        is_book_article: Whether this is a book article

    Returns:
        The document node for further extraction
    """
    if is_book_article:
        return item.find(".//BookDocument")
    return item  # For PubmedArticle, item itself is the main node


def _get_title(doc_node: ET.Element, doc_type: str) -> str:
    """
    Extract title from document node.

    Args:
        doc_node: Document node for extraction
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        Title text or empty string
    """
    if doc_type == "BookChapter":
        title = doc_node.findtext(".//ArticleTitle")  # Chapter title
        if not title:
            title = doc_node.findtext(
                ".//Book/BookTitle"
            )  # Book title if no chapter title
        return title or ""
    else:  # Article or Unknown
        return doc_node.findtext(".//ArticleTitle") or ""


def _get_abstract_texts(doc_node: ET.Element, doc_type: str) -> list[str]:
    """
    Extract abstract text segments from document node.
    Updated to handle new XML structure with multiple <OtherAbstract> elements.

    Args:
        doc_node: Document node for extraction
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        List of abstract text segments
    """
    abstract_texts = []

    if doc_type == "BookChapter":
        # Try to find abstract in various locations for books
        abstract_node = doc_node.find(".//Abstract")  # Common for book chapters
        if abstract_node is not None:
            abstract_texts.extend(
                [abst.text or "" for abst in abstract_node.findall(".//AbstractText")]
            )

        # Fallback to OtherAbstract or a general AbstractText if directly
        # under BookDocument
        if not abstract_texts:
            other_abstract_texts = [
                abst.text or ""
                for abst in doc_node.findall(".//OtherAbstract/AbstractText")
            ]
            abstract_texts.extend(other_abstract_texts)

        # Last attempt: direct AbstractText under BookDocument
        if not abstract_texts:
            abstract_texts.extend(
                [abst.text or "" for abst in doc_node.findall(".//AbstractText")]
            )
    else:  # Article or Unknown
        # Primary abstract
        primary_abstract_texts = [
            abst.text or "" for abst in doc_node.findall(".//Abstract/AbstractText")
        ]
        if primary_abstract_texts:
            abstract_texts.extend(primary_abstract_texts)

        # Handle new structure: multiple OtherAbstract elements with different Types
        # (e.g., Type="COI" for Conflict of Interest)
        other_abstract_nodes = doc_node.findall(".//OtherAbstract")
        for oa_node in other_abstract_nodes:
            abstract_type = oa_node.get("Type", "unknown")
            logger.debug(f"Found OtherAbstract with Type='{abstract_type}'")

            oa_text_nodes = oa_node.findall(".//AbstractText")
            if oa_text_nodes:
                # For COI and similar, we might want to skip or handle differently
                if abstract_type.upper() in ["COI", "CONFLICT", "COMPETING"]:
                    # Skip conflict of interest sections for main abstract
                    logger.debug(f"Skipping OtherAbstract Type='{abstract_type}'")
                    continue
                else:
                    # Include other types of abstracts
                    abstract_texts.extend([oat.text or "" for oat in oa_text_nodes])

        # Fallback: direct AbstractText elements if no structured Abstract found
        if not abstract_texts:
            abstract_texts.extend(
                [abst.text or "" for abst in doc_node.findall(".//AbstractText")]
            )

    return abstract_texts


def _get_journal(doc_node: ET.Element, doc_type: str) -> str:
    """
    Extract journal or book title from document node.

    Args:
        doc_node: Document node for extraction
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        Journal name, book title, or empty string
    """
    if doc_type == "BookChapter":
        return (
            doc_node.findtext(".//Book/BookTitle") or ""
        )  # Use BookTitle as Journal for books
    else:  # Article or Unknown
        return doc_node.findtext(".//Journal/Title") or ""


def _get_year_text(doc_node: ET.Element, doc_type: str) -> str:
    """
    Extract year text from document node.

    Args:
        doc_node: Document node for extraction
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        Year text or empty string
    """
    if doc_type == "BookChapter":
        year_node = doc_node.find(".//PubDate/Year")
        if year_node is None:
            year_node = doc_node.find(".//Book/PubDate/Year")
        return (year_node.text or "") if year_node is not None else ""
    else:  # Article or Unknown
        return doc_node.findtext(".//PubDate/Year") or ""


def _get_author_list_node(doc_node: ET.Element, doc_type: str) -> ET.Element | None:
    """
    Get author list node from document node.

    Args:
        doc_node: Document node for extraction
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        Author list XML node or None
    """
    if doc_type == "BookChapter":
        # Authors can be for the chapter or the book
        # First try to find chapter authors (AuthorList directly under
        # BookDocument, not under Book)
        author_list_node = doc_node.find(
            "./AuthorList"
        )  # Chapter authors (direct child)
        if author_list_node is None:
            author_list_node = doc_node.find(
                ".//Book/AuthorList[@Type='authors']"
            )  # Book authors
        if author_list_node is None:  # Sometimes editors are listed
            author_list_node = doc_node.find(".//Book/AuthorList[@Type='editors']")
        return author_list_node
    else:  # Article or Unknown
        return doc_node.find(".//AuthorList")


def _get_authors(author_list_node: ET.Element | None) -> str:
    """
    Extract formatted author names from author list node.

    Args:
        author_list_node: XML node containing author list

    Returns:
        Comma-separated author names or empty string
    """
    if author_list_node is None:
        return ""

    author_names = [
        f"{author.findtext('ForeName') or ''} "
        f"{author.findtext('LastName') or ''}".strip()
        for author in author_list_node.findall(".//Author")
        if author.findtext("LastName")
    ]
    return ", ".join(filter(None, author_names))


def _get_mesh_terms(doc_node: ET.Element, doc_type: str) -> str:
    """
    Extract MeSH (Medical Subject Headings) terms from document node.

    Args:
        doc_node: Document node for extraction
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        Comma-separated MeSH terms or empty string
    """
    mesh_terms = []

    # For regular articles, MeSH terms are typically in MedlineCitation/MeshHeadingList
    if doc_type == "Article":
        mesh_headings = doc_node.findall(".//MeshHeadingList/MeshHeading")
        for mesh_heading in mesh_headings:
            descriptor_name = mesh_heading.findtext("DescriptorName")
            if descriptor_name:
                mesh_terms.append(descriptor_name)
    elif doc_type == "BookChapter":
        # For book chapters, MeSH terms might be in a similar location
        mesh_headings = doc_node.findall(".//MeshHeadingList/MeshHeading")
        for mesh_heading in mesh_headings:
            descriptor_name = mesh_heading.findtext("DescriptorName")
            if descriptor_name:
                mesh_terms.append(descriptor_name)

    return ", ".join(mesh_terms)


def _get_doi(item: ET.Element, doc_type: str) -> str:
    """
    Extract DOI (Digital Object Identifier) from the full item element.

    Args:
        item: The full XML element (PubmedArticle or PubmedBookArticle)
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        DOI string or empty string
    """
    # DOI is typically found in PubmedData/ArticleIdList/ArticleId with IdType="doi"
    # This works for both articles and book articles
    article_ids = item.findall(".//ArticleIdList/ArticleId")
    for article_id in article_ids:
        id_type = article_id.get("IdType")
        if id_type == "doi" and article_id.text:
            return article_id.text.strip()

    return ""


def _get_keywords(doc_node: ET.Element, doc_type: str) -> str:
    """
    Extract author-provided keywords from document node.

    Args:
        doc_node: Document node for extraction
        doc_type: Type of document ("Article", "BookChapter", or "Unknown")

    Returns:
        Comma-separated keywords or empty string
    """
    keywords = []

    # Keywords are typically in KeywordList/Keyword
    keyword_elements = doc_node.findall(".//KeywordList/Keyword")
    for keyword_elem in keyword_elements:
        if keyword_elem.text:
            keywords.append(keyword_elem.text.strip())

    return ", ".join(keywords)


def _parse_item(item: ET.Element) -> PubmedDocument | None:
    """
    Parse a PubMed XML item into a PubmedDocument object.

    Args:
        item: XML element representing a PubMed article or book

    Returns:
        Populated PubmedDocument object or None if parsing fails
    """
    try:
        doc_type, is_book_article = _get_document_type(item)

        # For book articles, ensure we have a BookDocument node
        if is_book_article:
            doc_node = _get_document_node(item, is_book_article)
            if doc_node is None:
                pmid = item.findtext(".//PMID") or "unknown"
                logger.warning(
                    f"PMID {pmid}: Skipping PubmedBookArticle without "
                    f"BookDocument node."
                )
                # Return None for this item since we can't process it
                # without BookDocument
                return None
        else:
            doc_node = _get_document_node(item, is_book_article)

        # Check if we got a valid document node
        if doc_node is None:
            pmid = item.findtext(".//PMID") or "unknown"
            logger.warning(f"PMID {pmid}: Could not find document node.")
            return None

        # Extract all required fields
        pmid = doc_node.findtext(".//PMID") or ""
        title = _get_title(doc_node, doc_type)
        abstract_texts = _get_abstract_texts(doc_node, doc_type)
        abstract = " ".join(filter(None, abstract_texts)).strip()
        journal = _get_journal(doc_node, doc_type)

        # Handle year conversion
        year = None
        year_text = _get_year_text(doc_node, doc_type)
        if year_text and year_text.isdigit():
            year = int(year_text)

        # Handle authors
        author_list_node = _get_author_list_node(doc_node, doc_type)
        authors = _get_authors(author_list_node)

        # Extract additional fields
        mesh_terms = _get_mesh_terms(doc_node, doc_type)
        doi = _get_doi(item, doc_type)  # Pass the full item for DOI extraction
        keywords = _get_keywords(doc_node, doc_type)

        # Create and return the document
        return PubmedDocument(
            pmid=pmid,
            title=title,
            abstract=abstract,  # Will be empty string if not found
            document_type=doc_type,
            journal=journal,
            year=year,
            authors=authors or None,
            mesh_terms=mesh_terms or None,
            doi=doi or None,
            keywords=keywords or None,
        )
    except Exception as e:
        pmid = item.findtext(".//PMID") or "unknown"
        logger.error(f"PMID {pmid}: Failed to parse PubMed item: {e}")
        return None


def _parse_pubmed_xml(xml_string: str) -> list[PubmedDocument]:
    """
    Parse PubMed XML string into a list of PubmedDocument objects.

    Args:
        xml_string: Raw XML string from PubMed API

    Returns:
        List of parsed PubmedDocument objects
    """
    try:
        root = ET.fromstring(xml_string)
        docs = []

        # Handle both PubmedArticle and PubmedBookArticle
        articles = root.findall(".//PubmedArticle")
        book_articles = root.findall(".//PubmedBookArticle")
        total_items = len(articles) + len(book_articles)

        if total_items > 0:
            logger.info(
                f"Found {len(articles)} articles and {len(book_articles)} "
                f"book chapters to parse"
            )
        else:
            logger.warning(
                "No PubmedArticle or PubmedBookArticle elements found in XML"
            )

        # Process both types together
        articles_and_book_articles = articles + book_articles

        for item in articles_and_book_articles:
            doc = _parse_item(item)
            if doc:  # Only add non-None results
                docs.append(doc)

        return docs
    except ET.ParseError as e:
        logger.error(f"Failed to parse PubMed XML: {e}")
        return []


def _download_batch(
    batch: list[str], use_post_threshold: int = 200, timeout: int = 45
) -> str:
    """
    Download batch of PMIDs from PubMed, using shared utilities.

    Args:
        batch: List of PMIDs to download
        use_post_threshold: Threshold for using POST instead of GET
        timeout: Request timeout in seconds

    Returns:
        Raw XML response as string, or empty string on failure
    """
    return ncbi_utils.fetch_xml_with_retry(
        database="pubmed", ids=batch, rettype="abstract", timeout=timeout
    )


def fetch_abstracts(
    pmids: list[str],
    batch_size: int = 200,
    delay: float = 0.35,
    use_post_threshold: int = 200,
) -> list[PubmedDocument]:
    """
    Download abstracts and return validated PubmedDocument objects.
    Batches requests and uses POST for large batches.
    Includes retry mechanism with exponential backoff.

    This function handles both regular journal articles (PubmedArticle) and
    book chapters (PubmedBookArticle) from PubMed. The document_type field in
    the returned PubmedDocument objects will be set accordingly.

    Args:
        pmids: List of PubMed IDs to fetch
        batch_size: Number of PMIDs per request
        delay: Delay between batches in seconds to avoid overwhelming the
            PubMed API
        use_post_threshold: Threshold for using POST instead of GET requests

    Returns:
        List of PubmedDocument objects with document_type set to "Article",
        "BookChapter", or "Unknown" based on the source content type
    """
    if not pmids:
        return []

    logger.info(f"Fetching abstracts for {len(pmids)} PMIDs in batches of {batch_size}")
    docs = []
    processed_count = 0

    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        processed_count += len(batch)
        logger.debug(
            f"Processing batch {i // batch_size + 1} with {len(batch)} PMIDs "
            f"({processed_count}/{len(pmids)})"
        )

        xml = _download_batch(batch, use_post_threshold)

        if xml:  # Only parse if we got a response
            batch_docs = _parse_pubmed_xml(xml)
            if batch_docs:
                logger.debug(f"Retrieved {len(batch_docs)} documents from batch")
                # Count document types for logging
                article_count = sum(
                    1 for doc in batch_docs if doc.document_type == "Article"
                )
                book_count = sum(
                    1 for doc in batch_docs if doc.document_type == "BookChapter"
                )
                unknown_count = sum(
                    1 for doc in batch_docs if doc.document_type == "Unknown"
                )

                if article_count > 0:
                    logger.info(f"Batch includes {article_count} articles")
                if book_count > 0:
                    logger.info(f"Batch includes {book_count} book chapters")
                if unknown_count > 0:
                    logger.warning(
                        f"Batch includes {unknown_count} documents of unknown type"
                    )

                docs.extend(batch_docs)
            else:
                logger.warning(
                    f"No documents retrieved for batch starting with PMID {batch[0]}"
                )

        if i + batch_size < len(pmids):  # Don't delay after the last batch
            time.sleep(delay)

    logger.info(f"Successfully retrieved {len(docs)}/{len(pmids)} documents")
    # Report any missing PMIDs
    if len(docs) < len(pmids):
        retrieved_pmids = {doc.pmid for doc in docs}
        missing_pmids = set(pmids) - retrieved_pmids
        if len(missing_pmids) <= 10:
            logger.warning(f"Missing documents for PMIDs: {', '.join(missing_pmids)}")
        else:
            logger.warning(f"Missing documents for {len(missing_pmids)} PMIDs")

    return docs


def save_to_sqlite(docs: list[PubmedDocument], db_path: str = "data/pubmed.db") -> None:
    """
    Store documents in a local SQLite database.
    """
    if not docs:
        logger.warning("No documents to save.")
        return

    # Create directory if db_path contains a directory
    db_dir = os.path.dirname(db_path)
    if db_dir:  # Only create directory if there's actually a directory path
        os.makedirs(db_dir, exist_ok=True)
    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        for doc in docs:
            session.merge(doc)
        session.commit()
    num_docs = len(docs)
    logger.info(f"Saved {num_docs} documents to {db_path}")


def save_to_json(docs: list[PubmedDocument], filepath: str) -> None:
    """
    Export documents to a JSON file.
    """
    if not docs:
        logger.warning("No documents to save.")
        return
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump([doc.model_dump() for doc in docs], f, indent=2)
    num_docs = len(docs)
    logger.info(f"Saved {num_docs} documents to {filepath}")


def get_total_count(term: str) -> int:
    """Get total count of PubMed articles for a search term."""
    return ncbi_utils.get_total_count("pubmed", term)


def fetch_ids_for_term(
    search_term: str, max_uids_per_query: int = ncbi_utils.MAX_UIDS_PER_QUERY
) -> list[str]:
    """
    Fetch up to max_uids_per_query PMIDs for a given search term.
    PubMed ESearch API does not allow pagination beyond 10,000 UIDs.

    Args:
        search_term: The term to search for in PubMed
        max_uids_per_query: Maximum UIDs allowed per query (default: 10000)

    Returns:
        list[str]: List of PubMed IDs
    """
    return ncbi_utils.fetch_ids_for_term("pubmed", search_term, max_uids_per_query)


def _deduplicate_and_limit_ids(
    raw_ids: list[str], limit: int, seen: set[str]
) -> list[str]:
    """
    Deduplicates a list of raw IDs against a set of already seen IDs,
    and limits the number of new, unique IDs to be returned.

    Args:
        raw_ids: The list of IDs to process.
        limit: The maximum number of unique IDs to collect in the current
            context (this refers to the `remaining` or `limit` parameter
            of the calling function).
        seen: A set of already seen IDs. This set will be updated in-place.

    Returns:
        A list of new, unique IDs, ensuring not to exceed the overall
        limit when combined with already seen IDs.
    """
    return ncbi_utils.deduplicate_and_limit_ids(raw_ids, limit, seen)


def split_by_week(
    year: int,
    month: int,
    base_term: str,
    remaining: int,
    seen: set[str],
    max_uids_per_query: int = ncbi_utils.MAX_UIDS_PER_QUERY,
) -> list[str]:
    """
    Split a PubMed query into 7-day chunks for a given month, stopping once
    we've collected up to `remaining` unique PMIDs (or exhausted the
    weeks in that month).

    Args:
        year: The year to query
        month: The month to query
        base_term: The base search term to query PubMed with
        remaining: Maximum number of PMIDs to return
        seen: Set of already seen PMIDs
        max_uids_per_query: Maximum UIDs allowed per query (default: 10000)

    Returns:
        list[str]: List of unique PMIDs
    """
    ids: list[str] = []
    if remaining <= 0 or len(seen) >= remaining:
        return ids

    days_in_month = monthrange(year, month)[1]
    start_day = 1

    while start_day <= days_in_month and len(seen) < remaining:
        end_day = min(start_day + 6, days_in_month)
        date_range = (
            f"{year}/{month:02d}/{start_day:02d}:{year}/{month:02d}/{end_day:02d}[dp]"
        )
        week_term = f"({base_term}) AND {date_range}"

        week_count = get_total_count(week_term)
        if week_count == 0:
            # No PMIDs in this week, skip ahead
            start_day = end_day + 1
            continue

        if week_count > max_uids_per_query:
            logger.warning(
                f"Week {year}-{month:02d}-{start_day:02d} to "
                f"{end_day:02d} has {week_count} results (>{max_uids_per_query}). "
                "Consider refining your query."
            )

        actual_fetching_count = min(week_count, max_uids_per_query)
        logger.info(
            f"Querying PubMed for week {year}-{month:02d}-{start_day:02d}"
            f"to {end_day:02d}. "
            f"Reported week count: {week_count}."
            f"Will fetch up to {actual_fetching_count} PMIDs for this chunk."
        )

        week_ids = fetch_ids_for_term(week_term, max_uids_per_query)
        new_ids = _deduplicate_and_limit_ids(week_ids, remaining, seen)
        ids.extend(new_ids)
        start_day = end_day + 1

    return ids


def split_by_month(
    year: int,
    base_term: str,
    remaining: int,
    seen: set[str],
    max_uids_per_query: int = ncbi_utils.MAX_UIDS_PER_QUERY,
) -> list[str]:
    """
    Split a PubMed query into monthly chunks for a given year, stopping once
    we've collected up to `remaining` unique PMIDs (or exhausted
    the months in that year).

    Args:
        year: The year to query
        base_term: The base search term to query PubMed with
        remaining: Maximum number of PMIDs to return
        seen: Set of already seen PMIDs
        max_uids_per_query: Maximum UIDs allowed per query (default: 10000)

    Returns:
        list[str]: List of unique PMIDs
    """
    ids: list[str] = []
    for month in range(12, 0, -1):  # Iterate from December (12) down to January (1)
        if len(seen) >= remaining:
            break
        month_term = f"({base_term}) AND {year}/{month:02d}[dp]"
        month_count = get_total_count(month_term)
        if month_count == 0:
            continue
        if month_count > max_uids_per_query:
            logger.warning(
                f"Month {year}-{month:02d} exceeds {max_uids_per_query} results. "
                "Splitting by week."
            )
            week_ids = split_by_week(
                year, month, base_term, remaining, seen, max_uids_per_query
            )
            ids.extend(week_ids)
        else:
            to_fetch = min(month_count, remaining - len(seen))
            logger.info(f"Fetching {to_fetch} PMIDs for month {year}-{month:02d}")
            month_ids = fetch_ids_for_term(month_term, max_uids_per_query)
            new_ids = _deduplicate_and_limit_ids(month_ids, remaining, seen)
            ids.extend(new_ids)
    return ids


def split_by_year(
    term: str,
    remaining: int,
    seen: set[str],
    max_uids_per_query: int = ncbi_utils.MAX_UIDS_PER_QUERY,
) -> list[str]:
    """
    Split a PubMed query into yearly chunks, stopping once we've collected up to
    `remaining` unique PMIDs.

    Args:
        term: The search term to query PubMed with
        remaining: Maximum number of PMIDs to return
        seen: Set of already seen PMIDs
        max_uids_per_query: Maximum UIDs allowed per query (default: 10000)

    Returns:
        list[str]: List of unique PMIDs
    """
    ids: list[str] = []
    current_year = time.localtime().tm_year
    for year in range(current_year, 1945, -1):
        if len(seen) >= remaining:
            break
        year_term = f"({term}) AND {year}[dp]"
        year_count = get_total_count(year_term)
        if year_count == 0:
            continue
        if year_count > max_uids_per_query:
            logger.warning(
                f"Year {year} exceeds {max_uids_per_query} results. Splitting by month."
            )
            month_ids = split_by_month(year, term, remaining, seen, max_uids_per_query)
            ids.extend(month_ids)
        else:
            to_fetch = min(year_count, remaining - len(seen))
            logger.info(f"Fetching {to_fetch} PMIDs for year {year}")
            year_ids = fetch_ids_for_term(year_term, max_uids_per_query)
            new_ids = _deduplicate_and_limit_ids(year_ids, remaining, seen)
            ids.extend(new_ids)
    return ids


def get_pubmed_pmids(
    term: str, limit: int, *, max_uids_per_query: int = ncbi_utils.MAX_UIDS_PER_QUERY
) -> list[str]:
    """
    Retrieve PubMed PMIDs matching the term using modern pagination.

    Prefers simple pagination over complex time-based splitting.
    Falls back to time-based splitting if pagination fails.

    Args:
        term: The search term to query PubMed with
        limit: Maximum number of PMIDs to return
        max_uids_per_query: Maximum UIDs allowed per query (default: 10000)

    Returns:
        list[str]: List of unique PubMed IDs up to the limit
    """
    try:
        # Try modern pagination first
        pmids = ncbi_utils.paginated_id_fetch("pubmed", term, limit)
        if pmids:  # empty if pagination fails
            return pmids

        # Fallback to time-based splitting if pagination fails
        logger.warning("Modern pagination failed, falling back to time-based splitting")
        seen: set[str] = set()
        ids = split_by_year(term, limit, seen, max_uids_per_query)
        return ids[:limit]

    except Exception as e:
        logger.error(f"Error in get_pubmed_pmids for term '{term}': {e}")
        return []


def ingest_pubmed_abstracts(
    term: str,
    limit: int,
    output_path: str,
    output_format: str = "sqlite",
    batch_size: int = 200,
    delay: float = 0.34,
) -> dict:
    """Orchestrate PubMed ingestion: search -> fetch -> save.

    Args:
        term: PubMed search query
        limit: Maximum number of abstracts to fetch
        output_path: Path where output will be saved
        output_format: Output format ("sqlite" or "json")
        batch_size: Number of abstracts to fetch per batch
        delay: Delay between API requests (seconds)

    Returns:
        dict with keys:
            - doc_count: Number of documents successfully saved
            - output_path: Path where documents were saved
            - total_available: Total number of matching documents in PubMed

    """
    # Get total count
    total_count = get_total_count(term)
    actual_limit = min(limit, total_count)

    # Fetch PMIDs
    pmids = get_pubmed_pmids(term, actual_limit)
    if not pmids:
        logger.warning(f"No PMIDs found for term '{term}'")
        return {
            "doc_count": 0,
            "output_path": output_path,
            "total_available": total_count,
        }

    # Fetch abstracts
    docs = fetch_abstracts(pmids, batch_size=batch_size, delay=delay)
    if not docs:
        logger.error(f"Failed to fetch abstracts for {len(pmids)} PMIDs")
        return {
            "doc_count": 0,
            "output_path": output_path,
            "total_available": total_count,
        }

    # Save
    if output_format == "sqlite":
        save_to_sqlite(docs, output_path)
    else:
        save_to_json(docs, output_path)

    logger.info(
        f"Ingestion complete: {len(docs)} documents saved to {output_path} "
        f"(format: {output_format})"
    )

    return {
        "doc_count": len(docs),
        "output_path": output_path,
        "total_available": total_count,
    }


def process_pubmed_documents_from_db(
    input_db_path: str,
    output_db_path: str,
    config_path: str | None = None,
) -> dict:
    """Load PubMed docs from SQLite, process with NLP, return stats.

    Orchestrates the complete NLP processing pipeline for PubMed documents:
    1. Load documents from input SQLite database
    2. Convert to internal format
    3. Process with NER and entity linking
    4. Save results to output database

    Args:
        input_db_path: Path to SQLite database containing PubMed documents
        output_db_path: Path where NLP results will be saved
        config_path: Optional path to NER config file

    Returns:
        dict with keys:
            - doc_count: Number of documents processed
            - entity_count: Total number of entities extracted
            - umls_linked: Number of entities linked to UMLS
            - umls_rate: Percentage of entities linked to UMLS
            - output_path: Path where results were saved

    """
    # Load documents from database
    engine = create_db_engine(db_path=input_db_path)
    with Session(engine) as session:
        statement = select(PubmedDocument).order_by(PubmedDocument.pmid)
        pubmed_docs = session.exec(statement).all()

    if not pubmed_docs:
        logger.warning(f"No documents found in database: {input_db_path}")
        return {
            "doc_count": 0,
            "entity_count": 0,
            "umls_linked": 0,
            "umls_rate": 0,
            "output_path": output_db_path,
        }

    logger.info(f"Loaded {len(pubmed_docs)} documents from {input_db_path}")

    # Convert to internal format and process
    internal_docs = [pubmed_to_internal(doc) for doc in pubmed_docs]
    processed_docs = process_documents(internal_docs, config_path, output_db_path)

    # Calculate statistics from processed documents
    total_entities = sum(len(doc.entities) for doc in processed_docs)
    umls_linked = sum(
        sum(1 for e in doc.entities if e.umls_cui) for doc in processed_docs
    )
    umls_rate = (umls_linked / total_entities * 100) if total_entities > 0 else 0

    logger.info(
        f"NLP processing complete: {len(processed_docs)} documents, "
        f"{total_entities} entities ({umls_linked} UMLS-linked) saved to {output_db_path}"
    )

    return {
        "doc_count": len(processed_docs),
        "entity_count": total_entities,
        "umls_linked": umls_linked,
        "umls_rate": umls_rate,
        "output_path": output_db_path,
    }
