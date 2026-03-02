"""
PMC (PubMed Central) data ingestion and processing.

This module provides functions to fetch and process PMC full-text articles,
including metadata extraction and document processing.
"""

import logging
import time
import xml.etree.ElementTree as ET

from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument

from .ncbi_utils import (
    MAX_UIDS_PER_QUERY,
    fetch_ids_for_term,
    fetch_xml_with_retry,
    get_total_count,
    paginated_id_fetch,
)

logger = logging.getLogger("biomed_kg_agent.pmc")


def get_pmc_total_count(term: str) -> int:
    """Get total count of PMC articles for a search term."""
    return get_total_count("pmc", term)


def fetch_pmc_ids_for_term(
    search_term: str, max_uids: int = MAX_UIDS_PER_QUERY
) -> list[str]:
    """Fetch PMC IDs for a search term."""
    return fetch_ids_for_term("pmc", search_term, max_uids)


def get_pmc_pmids(term: str, limit: int) -> list[str]:
    """Get PMC IDs for a search term using streamlined pagination."""
    try:
        return paginated_id_fetch("pmc", term, limit)
    except Exception as e:
        logger.error(f"Error fetching PMC IDs for '{term}': {e}")
        return []


def _parse_pmc_xml(xml_content: str) -> list[PubmedDocument]:
    """Parse PMC XML content into PubmedDocument objects."""
    documents = []

    try:
        root = ET.fromstring(xml_content)
        articles = root.findall(".//article")

        logger.info(f"Found {len(articles)} PMC articles in XML")

        for article in articles:
            try:
                # Get metadata
                front = article.find("front")
                if front is None:
                    continue

                article_meta = front.find(".//article-meta")
                if article_meta is None:
                    continue

                # Extract ID (PMID or PMC ID)
                pmid = _extract_article_id(article_meta)
                if not pmid:
                    continue

                # Extract metadata
                title = _extract_title(article_meta)
                authors = _extract_authors(article_meta)
                journal = _extract_journal(front)
                year = _extract_year(article_meta)

                # Extract content (abstract + full text)
                abstract_text = _extract_abstract(article_meta)
                full_text = _extract_full_text(article)

                # Combine abstract and full text
                content = (
                    f"{abstract_text}\n\n{full_text}"
                    if abstract_text and full_text
                    else (abstract_text or full_text)
                )

                # Create document
                doc = PubmedDocument(
                    pmid=pmid,
                    title=title,
                    abstract=content,
                    authors=authors,
                    journal=journal,
                    year=year,
                    mesh_terms=[],  # PMC doesn't have MeSH terms
                    doi=None,
                    pmc_id=pmid if pmid.startswith("PMC") else None,
                )

                documents.append(doc)

            except Exception as e:
                logger.error(f"Error parsing PMC article: {e}")
                continue

    except ET.ParseError as e:
        logger.error(f"Error parsing PMC XML: {e}")

    return documents


def _extract_article_id(article_meta: ET.Element) -> str:
    """Extract article ID (PMID or PMC ID)."""
    # Try PMID first
    for article_id in article_meta.findall(".//article-id"):
        if article_id.get("pub-id-type") == "pmid" and article_id.text:
            return article_id.text

    # Try PMC ID
    for article_id in article_meta.findall(".//article-id"):
        if article_id.get("pub-id-type") == "pmc" and article_id.text:
            return article_id.text

    # Try any ID
    for article_id in article_meta.findall(".//article-id"):
        if article_id.text:
            return article_id.text

    return ""


def _extract_title(article_meta: ET.Element) -> str:
    """Extract article title."""
    title_elem = article_meta.find(".//article-title")
    return title_elem.text if title_elem is not None and title_elem.text else "No title"


def _extract_authors(article_meta: ET.Element) -> list[str]:
    """Extract author names."""
    authors = []
    for contrib in article_meta.findall('.//contrib[@contrib-type="author"]'):
        name = contrib.find(".//name")
        if name is not None:
            surname = name.find("surname")
            given_names = name.find("given-names")
            if surname is not None and given_names is not None:
                authors.append(f"{given_names.text} {surname.text}")
    return authors


def _extract_journal(front: ET.Element) -> str:
    """Extract journal name."""
    journal_meta = front.find(".//journal-meta")
    if journal_meta is not None:
        journal_title = journal_meta.find(".//journal-title")
        if journal_title is not None and journal_title.text:
            return journal_title.text
    return "Unknown journal"


def _extract_year(article_meta: ET.Element) -> int | None:
    """Extract publication year."""
    pub_date = article_meta.find(".//pub-date")
    if pub_date is not None:
        year_elem = pub_date.find("year")
        if year_elem is not None and year_elem.text:
            try:
                return int(year_elem.text)
            except (ValueError, TypeError):
                pass
    return None


def _extract_abstract(article_meta: ET.Element) -> str:
    """Extract abstract text."""
    abstract_elem = article_meta.find(".//abstract")
    if abstract_elem is not None:
        return "".join(abstract_elem.itertext()).strip()
    return ""


def _extract_full_text(article: ET.Element) -> str:
    """Extract full text from article body."""
    body = article.find("body")
    if body is not None:
        return "".join(body.itertext()).strip()
    return ""


def fetch_pmc_articles(
    pmc_ids: list[str], batch_size: int = 50
) -> list[PubmedDocument]:
    """
    Fetch full-text articles from PMC.

    Args:
        pmc_ids: List of PMC IDs
        batch_size: Number of articles per batch

    Returns:
        List of PubmedDocument objects with full text
    """
    if not pmc_ids:
        return []

    logger.info(f"Fetching {len(pmc_ids)} PMC articles in batches of {batch_size}")
    docs = []

    for i in range(0, len(pmc_ids), batch_size):
        batch = pmc_ids[i : i + batch_size]
        logger.debug(f"Fetching PMC batch {i//batch_size + 1}: {len(batch)} articles")

        # Use shared fetch function
        xml = fetch_xml_with_retry("pmc", batch)

        if xml:
            batch_docs = _parse_pmc_xml(xml)
            docs.extend(batch_docs)
            logger.debug(f"Retrieved {len(batch_docs)} PMC documents from batch")

        if i + batch_size < len(pmc_ids):
            time.sleep(0.35)  # Rate limiting

    logger.info(f"Successfully retrieved {len(docs)}/{len(pmc_ids)} PMC documents")
    return docs


def get_articles_from_working_source(
    term: str, limit: int = 10
) -> list[PubmedDocument]:
    """
    Get articles from whichever source is currently working.

    Tries PMC first (since PubMed is currently down), with fallback logic.
    """
    try:
        # Try PMC first
        logger.info(f"Trying PMC for term '{term}'...")
        pmc_ids = get_pmc_pmids(term, limit)

        if pmc_ids:
            logger.info(f"PMC returned {len(pmc_ids)} IDs, fetching articles...")
            articles = fetch_pmc_articles(pmc_ids)
            if articles:
                logger.info(f"Successfully retrieved {len(articles)} articles from PMC")
                return articles

        logger.warning("No results from PMC, would try PubMed but it's currently down")
        return []

    except Exception as e:
        logger.error(f"Error getting articles from working source: {e}")
        return []
