"""
Shared utilities for NCBI E-utilities API interactions.

This module provides common functionality shared between PubMed and PMC modules,
eliminating code duplication and providing a consistent API interface.
"""

import logging
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable
from typing import Any

import requests  # type: ignore

from biomed_kg_agent.config import settings

logger = logging.getLogger("biomed_kg_agent.ncbi_utils")

# Constants shared across NCBI APIs
# Note: NCBI E-utilities has a hard limit where retstart cannot exceed 9998
# This effectively limits us to 9999 records per query, not 10000
MAX_UIDS_PER_QUERY = 9999
MAX_RETRIES = 3
INITIAL_BACKOFF = 5


def make_ncbi_request(
    endpoint: str, params: dict[str, Any], timeout: int = 30
) -> ET.Element:
    """
    Make a request to NCBI E-utilities API with error handling.

    Args:
        endpoint: API endpoint (esearch.fcgi, efetch.fcgi)
        params: Query parameters
        timeout: Request timeout in seconds

    Returns:
        Parsed XML root element

    Raises:
        RuntimeError: If API returns error or request fails
    """
    url = settings.PUBMED_API_BASE + endpoint

    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)

        # Check for API errors
        error_elem = root.find(".//ERROR")
        if error_elem is not None:
            error_msg = error_elem.text or "Unknown API error"
            db = params.get("db", "unknown")
            logger.error(f"NCBI API error for database '{db}': {error_msg}")
            raise RuntimeError(f"NCBI API error: {error_msg}")

        return root

    except requests.RequestException as e:
        logger.error(f"Request failed to {endpoint}: {e}")
        raise RuntimeError(f"NCBI request failed: {e}")
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML response: {e}")
        raise RuntimeError(f"XML parsing failed: {e}")


def get_total_count(database: str, term: str) -> int:
    """Get total count of articles for a search term in specified database."""
    params = {
        "db": database,
        "term": term,
        "retmax": 0,
        "retmode": "xml",
    }

    root = make_ncbi_request("esearch.fcgi", params)
    count_elem = root.find(".//Count")
    return int(count_elem.text or 0) if count_elem is not None else 0


def fetch_ids_for_term(
    database: str,
    search_term: str,
    max_uids: int = MAX_UIDS_PER_QUERY,
    retstart: int = 0,
) -> list[str]:
    """Fetch article IDs for a search term from specified database."""
    params = {
        "db": database,
        "term": search_term,
        "retmax": max_uids,
        "retstart": retstart,
        "retmode": "xml",
    }

    root = make_ncbi_request("esearch.fcgi", params)
    ids = [id_elem.text for id_elem in root.findall(".//Id") if id_elem.text]
    return ids


def deduplicate_and_limit_ids(
    raw_ids: list[str], limit: int, seen: set[str]
) -> list[str]:
    """
    Deduplicate IDs and limit to specified count.

    Args:
        raw_ids: List of IDs to process
        limit: Maximum number of unique IDs to return
        seen: Set of already seen IDs (modified in place)

    Returns:
        List of new unique IDs
    """
    new_ids = []
    for id_val in raw_ids:
        if len(seen) >= limit:
            break
        if id_val not in seen:
            new_ids.append(id_val)
            seen.add(id_val)
    return new_ids


def paginated_id_fetch(database: str, term: str, limit: int) -> list[str]:
    """
    Fetch IDs using modern pagination approach.

    Args:
        database: NCBI database (pubmed, pmc, etc.)
        term: Search term
        limit: Maximum number of IDs to return

    Returns:
        List of unique IDs up to limit
    """
    try:
        total_count = get_total_count(database, term)
        logger.info(
            f"{database.upper()} search '{term}': {total_count:,} total results"
        )

        # Calculate how many IDs we actually need to fetch
        to_fetch = min(limit, total_count)

        # Guardrail: ESearch cannot return more than MAX_UIDS_PER_QUERY results
        # in a single request. If we need more than this, signal the caller to
        # use time-window splitting instead.
        if to_fetch > MAX_UIDS_PER_QUERY:
            logger.warning(
                f"Need to fetch {to_fetch:,} IDs but ESearch cap is "
                f"{MAX_UIDS_PER_QUERY:,}; use time-based splitting"
            )
            return []

        # Inform users about sampling bias for large corpora
        if total_count > MAX_UIDS_PER_QUERY and to_fetch > 0:
            logger.info(
                f"Corpus has {total_count:,} results (>{MAX_UIDS_PER_QUERY:,}). "
                f"Returning first {to_fetch:,} by relevance. "
                f"Use time-based splitting for representative sampling across all results."
            )

        # Fetch the IDs
        ids = fetch_ids_for_term(database, term, to_fetch, 0)
        return deduplicate_and_limit_ids(ids, limit, set())

    except Exception as e:
        logger.error(f"Error fetching {database} IDs for '{term}': {e}")
        return []


def fetch_xml_with_retry(
    database: str, ids: list[str], rettype: str = "abstract", timeout: int = 45
) -> str:
    """
    Fetch XML content with retry logic.

    Args:
        database: NCBI database
        ids: List of article IDs
        rettype: Return type (abstract, full, etc.)
        timeout: Request timeout

    Returns:
        Raw XML response or empty string on failure
    """
    params = {
        "db": database,
        "id": ",".join(ids),
        "retmode": "xml",
        "rettype": rettype,
        "email": settings.PUBMED_EMAIL,
        "tool": "biomed-kg-agent",
    }

    url = settings.PUBMED_API_BASE + "efetch.fcgi"

    for attempt in range(MAX_RETRIES):
        try:
            # Use POST for large requests
            if len(ids) >= 200:
                response = requests.post(url, data=params, timeout=timeout)
            else:
                response = requests.get(url, params=params, timeout=timeout)

            response.raise_for_status()
            return response.text

        except requests.RequestException as e:
            logger.warning(
                f"Error fetching {database} articles (attempt {attempt + 1}/{MAX_RETRIES}): {e}"
            )

            if attempt < MAX_RETRIES - 1:
                backoff_time = INITIAL_BACKOFF * (2**attempt)
                logger.info(f"Retrying in {backoff_time} seconds...")
                time.sleep(backoff_time)
            else:
                logger.error(
                    f"Failed to fetch {database} articles after {MAX_RETRIES} attempts"
                )
                return ""

    return ""


def extract_text_safely(element: ET.Element, xpath: str, default: str = "") -> str:
    """Safely extract text from XML element with XPath."""
    try:
        elem = element.find(xpath)
        return elem.text.strip() if elem is not None and elem.text else default
    except AttributeError:
        return default


def extract_int_safely(
    element: ET.Element, xpath: str, default: int | None = None
) -> int | None:
    """Safely extract integer from XML element with XPath."""
    try:
        elem = element.find(xpath)
        if elem is not None and elem.text:
            return int(elem.text.strip())
    except (AttributeError, ValueError, TypeError):
        pass
    return default


def extract_list_safely(
    element: ET.Element,
    xpath: str,
    text_func: Callable[[ET.Element], str] = lambda x: x.text or "",
) -> list[str]:
    """Safely extract list of text values from XML elements."""
    try:
        elements = element.findall(xpath)
        result = []
        for elem in elements:
            text = text_func(elem)
            if text and text.strip():
                result.append(text.strip())
        return result
    except (AttributeError, TypeError):
        return []
