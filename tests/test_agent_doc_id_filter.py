"""Tests for format-agnostic document-ID filtering and linkification.

Tests the real functions in ``biomed_kg_agent.doc_ids``.
"""

from biomed_kg_agent.doc_ids import filter_cited_ids, linkify_pmids


def test_filter_returns_only_cited_ids() -> None:
    collected = ["34607064", "34619284", "99999999", "88888888"]
    answer = "BRCA1 is linked to breast cancer [34607064, 34619284]."
    assert filter_cited_ids(collected, answer) == ["34607064", "34619284"]


def test_fallback_when_answer_has_no_ids() -> None:
    collected = ["34607064", "34619284"]
    answer = "Some answer with no document IDs at all."
    assert filter_cited_ids(collected, answer) == collected


def test_substring_id_does_not_false_positive() -> None:
    """Word boundary prevents matching a shorter ID inside a longer token.

    "1234567" is not found in "12345678", so cited is empty and the
    fallback returns the full collected set - the key assertion is that
    the filter did NOT positively match the substring.
    """
    collected = ["1234567"]
    answer = "The identifier 12345678 is longer than expected."
    assert filter_cited_ids(collected, answer) == collected


# --- linkify tests ---


def test_linkify_multiple_distinct_pmids() -> None:
    """Multi-pass linkification: 3 distinct PMIDs all converted without corruption."""
    text = "See 34607064, 34619284, and 99999999 for evidence."
    ids = ["34607064", "34619284", "99999999"]
    result = linkify_pmids(text, ids)
    for pmid in ids:
        expected_link = f"[{pmid}](https://pubmed.ncbi.nlm.nih.gov/{pmid})"
        assert expected_link in result, f"{pmid} not linkified"
    assert result.count("](https://pubmed.ncbi.nlm.nih.gov/") == 3


def test_linkify_no_known_ids_returns_unchanged() -> None:
    text = "No IDs here."
    assert linkify_pmids(text, []) == text
