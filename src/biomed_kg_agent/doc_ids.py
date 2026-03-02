"""Document-ID helpers: format-agnostic citation filtering and PubMed linkification.

Pure ``re`` dependency - no Streamlit, Neo4j, or LLM imports.
"""

import re

PUBMED_URL_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{doc_id}"


def filter_cited_ids(collected: list[str], answer: str) -> list[str]:
    """Return only *collected* IDs that the LLM actually cited in *answer*.

    Detection is format-agnostic: each known ID is checked for word-boundary
    presence via ``re.search``.  Falls back to the full *collected* list when
    no known IDs appear (preserves behaviour when the LLM omits citations).
    """
    cited = {d for d in collected if re.search(rf"\b{re.escape(d)}\b", answer)}
    if cited:
        return [d for d in collected if d in cited]
    return collected


def linkify_pmids(text: str, known_doc_ids: list[str]) -> str:
    """Replace known document IDs in *text* with clickable PubMed links.

    Iterates over *known_doc_ids* and performs one ``re.sub`` per ID using
    word boundaries so that IDs embedded in longer tokens are not matched.
    """
    if not known_doc_ids:
        return text
    for doc_id in known_doc_ids:
        url = PUBMED_URL_TEMPLATE.format(doc_id=doc_id)
        text = re.sub(
            rf"\b{re.escape(doc_id)}\b",
            f"[{doc_id}]({url})",
            text,
        )
    return text
