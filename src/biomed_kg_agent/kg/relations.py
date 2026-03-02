"""
Relation extraction functions for knowledge graph construction.

This module provides functions to extract relationships between entities,
starting with sentence-level co-occurrence relationships.
"""

import json
import logging
from typing import Any, TypedDict

from .models import Cooccurrence, Mention
from .utils import resolve_entity_pair


class CooccurrenceData(TypedDict):
    """Typed structure for tracking co-occurrence data during extraction."""

    count: int
    sentence_contexts: list[tuple[str, str]]  # (doc_id, sentence_text)


logger = logging.getLogger(__name__)


def extract_cooccurrences(
    mentions: list[Mention], max_evidence_sentences: int = 5
) -> list[Cooccurrence]:
    """Extract sentence-level co-occurrences from mentions.

    Note: This function extracts ALL co-occurrences without filtering.
    Filtering should be applied later (e.g., in notebooks for experimentation
    or during Neo4j migration using apply_cooccurrence_filters()).

    Evidence sentences: Selects diverse evidence sentences per relationship,
    prioritizing different documents and sentence positions to provide varied contexts
    for downstream agent reasoning.

    Args:
        mentions: List of entity mentions
        max_evidence_sentences: Maximum number of evidence sentences to collect per pair
            (default: 5)

    Returns:
        List of all co-occurrence statistics (unfiltered), with evidence sentences
    """
    pairs: dict[tuple[str, str], CooccurrenceData] = {}

    # Group by document + sentence, preserving sentence text
    sentence_groups: dict[tuple[str, int], dict[str, Any]] = {}
    for mention in mentions:
        key = (mention.doc_id, mention.sentence_id)
        if key not in sentence_groups:
            sentence_groups[key] = {"entities": [], "text": mention.sentence_text}
        sentence_groups[key]["entities"].append(mention.entity_id)

    # Count co-occurrences and collect sentence contexts
    for (doc_id, sent_id), group in sentence_groups.items():
        entities_list, sentence_text = group["entities"], group["text"]
        unique_entities = list(set(entities_list))
        for i, entity_a in enumerate(unique_entities):
            for entity_b in unique_entities[i + 1 :]:
                pair_key = resolve_entity_pair(entity_a, entity_b)
                if pair_key not in pairs:
                    pairs[pair_key] = {"count": 0, "sentence_contexts": []}
                pairs[pair_key]["count"] += 1
                pairs[pair_key]["sentence_contexts"].append((doc_id, sentence_text))

    # Build cooccurrences with diverse evidence sentences
    cooccurrences = []
    for pair, data in pairs.items():
        # Select diverse evidence contexts (doc_id, text)
        evidence_contexts = _select_diverse_evidence(
            data["sentence_contexts"],
            max_sentences=max_evidence_sentences,
        )

        # Extract aligned sentences and doc_ids from selected contexts
        # Keep parallel arrays: evidence_sentences[i] is from evidence_doc_ids[i]
        evidence_sentences = [ctx[1] for ctx in evidence_contexts]
        evidence_doc_ids = [ctx[0] for ctx in evidence_contexts]

        # Count unique documents from sentence contexts
        unique_docs = {ctx[0] for ctx in data["sentence_contexts"]}

        cooccurrences.append(
            Cooccurrence(
                entity_a_id=pair[0],
                entity_b_id=pair[1],
                sent_count=data["count"],
                docs_count=len(unique_docs),
                doc_ids_sample=(
                    json.dumps(evidence_doc_ids) if evidence_doc_ids else None
                ),
                evidence_sentences=(
                    json.dumps(evidence_sentences) if evidence_sentences else None
                ),
            )
        )

    logger.info(f"Extracted {len(cooccurrences)} co-occurrence pairs with evidence")
    return cooccurrences


def _select_diverse_evidence(
    contexts: list[tuple[str, str]], max_sentences: int = 5
) -> list[tuple[str, str]]:
    """Select diverse evidence sentences from co-occurrence contexts.

    Strategy: Prioritize different documents via round-robin selection
    to provide varied contexts for downstream reasoning.

    Args:
        contexts: List of (doc_id, sentence_text) tuples
        max_sentences: Maximum number of sentences to select

    Returns:
        List of selected contexts (doc_id, sentence_text) (up to max_sentences)
    """
    if not contexts:
        return []

    # Group by document to ensure document diversity
    by_doc: dict[str, list[str]] = {}
    for doc_id, sentence_text in contexts:
        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(sentence_text)

    # Round-robin selection across documents
    selected: list[tuple[str, str]] = []
    doc_ids = sorted(by_doc.keys())  # Stable ordering
    doc_index = 0

    while len(selected) < max_sentences and doc_ids:
        doc_id = doc_ids[doc_index]
        if by_doc[doc_id]:
            # Take first remaining sentence from this document
            sentence_text = by_doc[doc_id].pop(0)
            selected.append((doc_id, sentence_text))

        # If document exhausted, remove from rotation
        if not by_doc[doc_id]:
            doc_ids.pop(doc_index)
            if doc_ids:
                doc_index = doc_index % len(doc_ids)
        else:
            doc_index = (doc_index + 1) % len(doc_ids)

    return selected
