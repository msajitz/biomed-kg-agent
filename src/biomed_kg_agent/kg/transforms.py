"""
Data transformation functions for knowledge graph construction.

This module provides functions to convert processed documents from the NLP
pipeline into normalized knowledge graph models.
"""

import logging

from biomed_kg_agent.core.models import DocumentInternal

from .models import Entity, Mention
from .utils import normalize_entity_name, resolve_entity_id

logger = logging.getLogger(__name__)


def extract_entities_and_mentions(
    docs: list[DocumentInternal],
) -> tuple[list[Entity], list[Mention]]:
    """Extract entities and mentions from processed documents.

    Entity Storage Strategy:
    - Each unique entity_id creates exactly one Entity record
    - Entity.name stores UMLS preferred term if available, else first occurrence
    - All text variations are preserved in the Mention table for detailed analysis

    Args:
        docs: List of processed DocumentInternal objects

    Returns:
        Tuple of (entities, mentions) for knowledge graph construction
    """
    entities: dict[str, Entity] = {}
    mentions: list[Mention] = []

    for doc in docs:
        for extracted_entity in doc.entities:
            # Filter out <= 2 character entities to remove NER false positives
            # (e.g., "K" from acronym splitting, punctuation misclassification)
            # Design decision: ~2% of entities filtered, significantly improves KG quality
            # See docs/validation.md "Entity Filtering" section
            if len(extracted_entity.text.strip()) <= 2:
                continue

            # Create normalized entity (only on first occurrence of entity_id)
            entity_id = resolve_entity_id(extracted_entity)

            if entity_id not in entities:
                # Store colloquial name for human-readable queries
                entity_name = normalize_entity_name(extracted_entity.text)

                entities[entity_id] = Entity(
                    id=entity_id,
                    name=entity_name,
                    entity_type=extracted_entity.entity_type,
                    umls_cui=extracted_entity.umls_cui,
                    umls_preferred_name=extracted_entity.umls_preferred_name,
                    chebi_id=extracted_entity.chebi_id,
                    go_id=extracted_entity.go_id,
                    mesh_id=extracted_entity.mesh_id,
                )

            # Create lightweight mention (every occurrence)
            mentions.append(
                Mention(
                    doc_id=doc.id,
                    entity_id=entity_id,
                    text=extracted_entity.text,  # Original text preserved
                    sentence_id=extracted_entity.sentence_id,
                    sentence_text=extracted_entity.sentence_text,  # Full sentence context
                    start_pos=extracted_entity.start_pos,
                    end_pos=extracted_entity.end_pos,
                    source_label=extracted_entity.source_label,
                )
            )

    logger.info(f"Created {len(entities)} unique entities and {len(mentions)} mentions")
    return list(entities.values()), mentions
