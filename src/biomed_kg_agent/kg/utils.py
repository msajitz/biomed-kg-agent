"""
Utility functions for knowledge graph operations.

This module provides helper functions for entity ID generation, pair ordering,
and other KG-related operations.
"""

from biomed_kg_agent.nlp.models import ExtractedEntity


def normalize_entity_name(text: str) -> str:
    """Normalize entity display text for consistency.

    Lowercase and strip whitespace; keep unicode characters as-is.
    """
    return text.lower().strip()


def resolve_entity_id(extracted_entity: ExtractedEntity) -> str:
    """Resolve a stable entity identifier with priority: UMLS > ChEBI > GO > MeSH > normalized text.

    This provides a canonical, source-agnostic identity across mention variants.
    """
    if extracted_entity.umls_cui:
        return extracted_entity.umls_cui
    if extracted_entity.chebi_id:
        return extracted_entity.chebi_id
    if extracted_entity.go_id:
        return extracted_entity.go_id
    if extracted_entity.mesh_id:
        return extracted_entity.mesh_id
    # Fallback: normalized text
    return f"CUSTOM:{normalize_entity_name(extracted_entity.text)}"


def resolve_entity_pair(entity_a: str, entity_b: str) -> tuple[str, str]:
    """Return a deterministic (a, b) ordering by lexicographic comparison.

    This prevents duplicate rows for a↔b pairs.
    """
    return (entity_a, entity_b) if entity_a < entity_b else (entity_b, entity_a)
