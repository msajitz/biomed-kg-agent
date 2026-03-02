"""Tests for store utility functions."""

from biomed_kg_agent.kg.utils import normalize_entity_name, resolve_entity_pair
from biomed_kg_agent.nlp.models import ExtractedEntity


def test_normalize_entity_name() -> None:
    """Test entity name normalization."""
    # Test basic normalization
    assert normalize_entity_name("Glucose") == "glucose"
    assert normalize_entity_name("ASPIRIN") == "aspirin"
    assert normalize_entity_name(" p53 ") == "p53"

    # Test edge cases
    assert normalize_entity_name("") == ""
    assert normalize_entity_name("   ") == ""
    assert normalize_entity_name("COX-2") == "cox-2"
    assert normalize_entity_name("TNF-α") == "tnf-α"


def test_resolve_entity_pair() -> None:
    """Test deterministic pair ordering."""
    # Test basic ordering
    assert resolve_entity_pair("glucose", "aspirin") == ("aspirin", "glucose")
    assert resolve_entity_pair("aspirin", "glucose") == ("aspirin", "glucose")

    # Test same entity
    assert resolve_entity_pair("glucose", "glucose") == ("glucose", "glucose")

    # Test with different entity types
    assert resolve_entity_pair("C0017725", "C0004057") == ("C0004057", "C0017725")


def test_resolve_entity_id() -> None:
    """Test stable entity ID resolution."""
    from biomed_kg_agent.kg.utils import resolve_entity_id

    # Test with UMLS CUI
    entity = ExtractedEntity(
        text="glucose",
        start_pos=0,
        end_pos=7,
        source_model="test",
        entity_type="chemical",
        source_label="CHEMICAL",
        doc_id="test",
        sentence_id=0,
        sentence_text="glucose sentence",
        umls_cui="C0017725",
    )
    assert resolve_entity_id(entity) == "C0017725"

    # Test fallback to normalized text
    entity_no_umls = ExtractedEntity(
        text="Glucose ",
        start_pos=0,
        end_pos=7,
        source_model="test",
        entity_type="chemical",
        source_label="CHEMICAL",
        doc_id="test",
        sentence_id=0,
        sentence_text="Glucose test sentence",
    )
    assert resolve_entity_id(entity_no_umls) == "CUSTOM:glucose"
