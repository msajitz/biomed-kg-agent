"""Tests for biomedical NER module.

This module tests ONLY the pure NER functionality of BiomedicalNER class.
It does NOT test entity linking or pipeline integration.
"""

import pytest


def test_deduplicate_entities() -> None:
    """Test entity deduplication logic with mocked models."""
    from unittest.mock import patch

    from biomed_kg_agent.nlp.biomedical_ner import BiomedicalNER
    from biomed_kg_agent.nlp.models import ExtractedEntity

    # Mock the model loading to avoid downloading large models
    with patch.object(BiomedicalNER, "__init__", lambda self, **kwargs: None):
        ner = BiomedicalNER.__new__(BiomedicalNER)

        # Set up mock models with priorities
        ner.models = {
            "bc5cdr": {"name": "en_ner_bc5cdr_md", "priority": 3},
            "bionlp": {"name": "en_ner_bionlp13cg_md", "priority": 2},
            "craft": {"name": "en_ner_craft_md", "priority": 1},
        }

        # Create test entities with duplicates and overlaps
        entities = [
            # Exact duplicates - different models, same text and position
            ExtractedEntity(
                text="glucose",
                start_pos=0,
                end_pos=7,
                source_model="en_ner_bc5cdr_md",
                entity_type="chemical",
                source_label="CHEMICAL",
                doc_id="test",
                sentence_id=0,
                sentence_text="glucose mentioned",
            ),
            ExtractedEntity(
                text="glucose",
                start_pos=0,
                end_pos=7,
                source_model="en_ner_bionlp13cg_md",
                entity_type="chemical",
                source_label="SIMPLE_CHEMICAL",
                doc_id="test",
                sentence_id=0,
                sentence_text="glucose mentioned",
            ),
            # Case/whitespace variants - should be deduplicated
            ExtractedEntity(
                text="Glucose ",
                start_pos=10,
                end_pos=18,
                source_model="en_ner_craft_md",
                entity_type="chemical",
                source_label="CHEMICAL",
                doc_id="test",
                sentence_id=0,
                sentence_text="Glucose around here",
            ),
            ExtractedEntity(
                text=" glucose",
                start_pos=9,
                end_pos=17,
                source_model="en_ner_bc5cdr_md",
                entity_type="chemical",
                source_label="CHEBI",
                doc_id="test",
                sentence_id=0,
                sentence_text=" glucose nearby",
            ),
            # Different entities - should not be deduplicated
            ExtractedEntity(
                text="aspirin",
                start_pos=20,
                end_pos=27,
                source_model="en_ner_bc5cdr_md",
                entity_type="chemical",
                source_label="CHEMICAL",
                doc_id="test",
                sentence_id=0,
                sentence_text="aspirin present",
            ),
            # Close positions but different text - should not be deduplicated
            ExtractedEntity(
                text="cancer",
                start_pos=30,
                end_pos=36,
                source_model="en_ner_bionlp13cg_md",
                entity_type="disease",
                source_label="DISEASE",
                doc_id="test",
                sentence_id=0,
                sentence_text="cancer mention",
            ),
        ]

        # Test deduplication
        deduplicated = ner._deduplicate_entities(entities)

        # Should have 5 entities:
        # - bucket 0: 1 glucose (bc5cdr wins over bionlp)
        # - bucket 1: 1 glucose (pos 9, only one entity)
        # - bucket 2: 1 glucose (pos 10, only one entity)
        # - bucket 4: 1 aspirin
        # - bucket 6: 1 cancer
        assert len(deduplicated) == 5

        # Check that highest priority models were selected for duplicates
        glucose_entities = [
            e for e in deduplicated if e.text.lower().strip() == "glucose"
        ]
        assert len(glucose_entities) == 3  # Three position buckets: 0, 1, 2

        # Bucket 0 (pos 0): bc5cdr should win over bionlp
        bucket_0_glucose = next(e for e in glucose_entities if e.start_pos // 5 == 0)
        assert bucket_0_glucose.source_model == "en_ner_bc5cdr_md"

        # Bucket 1 (pos 9): only bc5cdr
        bucket_1_glucose = next(e for e in glucose_entities if e.start_pos // 5 == 1)
        assert bucket_1_glucose.source_model == "en_ner_bc5cdr_md"

        # Bucket 2 (pos 10): only craft
        bucket_2_glucose = next(e for e in glucose_entities if e.start_pos // 5 == 2)
        assert bucket_2_glucose.source_model == "en_ner_craft_md"

        # Other entities should be preserved
        assert any(e.text == "aspirin" for e in deduplicated)
        assert any(e.text == "cancer" for e in deduplicated)


def test_release_models_clears_shared_cache() -> None:
    """Test that release_models() clears the class-level _shared_models cache."""
    from unittest.mock import MagicMock

    from biomed_kg_agent.nlp.biomedical_ner import BiomedicalNER

    # Populate the shared cache with lightweight fakes
    BiomedicalNER._shared_models = {
        "bc5cdr": {"nlp": MagicMock(), "name": "en_ner_bc5cdr_md"},
        "bionlp": {"nlp": MagicMock(), "name": "en_ner_bionlp13cg_md"},
    }
    assert len(BiomedicalNER._shared_models) == 2

    BiomedicalNER.release_models()

    assert BiomedicalNER._shared_models == {}


@pytest.mark.slow
def test_biomedical_ner_extract_entities() -> None:
    """Test extract_entities method with real models."""
    from biomed_kg_agent.nlp.biomedical_ner import BiomedicalNER

    ner = BiomedicalNER()  # Uses default config

    # Test entity extraction
    text = "Glucose and aspirin are metabolized in liver cells"
    entities = ner.extract_entities(text, "test_doc")

    # Should return categorized entities
    assert isinstance(entities, dict)
    assert len(entities) > 0

    # Should have extracted entities with expected structure
    for category, entity_list in entities.items():
        assert isinstance(category, str)
        assert isinstance(entity_list, list)

        if entity_list:  # If entities found in this category
            entity = entity_list[0]
            assert hasattr(entity, "text")
            assert hasattr(entity, "entity_type")
            assert hasattr(entity, "start_pos")
            assert hasattr(entity, "end_pos")
            assert hasattr(entity, "source_model")
            assert entity.doc_id == "test_doc"
