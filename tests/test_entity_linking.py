"""Tests for entity linking module.

This module tests ONLY the EntityLinker class and UMLS linking functionality.
It does NOT test NER extraction or two-pass processing.
"""

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from biomed_kg_agent.nlp.config import LinkerConfig
from biomed_kg_agent.nlp.entity_linking import EntityLinker, UMLSMapping
from tests.mocks.biomedical_ner import create_mock_entity


def test_apply_umls_mappings() -> None:
    """Test the static apply_umls_mappings method."""
    # Create mock entities
    entities_by_entity_type = {
        "chemical": [
            create_mock_entity("Glucose", "CHEMICAL", 0, 7),
            create_mock_entity("Aspirin", "CHEMICAL", 10, 17),
        ],
        "disease": [
            create_mock_entity("cancer", "DISEASE", 20, 26),
        ],
    }

    # Create mock UMLS mappings
    umls_mappings: dict[str, UMLSMapping] = {
        "Glucose": {
            "umls_cui": "C0017725",
            "linking_confidence": 0.95,
            "umls_preferred_name": "Glucose",
            "umls_semantic_types": None,
        },
        "Aspirin": {
            "umls_cui": "C0004057",
            "linking_confidence": 0.89,
            "umls_preferred_name": "Aspirin",
            "umls_semantic_types": None,
        },
        "cancer": {
            "umls_cui": None,
            "linking_confidence": 0.45,
            "umls_preferred_name": None,
            "umls_semantic_types": None,
        },  # Below threshold
    }

    # Apply mappings
    linked_entities = EntityLinker.apply_umls_mappings(
        entities_by_entity_type, umls_mappings
    )

    # Verify structure is preserved
    assert "chemical" in linked_entities
    assert "disease" in linked_entities
    assert len(linked_entities["chemical"]) == 2
    assert len(linked_entities["disease"]) == 1

    # Verify UMLS CUIDs and preferred names are applied correctly
    glucose_entity = linked_entities["chemical"][0]
    aspirin_entity = linked_entities["chemical"][1]
    cancer_entity = linked_entities["disease"][0]

    assert glucose_entity.umls_cui == "C0017725"
    assert glucose_entity.linking_confidence == 0.95
    assert glucose_entity.umls_preferred_name == "Glucose"

    assert aspirin_entity.umls_cui == "C0004057"
    assert aspirin_entity.linking_confidence == 0.89
    assert aspirin_entity.umls_preferred_name == "Aspirin"

    # Cancer should have no CUI due to low confidence but should have confidence stored
    assert cancer_entity.umls_cui is None
    assert cancer_entity.linking_confidence == 0.45
    assert cancer_entity.umls_preferred_name is None


def test_entity_linker_initialization() -> None:
    """Test EntityLinker initialization with different configs."""
    # Test with enabled linking
    enabled_config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(enabled_config)
    assert linker.linker_config.enabled is True
    assert linker.linker_config.confidence_threshold == 0.7

    # Test with disabled linking
    disabled_config = LinkerConfig(
        enabled=False, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker_disabled = EntityLinker(disabled_config)
    assert linker_disabled.linker_config.enabled is False


def test_apply_umls_mappings_empty_input() -> None:
    """Test apply_umls_mappings with empty inputs."""
    # Empty entities
    result = EntityLinker.apply_umls_mappings({}, {})
    assert result == {}

    # Empty mappings
    entities = {"chemical": [create_mock_entity("Glucose", "CHEMICAL", 0, 7)]}
    result = EntityLinker.apply_umls_mappings(entities, {})
    assert "chemical" in result
    assert len(result["chemical"]) == 1
    assert result["chemical"][0].umls_cui is None


def test_apply_umls_mappings_preserves_original_data() -> None:
    """
    Test that apply_umls_mappings modifies entities in-place
    (memory-efficient for large corpora).
    """
    original_entity = create_mock_entity("Glucose", "CHEMICAL", 0, 7)
    entities = {"chemical": [original_entity]}

    mappings: dict[str, UMLSMapping] = {
        "Glucose": {
            "umls_cui": "C0017725",
            "linking_confidence": 0.95,
            "umls_preferred_name": "Glucose",
            "umls_semantic_types": None,
        }
    }

    result = EntityLinker.apply_umls_mappings(entities, mappings)

    # Entities are now modified IN-PLACE for memory efficiency (prevents OOM on 77K+ entity corpora)
    # The original entity should have UMLS mapping applied
    assert original_entity.umls_cui == "C0017725"
    assert original_entity.linking_confidence == 0.95
    assert original_entity.umls_preferred_name == "Glucose"

    # Result contains references to the same (now modified) entities
    assert result["chemical"][0] is original_entity
    assert result["chemical"][0].umls_cui == "C0017725"
    assert result["chemical"][0].linking_confidence == 0.95
    assert result["chemical"][0].umls_preferred_name == "Glucose"


def test_apply_umls_mappings_confidence_threshold_logic() -> None:
    """Test confidence threshold logic in apply_umls_mappings (fast unit test)."""
    entities = {
        "chemical": [
            create_mock_entity("HighConf", "CHEMICAL", 0, 8),
            create_mock_entity("LowConf", "CHEMICAL", 10, 17),
        ]
    }

    # Mock mappings with different confidence levels
    mappings: dict[str, UMLSMapping] = {
        "HighConf": {
            "umls_cui": "C0001234",
            "linking_confidence": 0.95,
            "umls_preferred_name": "High Confidence Entity",
            "umls_semantic_types": None,
        },  # Above threshold
        "LowConf": {
            "umls_cui": None,
            "linking_confidence": 0.45,
            "umls_preferred_name": None,
            "umls_semantic_types": None,
        },  # Below threshold (filtered out by linker)
    }

    result = EntityLinker.apply_umls_mappings(entities, mappings)

    # High confidence should have CUI
    assert result["chemical"][0].umls_cui == "C0001234"
    assert result["chemical"][0].linking_confidence == 0.95

    # Low confidence should have None CUI (already filtered by linker)
    assert result["chemical"][1].umls_cui is None
    assert result["chemical"][1].linking_confidence == 0.45


def test_entity_linker_disabled_config() -> None:
    """Test EntityLinker behavior when disabled (fast unit test)."""
    disabled_config = LinkerConfig(
        enabled=False, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(disabled_config)

    # _get_umls_linker should return None when disabled
    assert linker._get_umls_linker() is None

    # link_entities should return None mappings when disabled
    entity_texts = {"glucose", "aspirin"}
    mappings = linker.link_entities(entity_texts)

    assert len(mappings) == 2
    for entity_text in entity_texts:
        assert entity_text in mappings
        assert mappings[entity_text]["umls_cui"] is None
        assert mappings[entity_text]["linking_confidence"] is None
        assert mappings[entity_text]["umls_preferred_name"] is None


# Integration tests with real models (marked as slow)


@pytest.mark.slow
def test_get_umls_linker_integration() -> None:
    """Integration test for _get_umls_linker with real spaCy models."""
    # Test with enabled configuration
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(config)

    # This will load real models
    nlp = linker._get_umls_linker()

    # Verify model loaded successfully
    assert nlp is not None
    assert "scispacy_linker" in nlp.pipe_names

    # Test disabled configuration
    disabled_config = LinkerConfig(
        enabled=False, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    disabled_linker = EntityLinker(disabled_config)

    # Should return None when disabled
    assert disabled_linker._get_umls_linker() is None


@pytest.mark.slow
def test_link_entities_integration() -> None:
    """Integration test for link_entities with real UMLS linking."""
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.6
    )
    linker = EntityLinker(config)

    # Test with known biomedical entities
    entity_texts = {"glucose", "aspirin", "cancer"}

    # This will perform real UMLS linking
    mappings = linker.link_entities(entity_texts)

    # Verify all entities were processed
    assert len(mappings) == 3
    assert "glucose" in mappings
    assert "aspirin" in mappings
    assert "cancer" in mappings

    # Check that mappings have expected structure
    for entity_text, mapping in mappings.items():
        assert "umls_cui" in mapping
        assert "linking_confidence" in mapping

        # Some entities should have valid CUIDs (above threshold)
        if mapping["umls_cui"] is not None:
            assert isinstance(mapping["umls_cui"], str)
            assert mapping["umls_cui"].startswith("C")  # UMLS CUIDs start with C

        # Confidence should be a float or None
        if mapping["linking_confidence"] is not None:
            assert isinstance(mapping["linking_confidence"], float)
            assert 0.0 <= mapping["linking_confidence"] <= 1.0


# NEW TESTS FOR MEMORY-EFFICIENT CHUNKED UMLS LINKING


@pytest.fixture
def mock_umls_linker() -> Any:
    """Mock UMLS linker for testing chunked processing."""
    with patch("biomed_kg_agent.nlp.entity_linking.spacy.load") as mock_spacy_load:
        mock_nlp = MagicMock()
        mock_spacy_load.return_value = mock_nlp

        # Mock the scispacy_linker pipe
        mock_nlp.pipe_names = ["scispacy_linker"]

        # Mock the processing of documents
        def mock_pipe_process(texts: list) -> list:
            """Mock the spaCy pipe processing."""
            docs = []
            for text in texts:
                doc = MagicMock()
                doc.text = text

                # Mock entities with UMLS linking
                if "glucose" in text.lower():
                    ent = MagicMock()
                    ent.text = "glucose"
                    ent._.kb_ents = [("C0017725", 0.95)]  # High confidence
                    doc.ents = [ent]
                elif "aspirin" in text.lower():
                    ent = MagicMock()
                    ent.text = "aspirin"
                    ent._.kb_ents = [("C0004057", 0.89)]  # High confidence
                    doc.ents = [ent]
                elif "cancer" in text.lower():
                    ent = MagicMock()
                    ent.text = "cancer"
                    ent._.kb_ents = [("C0006826", 0.45)]  # Low confidence
                    doc.ents = [ent]
                else:
                    doc.ents = []  # No entities

                docs.append(doc)
            return docs

        mock_nlp.pipe.side_effect = mock_pipe_process
        yield mock_nlp


@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._get_umls_linker")
def test_link_entities_chunked_processing_small_set(
    mock_get_linker: Any, mock_umls_linker: Any
) -> None:
    """Test chunked processing with a small set of entities (< chunk_size)."""
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )

    mock_get_linker.return_value = mock_umls_linker

    linker = EntityLinker(config)

    # Small set of entities (should be processed in single chunk)
    entity_texts = {"glucose", "aspirin", "cancer"}

    mappings = linker.link_entities(entity_texts)

    # Verify results
    assert len(mappings) == 3
    assert mappings["glucose"]["umls_cui"] == "C0017725"
    assert mappings["aspirin"]["umls_cui"] == "C0004057"
    assert mappings["cancer"]["umls_cui"] is None  # Below threshold


@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._process_entity_chunk")
@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._get_umls_linker")
def test_link_entities_chunked_processing_large_set(
    mock_get_linker: Any, mock_process_chunk: Any
) -> None:
    """Test chunked processing with a large set of entities (> chunk_size)."""
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )

    # Create a large set of entities (15,000 entities to test chunking)
    entity_texts = {f"entity_{i}" for i in range(15000)}

    mock_get_linker.return_value = MagicMock()

    # Mock chunk processing to return dummy mappings
    def mock_chunk_side_effect(
        nlp: Any, chunk: list
    ) -> dict:  # nlp comes first in the actual method signature
        return {
            entity: {"umls_cui": f"C{i:07d}", "linking_confidence": 0.8}
            for i, entity in enumerate(chunk)
        }

    mock_process_chunk.side_effect = mock_chunk_side_effect

    linker = EntityLinker(config)
    mappings = linker.link_entities(entity_texts)

    # Verify chunking occurred
    # 15,000 entities should be processed in 2 chunks: 10,000 + 5,000
    assert mock_process_chunk.call_count == 2

    # Verify all entities were processed
    assert len(mappings) == 15000

    # Verify chunk sizes
    call_args_list = mock_process_chunk.call_args_list
    assert (
        len(call_args_list[0][0][1]) == 10000
    )  # First chunk (second argument is the chunk)
    assert len(call_args_list[1][0][1]) == 5000  # Second chunk


@patch("biomed_kg_agent.nlp.entity_linking.gc.collect")
@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._process_entity_chunk")
@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._get_umls_linker")
def test_link_entities_chunked_processing_memory_cleanup(
    mock_get_linker: Any, mock_process_chunk: Any, mock_gc_collect: Any
) -> None:
    """Test that garbage collection is called between chunks for memory cleanup."""
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )

    # Create entities that will require 2 chunks
    entity_texts = {f"entity_{i}" for i in range(15000)}

    mock_get_linker.return_value = MagicMock()

    # Mock to return proper mappings for each entity
    def mock_chunk_side_effect(nlp: Any, chunk: list) -> dict:
        return {
            entity: {"umls_cui": None, "linking_confidence": None} for entity in chunk
        }

    mock_process_chunk.side_effect = mock_chunk_side_effect

    linker = EntityLinker(config)
    linker.link_entities(entity_texts)

    # Garbage collection should be called between chunks
    # Should be called at least once (after first chunk)
    assert mock_gc_collect.call_count >= 1


@patch("biomed_kg_agent.nlp.entity_linking.logger")
@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._process_entity_chunk")
@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._get_umls_linker")
def test_link_entities_chunked_processing_progress_logging(
    mock_get_linker: Any, mock_process_chunk: Any, mock_logger: Any
) -> None:
    """Test that progress is logged during chunked processing."""
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )

    # Create entities that will require multiple chunks
    entity_texts = {f"entity_{i}" for i in range(25000)}  # 3 chunks

    mock_get_linker.return_value = MagicMock()

    # Mock to return proper mappings for each entity
    def mock_chunk_side_effect(nlp: Any, chunk: list) -> dict:
        return {
            entity: {"umls_cui": None, "linking_confidence": None} for entity in chunk
        }

    mock_process_chunk.side_effect = mock_chunk_side_effect

    linker = EntityLinker(config)
    linker.link_entities(entity_texts)

    # Should log progress for each chunk
    progress_calls = [
        call
        for call in mock_logger.info.call_args_list
        if "Processing chunk" in str(call)
    ]
    assert len(progress_calls) == 3  # 3 chunks


@patch("biomed_kg_agent.nlp.entity_linking.gc.collect")
def test_process_entity_chunk_batch_processing(mock_gc: Any) -> None:
    """Test _process_entity_chunk processes entities in small batches."""
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(config)

    # Create a chunk larger than batch size (50) and GC threshold (200)
    chunk = [
        f"entity_{i}" for i in range(250)
    ]  # Should be 5 batches: 50, 50, 50, 50, 50

    mock_nlp = MagicMock()

    # Mock the pipe processing
    def mock_pipe_process(texts: list) -> list:
        docs = []
        for text in texts:
            doc = MagicMock()
            doc.text = text
            doc.ents = []  # No entities for simplicity
            docs.append(doc)
        return docs

    mock_nlp.pipe.side_effect = mock_pipe_process

    result = linker._process_entity_chunk(mock_nlp, chunk)

    # Should process all entities (even with no UMLS matches, should return None mappings)
    assert len(result) == 250

    # All entities should have mappings (even if None)
    for entity in chunk:
        assert entity in result
        assert "umls_cui" in result[entity]
        assert "linking_confidence" in result[entity]

    # Should call pipe multiple times (for batches)
    assert mock_nlp.pipe.call_count == 5

    # Should call garbage collection every 200 entities
    # With 250 entities: GC after entity 200 (batch 4 completes)
    assert mock_gc.call_count == 1


@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._process_entity_chunk")
def test_link_entities_disabled_chunking(mock_process_chunk: Any) -> None:
    """Test that chunking is bypassed when UMLS linking is disabled."""
    config = LinkerConfig(
        enabled=False, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(config)

    # Large set of entities
    entity_texts = {f"entity_{i}" for i in range(15000)}

    mappings = linker.link_entities(entity_texts)

    # Should not call chunking when disabled
    mock_process_chunk.assert_not_called()

    # Should return empty mappings
    assert len(mappings) == 15000
    for mapping in mappings.values():
        assert mapping["umls_cui"] is None
        assert mapping["linking_confidence"] is None


@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._process_entity_chunk")
@patch("biomed_kg_agent.nlp.entity_linking.EntityLinker._get_umls_linker")
def test_link_entities_chunk_size_boundary(
    mock_get_linker: Any, mock_process_chunk: Any
) -> None:
    """Test chunking behavior at exact chunk size boundaries."""
    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )

    mock_get_linker.return_value = MagicMock()

    # Mock to return proper mappings for each entity
    def mock_chunk_side_effect(nlp: Any, chunk: list) -> dict:
        return {
            entity: {"umls_cui": None, "linking_confidence": None} for entity in chunk
        }

    mock_process_chunk.side_effect = mock_chunk_side_effect

    # Test with exactly chunk_size entities (10,000)
    entity_texts = {f"entity_{i}" for i in range(10000)}

    linker = EntityLinker(config)
    linker.link_entities(entity_texts)

    # Should process in exactly 1 chunk
    assert mock_process_chunk.call_count == 1
    assert (
        len(mock_process_chunk.call_args_list[0][0][1]) == 10000
    )  # Second argument is the chunk

    # Reset mocks for second test
    mock_process_chunk.reset_mock()

    # Test with chunk_size + 1 entities (10,001)
    entity_texts = {f"entity_{i}" for i in range(10001)}

    linker.link_entities(entity_texts)

    # Should process in 2 chunks: 10,000 + 1
    assert mock_process_chunk.call_count == 2
    assert (
        len(mock_process_chunk.call_args_list[0][0][1]) == 10000
    )  # First chunk (second argument)
    assert (
        len(mock_process_chunk.call_args_list[1][0][1]) == 1
    )  # Second chunk (second argument)


def test_pattern_based_corrections() -> None:
    """Test pattern-based entity type corrections via production path (apply_umls_mappings)."""
    from biomed_kg_agent.nlp.models import ExtractedEntity

    # Test inhibitor correction
    entities = {
        "gene": [
            ExtractedEntity(
                text="CDK4/6 Inhibition",
                entity_type="gene",
                start_pos=0,
                end_pos=17,
                source_label="GENE_OR_GENE_PRODUCT",
                sentence_id=0,
                sentence_text="CDK4/6 Inhibition is a therapeutic approach.",
            )
        ]
    }
    # Empty UMLS mappings to test pattern corrections alone
    corrected = EntityLinker.apply_umls_mappings(entities, {})
    assert "chemical" in corrected
    assert corrected["chemical"][0].text == "CDK4/6 Inhibition"

    # Test antibody correction
    entities = {
        "gene": [
            ExtractedEntity(
                text="Tumor Antigen-specific Antibody",
                entity_type="gene",
                start_pos=0,
                end_pos=31,
                source_label="GENE_OR_GENE_PRODUCT",
                sentence_id=0,
                sentence_text="Tumor Antigen-specific Antibody targets cancer cells.",
            )
        ]
    }
    corrected = EntityLinker.apply_umls_mappings(entities, {})
    assert "chemical" in corrected
    assert corrected["chemical"][0].text == "Tumor Antigen-specific Antibody"

    # Test syndrome correction
    entities = {
        "gene": [
            ExtractedEntity(
                text="Hamartoma Syndrome, Multiple",
                entity_type="gene",
                start_pos=0,
                end_pos=28,
                source_label="GENE_OR_GENE_PRODUCT",
                sentence_id=0,
                sentence_text="Hamartoma Syndrome, Multiple is a genetic disorder.",
            )
        ]
    }
    corrected = EntityLinker.apply_umls_mappings(entities, {})
    assert "disease" in corrected
    assert corrected["disease"][0].text == "Hamartoma Syndrome, Multiple"

    # Test inhibitor activity NOT corrected (biological process exception)
    entities = {
        "gene": [
            ExtractedEntity(
                text="ATPase inhibitor activity",
                entity_type="gene",
                start_pos=0,
                end_pos=24,
                source_label="GENE_OR_GENE_PRODUCT",
                sentence_id=0,
                sentence_text="ATPase inhibitor activity is measured in cells.",
            )
        ]
    }
    corrected = EntityLinker.apply_umls_mappings(entities, {})
    assert "gene" in corrected  # Not changed due to 'activity' exception
    assert corrected["gene"][0].text == "ATPase inhibitor activity"

    # Test multiple entities in same group
    entities = {
        "gene": [
            ExtractedEntity(
                text="ALK Inhibitor",
                entity_type="gene",
                start_pos=0,
                end_pos=13,
                source_label="GENE_OR_GENE_PRODUCT",
                sentence_id=0,
                sentence_text="ALK Inhibitor and BRCA1 gene are studied.",
            ),
            ExtractedEntity(
                text="BRCA1 gene",
                entity_type="gene",
                start_pos=20,
                end_pos=30,
                source_label="GENE_OR_GENE_PRODUCT",
                sentence_id=0,
                sentence_text="ALK Inhibitor and BRCA1 gene are studied.",
            ),
        ]
    }
    corrected = EntityLinker.apply_umls_mappings(entities, {})
    assert "chemical" in corrected  # ALK Inhibitor corrected
    assert len(corrected["chemical"]) == 1
    assert corrected["chemical"][0].text == "ALK Inhibitor"
    assert "gene" in corrected  # BRCA1 gene unchanged
    assert len(corrected["gene"]) == 1
    assert corrected["gene"][0].text == "BRCA1 gene"


def test_pattern_corrections_plural_forms() -> None:
    """Regression test: plurals (inhibitors, antibodies) must match patterns.

    Bug fix (2025-11): Pattern \b(inhibitor|inhibition)\b didn't match 'Inhibitors'.
    Fixed with inhibitors? to handle both singular and plural.
    """
    from biomed_kg_agent.nlp.models import ExtractedEntity

    # High-impact case from production data: plural inhibitors misclassified
    entities = {
        "gene": [
            ExtractedEntity(
                text="Tyrosine Kinase Inhibitors",
                entity_type="gene",
                start_pos=0,
                end_pos=26,
                source_label="GENE_OR_GENE_PRODUCT",
                sentence_id=0,
                sentence_text="Tyrosine Kinase Inhibitors are used.",
            )
        ],
        "cell_type": [
            ExtractedEntity(
                text="Immune Checkpoint Inhibitors",
                entity_type="cell_type",
                start_pos=0,
                end_pos=28,
                source_label="CELL_TYPE",
                sentence_id=1,
                sentence_text="Immune Checkpoint Inhibitors boost immunity.",
            )
        ],
    }
    # Test via production path
    corrected = EntityLinker.apply_umls_mappings(entities, {})
    assert "chemical" in corrected
    assert len(corrected["chemical"]) == 2
    assert {e.text for e in corrected["chemical"]} == {
        "Tyrosine Kinase Inhibitors",
        "Immune Checkpoint Inhibitors",
    }


# ============================================================================
# DATABASE CACHING TESTS (integration tests for cache retrieval)
# ============================================================================


@pytest.fixture
def temp_db_with_schema() -> Any:
    """Fixture providing a temporary database with schema initialized."""
    import os
    import tempfile

    from sqlmodel import SQLModel, create_engine

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    engine = create_engine(f"sqlite:///{db_path}")
    SQLModel.metadata.create_all(engine)

    yield db_path

    os.unlink(db_path)


def test_cached_umls_retrieval_and_filtering(temp_db_with_schema: str) -> None:
    """Test cache retrieval: empty DB, with data, and entity filtering."""
    from biomed_kg_agent.core.models import DocumentInternal
    from biomed_kg_agent.nlp.models import ExtractedEntity
    from biomed_kg_agent.nlp.persistence import save_nlp_results

    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(config)

    # Test 1: Empty database returns empty dict
    result = linker._get_cached_umls_mappings(
        {"glucose", "aspirin"}, temp_db_with_schema
    )
    assert result == {}

    # Test 2: Retrieve cached entities (including failed links with confidence)
    from biomed_kg_agent.nlp.config import NerConfig, get_default_priorities

    doc = DocumentInternal(id="doc1", title="Test", text="test", source="test")
    doc.entities = [
        ExtractedEntity(
            text="glucose",
            entity_type="chemical",
            source_label="CHEMICAL",
            start_pos=0,
            end_pos=7,
            sentence_id=0,
            sentence_text="glucose",
            doc_id="doc1",
            umls_cui="C0017725",
            linking_confidence=0.95,
            umls_preferred_name="Glucose",
            umls_semantic_types=["T109", "T121"],
        ),
        ExtractedEntity(
            text="cancer",
            entity_type="disease",
            source_label="DISEASE",
            start_pos=8,
            end_pos=14,
            sentence_id=0,
            sentence_text="cancer",
            doc_id="doc1",
            umls_cui=None,
            linking_confidence=0.45,  # Below threshold
        ),
    ]
    ner_config = NerConfig(model_priorities=get_default_priorities(), linker=config)
    save_nlp_results([doc], ner_config, db_path=temp_db_with_schema)

    result = linker._get_cached_umls_mappings(
        {"glucose", "cancer", "not_in_db"}, temp_db_with_schema
    )
    assert len(result) == 2
    assert result["glucose"]["umls_cui"] == "C0017725"
    assert result["glucose"]["umls_semantic_types"] == ["T109", "T121"]
    assert result["cancer"]["umls_cui"] is None  # Failed link cached
    assert result["cancer"]["linking_confidence"] == 0.45
    assert "not_in_db" not in result


@pytest.mark.parametrize(
    "cache_scenario,cached_entities,query_entities,expected_cached,expected_new",
    [
        ("full_hit", ["glucose", "aspirin"], ["glucose", "aspirin"], 2, 0),
        ("partial_hit", ["glucose"], ["glucose", "aspirin"], 1, 1),
        ("no_hit", [], ["glucose", "aspirin"], 0, 2),
    ],
)
def test_link_entities_with_cache_scenarios(
    temp_db_with_schema: str,
    cache_scenario: str,
    cached_entities: list[str],
    query_entities: list[str],
    expected_cached: int,
    expected_new: int,
) -> None:
    """Test link_entities_with_cache cache hit scenarios (full/partial/none)."""
    from biomed_kg_agent.core.models import DocumentInternal
    from biomed_kg_agent.nlp.models import ExtractedEntity
    from biomed_kg_agent.nlp.persistence import save_nlp_results

    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(config)

    # Pre-populate cache
    if cached_entities:
        doc = DocumentInternal(id="doc1", title="Test", text="test", source="test")
        doc.entities = [
            ExtractedEntity(
                text=entity,
                entity_type="chemical",
                source_label="CHEMICAL",
                start_pos=0,
                end_pos=7,
                sentence_id=0,
                sentence_text=entity,
                doc_id="doc1",
                umls_cui=f"C{i:07d}",
                linking_confidence=0.9,
            )
            for i, entity in enumerate(cached_entities)
        ]
        from biomed_kg_agent.nlp.config import NerConfig, get_default_priorities

        ner_config = NerConfig(model_priorities=get_default_priorities(), linker=config)
        save_nlp_results([doc], ner_config, db_path=temp_db_with_schema)

    # Mock link_entities for new entities
    new_mappings = {
        entity: {
            "umls_cui": f"C{i:07d}",
            "linking_confidence": 0.85,
            "umls_preferred_name": entity.capitalize(),
            "umls_semantic_types": None,
        }
        for i, entity in enumerate(query_entities)
        if entity not in cached_entities
    }

    with patch.object(linker, "link_entities", return_value=new_mappings) as mock_link:
        result = linker.link_entities_with_cache(
            set(query_entities), temp_db_with_schema
        )

        # Verify cache behavior
        if expected_new == 0:
            mock_link.assert_not_called()
        else:
            mock_link.assert_called_once()
            called_entities = mock_link.call_args[0][0]
            assert len(called_entities) == expected_new

        # Verify results
        assert len(result) == len(query_entities)
        for entity in query_entities:
            assert entity in result
            assert result[entity]["umls_cui"] is not None


def test_cached_umls_large_query_chunking(temp_db_with_schema: str) -> None:
    """Test cache handles large entity sets (>800) via SQLite query chunking."""
    from biomed_kg_agent.core.models import DocumentInternal
    from biomed_kg_agent.nlp.models import ExtractedEntity
    from biomed_kg_agent.nlp.persistence import save_nlp_results

    config = LinkerConfig(
        enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
    )
    linker = EntityLinker(config)

    # Create 1000 entities (exceeds SQLITE_QUERY_CHUNK_SIZE of 800)
    doc = DocumentInternal(id="doc1", title="Test", text="test", source="test")
    doc.entities = [
        ExtractedEntity(
            text=f"entity_{i}",
            entity_type="chemical",
            source_label="CHEMICAL",
            start_pos=i * 10,
            end_pos=i * 10 + 8,
            sentence_id=0,
            sentence_text=f"entity_{i}",
            doc_id="doc1",
            umls_cui=f"C{i:07d}",
            linking_confidence=0.8,
        )
        for i in range(1000)
    ]
    from biomed_kg_agent.nlp.config import NerConfig, get_default_priorities

    ner_config = NerConfig(model_priorities=get_default_priorities(), linker=config)
    save_nlp_results([doc], ner_config, db_path=temp_db_with_schema)

    # Query all entities - should handle chunking transparently
    entity_texts = {f"entity_{i}" for i in range(1000)}
    result = linker._get_cached_umls_mappings(entity_texts, temp_db_with_schema)

    # Verify all entities retrieved despite SQLite IN clause limits
    assert len(result) == 1000
    for i in range(1000):
        assert f"entity_{i}" in result
        assert result[f"entity_{i}"]["umls_cui"] == f"C{i:07d}"
