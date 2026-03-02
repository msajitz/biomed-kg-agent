"""Mock implementations for biomedical NER components to avoid loading heavy models in tests."""

from typing import Dict, List

from biomed_kg_agent.nlp.config import LinkerConfig
from biomed_kg_agent.nlp.models import ExtractedEntity


def create_mock_entity(
    text: str,
    source_label: str,
    start_char: int,
    end_char: int,
    doc_id: str = "test_doc",
    sentence_text: str = "Mock sentence for testing.",
    confidence: float = 0.9,
    entity_type: str | None = None,
) -> ExtractedEntity:
    """Create a mock ExtractedEntity for testing.

    Args:
        text: Entity text
        source_label: Original spaCy label (e.g., 'CHEMICAL', 'DISEASE')
        start_char: Start position
        end_char: End position
        doc_id: Document ID
        sentence_text: Sentence text
        confidence: NER confidence
        entity_type: Normalized entity type; if None, derived from source_label
    """
    # Derive entity_type from source_label if not provided
    if entity_type is None:
        entity_type = source_label.lower()

    return ExtractedEntity(
        text=text,
        start=start_char,
        end=end_char,
        source_model="mock_model",
        entity_type=entity_type,
        source_label=source_label,
        doc_id=doc_id,
        sentence_id=0,
        sentence_text=sentence_text,
        ner_confidence=confidence,
        umls_preferred_name=None,  # Default to None for tests
    )


class BiomedicalNER:
    """Mock BiomedicalNER for fast testing without loading heavy spaCy models."""

    def __init__(self, use_case: str = "default", linking_profile: str = "dev"):
        self.use_case = use_case
        self.linking_profile = linking_profile
        # Mock linker config
        self.linker_config = LinkerConfig(
            enabled=True, core_model="en_core_sci_sm", confidence_threshold=0.7
        )

    def extract_entities(
        self, text: str, doc_id: str
    ) -> Dict[str, List[ExtractedEntity]]:
        """Mock entity extraction that returns predictable test entities."""
        if not text.strip():
            return {}

        # Create mock entities based on common biomedical terms
        entities = {}

        # Mock chemical entities
        if any(
            term in text.lower() for term in ["glucose", "aspirin", "drug", "compound"]
        ):
            entities["chemical"] = [
                ExtractedEntity(
                    text="glucose" if "glucose" in text.lower() else "aspirin",
                    start_pos=0,
                    end_pos=7,
                    source_model="mock_bc5cdr",
                    entity_type="chemical",
                    source_label="CHEMICAL",
                    doc_id=doc_id,
                    sentence_id=0,
                    sentence_text=text,  # Use full text as sentence
                    ner_confidence=0.95,
                )
            ]

        # Mock disease entities
        if any(
            term in text.lower()
            for term in ["cancer", "tumor", "disease", "inflammation"]
        ):
            entities["disease"] = [
                ExtractedEntity(
                    text="cancer" if "cancer" in text.lower() else "inflammation",
                    start_pos=10,
                    end_pos=16,
                    source_model="mock_bc5cdr",
                    entity_type="disease",
                    source_label="DISEASE",
                    doc_id=doc_id,
                    sentence_id=0,
                    sentence_text=text,  # Use full text as sentence
                    ner_confidence=0.92,
                )
            ]

        # Mock gene entities
        if any(term in text.lower() for term in ["p53", "gene", "protein", "cox-2"]):
            entities["gene"] = [
                ExtractedEntity(
                    text="p53" if "p53" in text.lower() else "COX-2",
                    start_pos=20,
                    end_pos=23,
                    source_model="mock_bionlp13cg",
                    entity_type="gene",
                    source_label="GENE_OR_GENE_PRODUCT",
                    doc_id=doc_id,
                    sentence_id=0,
                    sentence_text=text,  # Use full text as sentence
                    ner_confidence=0.88,
                )
            ]

        return entities
