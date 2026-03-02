"""
Pure biomedical Named Entity Recognition (NER) using multiple scispaCy models.

This module provides comprehensive biomedical entity extraction using multiple scispaCy
models with intelligent deduplication and universal ontology mapping.

The universal approach ensures entities are consistently categorized,
enabling seamless knowledge graph construction and interoperability.

API Design - Class-Based Interface:

BiomedicalNER: Main class for pure entity extraction
- extract_entities(): Extract entities without UMLS linking

This modular design enables:
- Complete data source independence
- Separation of NER from entity linking
- Flexible integration with two-pass processing

For entity linking, use the two_pass_processor module which coordinates
NER extraction and UMLS linking efficiently.

Primary Usage:
    from biomed_kg_agent.nlp.biomedical_ner import BiomedicalNER
    from biomed_kg_agent.nlp.config import load_ner_config
    from biomed_kg_agent.nlp.two_pass_processor import process_documents_two_pass

    # Pure NER extraction with custom config
    config = load_ner_config("custom.yaml")
    ner = BiomedicalNER(config=config)
    entities = ner.extract_entities("Glucose metabolism in liver cells")

    # Or use default config
    ner = BiomedicalNER()
    entities = ner.extract_entities("Glucose metabolism in liver cells")

    # For linking, use two-pass processing
    texts = ["Glucose metabolism in liver cells"]
    entities_by_doc, umls_mappings = process_documents_two_pass(texts)

Classes:
- BiomedicalNER: Main class for entity extraction using all models

The module automatically uses all available scispaCy NER models (BC5CDR, BioNLP13CG, CRAFT)
for comprehensive biomedical entity coverage across all research domains.
"""

import gc
import logging

import spacy

from biomed_kg_agent.nlp.config import NerConfig, load_ner_config
from biomed_kg_agent.nlp.models import ExtractedEntity

logger = logging.getLogger(__name__)


# Universal ontology mapping for knowledge graph construction
# This ensures consistent entity categorization across all domains and compatibility
# with external knowledge bases and semantic web standards
#
# Mapping Strategy:
# - Flat categories for MVP/demo simplicity while preserving original labels
# - Two-level typing: universal category + source_type for domain experts
# - Hierarchical anatomy grouping (all anatomy subtypes -> "anatomy")
# - Chemical entities: CHEMICAL, SIMPLE_CHEMICAL, CHEBI -> "chemical"
# - GO terms: Currently all GO -> "biological_process" (future: sub-classify with GOATOOLS)
# - Dead mappings removed: DNA, RNA, PROTEIN, SPECIES, GO_BP, GO_MF, GO_CC
#   (These labels are not produced by current BC5CDR/BioNLP13CG/CRAFT ensemble)
#
# Future Integration Notes:
# - JNLPBA model would add: DNA, RNA, PROTEIN labels back
# - Consider GO sub-classification: GO_BP, GO_MF, GO_CC after GOATOOLS integration
UNIVERSAL_ONTOLOGY_MAPPING = {
    # Chemical entities - BC5CDR and CRAFT
    "CHEMICAL": "chemical",
    "SIMPLE_CHEMICAL": "chemical",
    "CHEBI": "chemical",
    # Biological entities - genes and proteins
    "GENE_OR_GENE_PRODUCT": "gene",
    "GGP": "gene",  # CRAFT gene/protein
    # Disease and disorder entities
    "DISEASE": "disease",
    "CANCER": "disease",
    # Anatomical entities - hierarchical approach
    "ORGAN": "anatomy",
    "TISSUE": "anatomy",
    "ANATOMICAL_SYSTEM": "anatomy",
    "MULTI_TISSUE_STRUCTURE": "anatomy",
    "DEVELOPING_ANATOMICAL_STRUCTURE": "anatomy",
    "IMMATERIAL_ANATOMICAL_ENTITY": "anatomy",
    "ORGANISM_SUBDIVISION": "anatomy",
    # Cellular entities
    "CELL": "cell_type",
    "CL": "cell_type",  # CRAFT cell ontology
    "CELLULAR_COMPONENT": "cellular_component",  # Keep specific for GO compatibility
    # Biological processes and functions - GO terms
    "GO": "biological_process",  # CRAFT GO terms (will need sub-classification)
    # Organism entities
    "ORGANISM": "organism",
    "TAXON": "organism",  # CRAFT taxonomy - map to organism
    # Sequence features - valuable for gene relationships
    "SO": "sequence_feature",  # CRAFT Sequence Ontology
    # Chemical compounds at residue level
    "AMINO_ACID": "amino_acid",  # Keep separate for metabolomics
    # Substance and pathology
    "ORGANISM_SUBSTANCE": "substance",
    "PATHOLOGICAL_FORMATION": "pathology",
}


class BiomedicalNER:
    """
    Universal biomedical NER for knowledge graph construction.

    Combines all available scispaCy NER models for comprehensive entity extraction
    with universal ontology mapping and deduplication.

    This is the single, unified class for all biomedical entity extraction.
    """

    # Available scispaCy models with their characteristics
    AVAILABLE_MODELS = {
        "bc5cdr": {
            "name": "en_ner_bc5cdr_md",
            "description": "Chemical and disease focused (BC5CDR dataset)",
        },
        "bionlp": {
            "name": "en_ner_bionlp13cg_md",
            "description": "Comprehensive biomedical entities (BioNLP13CG dataset)",
        },
        "craft": {
            "name": "en_ner_craft_md",
            "description": "Ontology-linked entities (CRAFT dataset)",
        },
    }

    # Shared model cache: Reuses loaded models across instances (singleton pattern)
    # Trade-off: First configuration pays model loading penalty (for 3 models)
    # Benefit: Prevents memory overflow from processing many documents
    # Design: Fresh instances with shared model cache for memory safety + performance
    _shared_models: dict[str, dict] = {}

    @classmethod
    def release_models(cls) -> None:
        """Release shared NER models to free memory.

        Call between processing passes when NER models are no longer needed
        (e.g., before loading the heavier UMLS linker in Pass 2).
        Reclaims ~300-600 MB for 3 scispaCy NER models.
        Models will be reloaded on next BiomedicalNER instantiation if needed.
        """
        if cls._shared_models:
            model_keys = list(cls._shared_models.keys())
            cls._shared_models.clear()
            gc.collect()
            logger.info(f"Released {len(model_keys)} NER models: {model_keys}")

    def __init__(self, config: NerConfig | None = None):
        """
        Initialize with configurable NER model and entity linking settings.

        Args:
            config: NerConfig object. If None, loads default config from ner_config.yaml
        """
        self.models = {}

        # Load configuration
        self.config = config if config is not None else load_ner_config()
        self.model_priorities = self.config.model_priorities
        self.linker_config = self.config.linker

        # Load all available models (use shared cache to save memory)
        for model_key, model_info in self.AVAILABLE_MODELS.items():
            try:
                model_name = model_info["name"]
                priority = self.model_priorities.get(model_key, 1)  # Default priority 1

                # Check if model is already loaded in shared cache
                if model_key not in self._shared_models:
                    logger.info(
                        f"Loading {model_key} ({model_name}) into shared cache..."
                    )
                    nlp = spacy.load(model_name)
                    self._shared_models[model_key] = {
                        "nlp": nlp,
                        "name": model_name,
                    }
                    logger.info(f"Loaded {model_key}: {model_info['description']}")
                else:
                    logger.info(
                        f"Reusing cached {model_key}: {model_info['description']}"
                    )

                # Reference the shared model with priority
                self.models[model_key] = {
                    "nlp": self._shared_models[model_key]["nlp"],
                    "name": model_name,
                    "priority": priority,
                }
            except Exception as e:
                logger.error(f"Failed to load {model_key}: {e}")

        if not self.models:
            raise RuntimeError("No scispaCy models could be loaded")

        logger.info(f"Initialized NER with {len(self.models)} models")
        logger.info(f"Model priorities: {self.model_priorities}")
        logger.info(f"Linker config: {self.linker_config}")

    def extract_entities(
        self, text: str, doc_id: str = "unknown"
    ) -> dict[str, list[ExtractedEntity]]:
        """
        Extract entities using all available models with universal categorization.

        Args:
            text: Input text to process
            doc_id: Document identifier for provenance tracking

        Returns:
            Entities grouped by universal category, ready for knowledge graph construction
        """
        all_entities = []

        # Extract entities from all models
        for model_key, model_info in self.models.items():
            nlp = model_info["nlp"]
            model_name = model_info["name"]

            try:
                doc = nlp(text)

                # Process sentences to get sentence-level granularity
                for sent_id, sent in enumerate(doc.sents):
                    for ent in sent.ents:
                        entity_type = UNIVERSAL_ONTOLOGY_MAPPING.get(
                            ent.label_, "unknown"
                        )

                        entity = ExtractedEntity(
                            text=ent.text,
                            start_pos=ent.start_char,
                            end_pos=ent.end_char,
                            source_model=model_name,
                            entity_type=entity_type,
                            source_label=ent.label_,  # Store original spaCy label
                            doc_id=doc_id,
                            sentence_id=sent_id,
                            sentence_text=sent.text,  # Capture full sentence text
                            # scispaCy models don't provide detection confidence
                            ner_confidence=None,
                            # Will be populated during entity linking
                            linking_confidence=None,
                            umls_cui=None,
                            umls_preferred_name=None,
                            chebi_id=None,
                            go_id=None,
                            mesh_id=None,
                        )
                        all_entities.append(entity)

                entity_count = len(
                    [e for e in all_entities if e.source_model == model_name]
                )
                logger.debug(f"{model_key}: {entity_count} entities")
            except Exception as e:
                logger.error(f"Error processing with {model_key}: {e}")

        # Deduplicate entities across models
        deduplicated_entities = self._deduplicate_entities(all_entities)
        logger.debug(
            f"Deduplicated: {len(all_entities)} -> "
            f"{len(deduplicated_entities)} entities"
        )

        # Group by entity_type
        entity_types: dict[str, list[ExtractedEntity]] = {}
        for entity in deduplicated_entities:
            if entity.entity_type not in entity_types:
                entity_types[entity.entity_type] = []
            entity_types[entity.entity_type].append(entity)

        return entity_types

    def _deduplicate_entities(
        self, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        """Remove duplicate entities based on text, position, and model priority.

        Deduplication Strategy:
        =====================
        When multiple NER models extract entities from the same text, they often identify
        the same conceptual entity with slight variations (different boundaries, whitespace,
        capitalization). This method groups likely duplicates and selects the best prediction.

        Algorithm:
        1. Normalize entity text (lowercase, strip whitespace)
        2. Group entities by position buckets (5-character windows)
        3. Create composite keys: (normalized_text, position_group)
        4. For each group, select entity from highest-priority model

        Examples:
        - "Glucose" (pos 0) + "glucose " (pos 1) -> Same group: ("glucose", 0)
        - "p53" (pos 0) + "p53" (pos 10) -> Different groups: avoid false grouping

        Trade-offs:
        - Handles common NER boundary/formatting differences effectively
        - May miss some complex overlapping entities (e.g., nested mentions)
        - 5-character window balances precision vs. recall for biomedical text

        TODO: Consider storing all model results for KG flexibility
        Current: Store primary result per entity (highest priority model)
        Future: Store all model results per entity for multi-property nodes in knowledge graphs

        Args:
            entities: List of entities from all NER models

        Returns:
            Deduplicated list with one entity per conceptual mention
        """
        # Get model priority mapping
        model_priority = {}
        for model_info in self.models.values():
            model_priority[model_info["name"]] = model_info["priority"]

        # Group entities by text and approximate position
        entity_groups: dict[tuple[str, int], list[ExtractedEntity]] = {}
        for entity in entities:
            # Step 1: Normalize entity text (lowercase, strip whitespace)
            text_key = entity.text.lower().strip()

            # Step 2: Create position buckets (5-character windows)
            # Entities starting within 5 characters are considered "close"
            pos_group = entity.start_pos // 5

            # Step 3: Create composite keys combining text and position
            key = (text_key, pos_group)

            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(entity)

        # Step 4: Select best entity from each group based on model priority
        deduplicated = []
        for group_entities in entity_groups.values():
            if len(group_entities) == 1:
                # No duplicates in this group
                deduplicated.append(group_entities[0])
            else:
                # Multiple entities in group - select highest priority model
                group_entities.sort(
                    key=lambda e: model_priority.get(e.source_model, 0), reverse=True
                )
                deduplicated.append(group_entities[0])

        return deduplicated

    def get_info(self) -> dict:
        """Get information about loaded models and capabilities."""
        return {
            "loaded_models": list(self.models.keys()),
            "total_models": len(self.models),
            "available_models": list(self.AVAILABLE_MODELS.keys()),
            "universal_categories": list(set(UNIVERSAL_ONTOLOGY_MAPPING.values())),
            "ontology_mapping": UNIVERSAL_ONTOLOGY_MAPPING,
            "umls_linking_config": (
                self.linker_config.model_dump() if self.linker_config else None
            ),
        }
