"""
Pure UMLS entity linking for biomedical entities.

This module provides UMLS (Unified Medical Language System) entity linking
functionality that is independent of the NER extraction process.

Classes:
- EntityLinker: Main class for UMLS entity linking operations

The module uses scispaCy's built-in UMLS knowledge base for linking entities
to standard biomedical ontologies.
"""

import gc
import logging
import re
from typing import Optional, TypedDict

import spacy
from spacy.language import Language
from sqlmodel import Session, create_engine, select

from biomed_kg_agent.nlp.config import LinkerConfig
from biomed_kg_agent.nlp.models import ExtractedEntity

logger = logging.getLogger(__name__)

# Memory-efficient processing constants
UMLS_LINKING_CHUNK_SIZE = 10000  # entities per chunk for memory-efficient processing
UMLS_BATCH_SIZE = 50  # entities per spaCy batch (2.5x speedup, memory-safe)
SQLITE_QUERY_CHUNK_SIZE = 800  # SQLite IN clause chunking to avoid variable limits
PROGRESS_LOG_INTERVAL = 1000  # log progress every N entities
GC_INTERVAL = 200  # GC every N entities (aligns with tests)


class UMLSMapping(TypedDict):
    """Type definition for UMLS entity linking results."""

    umls_cui: str | None
    linking_confidence: float | None
    umls_preferred_name: str | None
    umls_semantic_types: list[str] | None


class EntityLinker:
    """
    Pure UMLS entity linking using scispaCy's built-in knowledge base.

    This class handles only the entity linking process, taking extracted entities
    or entity texts and linking them to UMLS CUIDs.
    """

    def __init__(self, linker_config: LinkerConfig):
        """
        Initialize the entity linker with configuration.

        Args:
            linker_config: Configuration for UMLS linking
        """
        self.linker_config = linker_config
        self._umls_nlp: Optional[Language] = None

    def _get_umls_linker(self) -> Optional[Language]:
        """Lazy load UMLS linker using scispaCy's built-in UMLS knowledge base."""
        if not self.linker_config.enabled:
            logger.info("UMLS entity linking disabled by configuration")
            return None

        if self._umls_nlp is None:
            try:
                import scispacy  # noqa: F401
                import scispacy.linking  # noqa: F401 - Registers scispacy_linker factory

                core_model = self.linker_config.core_model
                self._umls_nlp = spacy.load(core_model)

                # Use scispaCy's dedicated linker pipe name
                if (
                    self._umls_nlp is not None
                    and "scispacy_linker" not in self._umls_nlp.pipe_names
                ):
                    self._umls_nlp.add_pipe("scispacy_linker")
                logger.info(
                    f"UMLS entity linking enabled with {core_model} (scispacy_linker)"
                )
                logger.info(
                    "Using scispaCy's built-in UMLS KB (~2GB, cached after first download)"
                )
            except Exception as e:
                logger.warning(f"UMLS entity linking unavailable: {e}")
                logger.info(
                    "Install scispaCy models: python -m spacy download en_core_sci_sm"
                )
                return None
        return self._umls_nlp

    def _get_cached_umls_mappings(
        self,
        entity_texts: set[str],
        output_db: str,
    ) -> dict[str, UMLSMapping]:
        """
        Retrieve existing UMLS mappings from database cache.

        Args:
            entity_texts: Set of entity texts to look up
            output_db: Database path

        Returns:
            Dictionary mapping entity_text -> {"umls_cui": cui, "confidence": conf}
            for entities found in cache
        """
        if not entity_texts:
            return {}

        try:
            engine = create_engine(f"sqlite:///{output_db}")
            cached_mappings: dict[str, UMLSMapping] = {}

            with Session(engine) as session:
                # Query for existing entity mappings that have been through UMLS linking
                # Use chunking to avoid SQLite "too many SQL variables" limits on large IN clauses
                texts_list = list(entity_texts)
                for i in range(0, len(texts_list), SQLITE_QUERY_CHUNK_SIZE):
                    chunk = texts_list[i : i + SQLITE_QUERY_CHUNK_SIZE]

                    cached_entities = session.exec(
                        select(ExtractedEntity).where(
                            ExtractedEntity.text.in_(chunk),  # type: ignore
                            # Only get entities that have been through UMLS linking
                            # This includes successful and failed attempts; we
                            # distinguish by checking if linking_confidence is
                            # not null (indicates UMLS attempted)
                            ExtractedEntity.linking_confidence.is_not(None),  # type: ignore
                        )
                    ).all()

                    for entity in cached_entities:
                        cached_mappings[entity.text] = {
                            "umls_cui": entity.umls_cui,
                            "linking_confidence": entity.linking_confidence,
                            "umls_preferred_name": entity.umls_preferred_name,
                            "umls_semantic_types": entity.umls_semantic_types,
                        }

            return cached_mappings

        except Exception as e:
            logger.warning(f"Failed to retrieve cached UMLS mappings: {e}")
            return {}

    def link_entities_with_cache(
        self,
        unique_entity_texts: set[str],
        output_db: str,
    ) -> dict[str, UMLSMapping]:
        """
        Link entity texts to UMLS CUIDs with database cache optimization.

        First checks database for existing mappings, then only processes
        uncached entities through expensive UMLS linking.

        Args:
            unique_entity_texts: Set of unique entity text strings
            output_db: Database path

        Returns:
            Dictionary mapping entity_text -> {"umls_cui": cui, "confidence": conf}
        """
        total_entities = len(unique_entity_texts)
        logger.info(f"Linking {total_entities} unique entities to UMLS...")

        # Step 1: Check database cache for existing mappings
        logger.info("Checking database cache for existing UMLS mappings...")
        cached_mappings = self._get_cached_umls_mappings(unique_entity_texts, output_db)

        # Step 2: Identify entities that need fresh UMLS linking
        uncached_entities = unique_entity_texts - set(cached_mappings.keys())

        cache_hit_rate = (
            (len(cached_mappings) / total_entities * 100) if total_entities > 0 else 0
        )
        logger.info(
            f"Cache results: {len(cached_mappings)}/{total_entities} entities "
            f"cached ({cache_hit_rate:.1f}% hit rate)"
        )

        # Step 3: Process uncached entities through UMLS linking
        if uncached_entities:
            logger.info(
                f"Processing {len(uncached_entities)} new entities through UMLS..."
            )
            new_mappings = self.link_entities(uncached_entities)
            cached_mappings.update(new_mappings)
        else:
            logger.info("All entities found in cache - no UMLS processing needed")

        return cached_mappings

    def link_entities(self, unique_entity_texts: set[str]) -> dict[str, UMLSMapping]:
        """
        Link entity texts to UMLS CUIDs with deduplication for memory efficiency.

        Processes unique entity texts to avoid duplicate UMLS processing,
        making it suitable for both single documents and large collections.

        Args:
            unique_entity_texts: Set of unique entity text strings

        Returns:
            Dictionary mapping entity_text -> {"umls_cui": cui, "confidence": conf}
        """
        nlp = self._get_umls_linker()
        if nlp is None:
            logger.warning("UMLS linker unavailable, returning empty mappings")
            return {
                text: {
                    "umls_cui": None,
                    "linking_confidence": None,
                    "umls_preferred_name": None,
                    "umls_semantic_types": None,
                }
                for text in unique_entity_texts
            }

        total_entities = len(unique_entity_texts)
        logger.info(f"Linking {total_entities} unique entities to UMLS...")
        logger.info(
            f"Using memory-efficient chunked processing "
            f"({UMLS_LINKING_CHUNK_SIZE} entity chunks)"
        )

        # Convert to list for processing
        entity_list = list(unique_entity_texts)
        entity_mappings: dict[str, UMLSMapping] = {}

        # Memory-efficient processing
        total_chunks = (
            len(entity_list) + UMLS_LINKING_CHUNK_SIZE - 1
        ) // UMLS_LINKING_CHUNK_SIZE

        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * UMLS_LINKING_CHUNK_SIZE
            end_idx = min(start_idx + UMLS_LINKING_CHUNK_SIZE, len(entity_list))
            chunk = entity_list[start_idx:end_idx]

            logger.info(
                f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk)} entities)"
            )

            # Process chunk with smaller batches for stability
            chunk_mappings = self._process_entity_chunk(nlp, chunk)
            entity_mappings.update(chunk_mappings)

            progress_pct = end_idx / total_entities * 100
            logger.info(
                f"   Progress: {end_idx}/{total_entities} entities ({progress_pct:.1f}%)"
            )

            # Aggressive memory cleanup between chunks
            del chunk_mappings
            gc.collect()

            logger.info(f"Chunk {chunk_idx + 1}/{total_chunks} complete")

        logger.info(f"UMLS linking complete: {len(entity_mappings)} entities processed")

        # Count successful links
        successful_links = sum(
            1 for mapping in entity_mappings.values() if mapping["umls_cui"] is not None
        )
        total_entities = len(entity_mappings)
        success_rate = (
            (successful_links / total_entities * 100) if total_entities > 0 else 0.0
        )
        logger.info(
            f"Successfully linked {successful_links}/{total_entities} unique entity texts "
            f"({success_rate:.1f}% text-level linking)"
        )

        return entity_mappings

    def _process_entity_chunk(
        self, nlp: Language, entity_chunk: list[str]
    ) -> dict[str, UMLSMapping]:
        """
        Process a single chunk of entities with fine-grained batching.

        Args:
            nlp: The UMLS-enabled spaCy pipeline
            entity_chunk: List of entity texts to process (up to UMLS_LINKING_CHUNK_SIZE)

        Returns:
            Dictionary mapping entity_text -> UMLSMapping
        """
        chunk_mappings: dict[str, UMLSMapping] = {}

        for i in range(0, len(entity_chunk), UMLS_BATCH_SIZE):
            batch = entity_chunk[i : i + UMLS_BATCH_SIZE]

            try:
                # Process batch through UMLS linker
                docs = list(nlp.pipe(batch))

                # Extract UMLS mappings
                for entity_text, doc in zip(batch, docs):
                    umls_cui = None
                    confidence = None
                    umls_preferred_name = None
                    umls_semantic_types = None

                    if doc.ents and len(doc.ents) > 0:
                        ent = doc.ents[0]
                        if hasattr(ent._, "kb_ents") and ent._.kb_ents:
                            cui, conf = ent._.kb_ents[0]
                            confidence = conf

                            # Apply confidence threshold
                            threshold = self.linker_config.confidence_threshold
                            if conf >= threshold:
                                umls_cui = cui
                                # Fetch UMLS preferred term and semantic types from KB
                                umls_preferred_name, umls_semantic_types = (
                                    self._get_umls_data(nlp, cui)
                                )

                    chunk_mappings[entity_text] = {
                        "umls_cui": umls_cui,
                        "linking_confidence": confidence,
                        "umls_preferred_name": umls_preferred_name,
                        "umls_semantic_types": umls_semantic_types,
                    }

                # Progress logging at defined intervals
                if (i + len(batch)) % PROGRESS_LOG_INTERVAL == 0 or (
                    i + len(batch)
                ) == len(entity_chunk):
                    current_count = i + len(batch)
                    total_count = len(entity_chunk)
                    msg = f"   Batch progress: {current_count}/{total_count} entities in chunk"
                    logger.info(msg)

                # Cleanup after each batch (conservative optimization)
                del docs
                if (i + len(batch)) % GC_INTERVAL == 0:
                    gc.collect()

            except Exception as e:
                logger.error(
                    f"Failed to process batch {i}-{i+len(batch)} in chunk: {e}"
                )
                # Add empty mappings for failed batch
                for entity_text in batch:
                    chunk_mappings[entity_text] = {
                        "umls_cui": None,
                        "linking_confidence": None,
                        "umls_preferred_name": None,
                        "umls_semantic_types": None,
                    }

        return chunk_mappings

    def _get_umls_data(
        self, nlp: Language, cui: str
    ) -> tuple[Optional[str], Optional[list[str]]]:
        """
        Get UMLS data (preferred name and semantic types) for a given CUI from scispaCy KB.

        Args:
            nlp: The UMLS-enabled spaCy pipeline
            cui: UMLS Concept Unique Identifier

        Returns:
            Tuple of (preferred_name, semantic_types) where semantic_types is list of TUI codes
        """
        try:
            linker = nlp.get_pipe("scispacy_linker")
            if hasattr(linker, "kb") and hasattr(linker.kb, "cui_to_entity"):
                entity = linker.kb.cui_to_entity.get(cui)
                if entity:
                    # Extract preferred name
                    preferred_name = None
                    if hasattr(entity, "canonical_name") and entity.canonical_name:
                        preferred_name = entity.canonical_name
                    elif hasattr(entity, "aliases") and entity.aliases:
                        preferred_name = entity.aliases[0]

                    # Extract semantic types (TUI codes like 'T121', 'T047')
                    semantic_types = None
                    if hasattr(entity, "types") and entity.types:
                        semantic_types = entity.types

                    return preferred_name, semantic_types
        except Exception as e:
            logger.debug(f"Failed to get UMLS data for {cui}: {e}")
        return None, None

    @staticmethod
    def apply_umls_mappings(
        entities_by_entity_type: dict[str, list[ExtractedEntity]],
        umls_mappings: dict[str, UMLSMapping],
    ) -> dict[str, list[ExtractedEntity]]:
        """
        Apply UMLS mappings to extracted entities IN-PLACE (memory-efficient).

        This completes the two-pass process by applying the UMLS links
        from pass 2 back to the entities extracted in pass 1.

        NER models are the sole source of entity_type. UMLS provides CUI enrichment
        (umls_cui, umls_preferred_name, umls_semantic_types) on all linked entities.

        Args:
            entities_by_entity_type: Entities from NER extraction
            umls_mappings: UMLS mappings from link_entities

        Returns:
            Entities with UMLS CUI enrichment applied
        """
        # Memory-efficient: Modify entities IN-PLACE, then re-group only if type changed
        # This avoids creating 77K+ entity copies that caused OOM
        linked_entities: dict[str, list[ExtractedEntity]] = {}

        for entity_type, entity_list in entities_by_entity_type.items():
            for entity in entity_list:
                # Apply UMLS mapping IN-PLACE if available
                if entity.text in umls_mappings:
                    mapping = umls_mappings[entity.text]
                    # Type-safe assignments
                    cui_value = mapping["umls_cui"]
                    entity.umls_cui = cui_value if isinstance(cui_value, str) else None

                    conf_value = mapping["linking_confidence"]
                    entity.linking_confidence = (
                        conf_value if isinstance(conf_value, float) else None
                    )

                    pref_name_value = mapping.get("umls_preferred_name")
                    entity.umls_preferred_name = (
                        pref_name_value if isinstance(pref_name_value, str) else None
                    )

                    semantic_types_value = mapping.get("umls_semantic_types")
                    if isinstance(semantic_types_value, list) and semantic_types_value:
                        entity.umls_semantic_types = semantic_types_value

                # Apply pattern-based corrections IN-PLACE
                EntityLinker.apply_pattern_correction_inplace(entity)

                # Group by the final corrected entity_type
                final_type = entity.entity_type
                if final_type not in linked_entities:
                    linked_entities[final_type] = []
                linked_entities[final_type].append(entity)

        return linked_entities

    @staticmethod
    def apply_pattern_correction_inplace(entity: ExtractedEntity) -> None:
        """Apply pattern-based post-corrections to a single entity IN-PLACE.

        Handles cases where human language conventions are unambiguous:
        - "X inhibitor/inhibition" -> always chemical (drug/therapeutic)
        - "X antibody/antibodies" -> always chemical (protein therapeutic)
        - "X syndrome" -> always disease (clinical condition)

        Applied after UMLS enrichment to correct compound noun phrases
        where the head noun changes the semantic category.

        Args:
            entity: Entity to correct (modified in-place)
        """
        # Pattern 1: Inhibitor/Inhibition -> chemical
        if re.search(r"\b(inhibitors?|inhibition)\b", entity.text, re.I):
            if entity.entity_type in [
                "gene",
                "sequence_feature",
                "cell_type",
                "disease",
                "organism",
                "anatomy",
                "cellular_component",
                "biological_process",
                "substance",
            ]:
                # Don't override biological process/activity terms
                if not re.search(r"\b(activity|pathway|process)\b", entity.text, re.I):
                    entity.entity_type = "chemical"

        # Pattern 2: Antibody/Antibodies -> chemical
        elif re.search(r"\bantibod(y|ies)\b", entity.text, re.I):
            if entity.entity_type in [
                "gene",
                "sequence_feature",
                "biological_process",
                "cell_type",
            ]:
                entity.entity_type = "chemical"

        # Pattern 3: Syndrome -> disease
        elif re.search(r"\bsyndrome\b", entity.text, re.I):
            if entity.entity_type in [
                "gene",
                "chemical",
                "cell_type",
                "organism",
                "anatomy",
                "sequence_feature",
                "pathology",
            ]:
                entity.entity_type = "disease"
