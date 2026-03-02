"""Database-driven two-pass processor: NER extraction -> UMLS linking with auto-resume."""

import gc
import logging
import uuid
from contextlib import contextmanager
from typing import Generator, Optional

from sqlmodel import Session, SQLModel, create_engine, select

from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.nlp.biomedical_ner import BiomedicalNER
from biomed_kg_agent.nlp.config import NerConfig, load_ner_config
from biomed_kg_agent.nlp.entity_linking import EntityLinker, UMLSMapping
from biomed_kg_agent.nlp.models import ExtractedEntity, ProcessedDocument
from biomed_kg_agent.nlp.persistence import (
    load_entities,
    save_nlp_results,
)

logger = logging.getLogger(__name__)


def process_documents_two_pass(
    texts: list[str],
    doc_ids: Optional[list[str]] = None,
    config_path: Optional[str] = None,
    output_db: Optional[str] = None,
    doc_metadata: Optional[dict[str, dict]] = None,
) -> tuple[list[dict[str, list[ExtractedEntity]]], dict[str, UMLSMapping]]:
    """
    Two-pass processing: NER extraction -> UMLS linking, with DB persistence and auto-resume.

    Memory-efficient: entities are persisted to DB during Pass 1 and never
    accumulated in memory. NER models are released between passes. Results
    are reloaded from DB only at the end.

    Args:
        texts: List of document texts to process
        doc_ids: Optional list of document IDs (generates UUIDs if None)
        config_path: Path to NER config file (optional, defaults to ner_config.yaml)
        output_db: Optional database path override (defaults to data/nlp_results.db)
        doc_metadata: Optional metadata for documents

    Returns:
        Tuple of (document_entities, umls_mappings) where document_entities is
        a list of dicts mapping entity_type -> list of ExtractedEntity (one per doc),
        and umls_mappings maps entity text -> UMLSMapping.
    """
    logger.info(f"Starting two-pass processing for {len(texts)} documents")

    config = load_ner_config(config_path)
    doc_ids, output_db = _prepare_doc_ids_and_db_path(texts, doc_ids, output_db)
    processed_ids = _get_processed_doc_ids(config, output_db)
    unprocessed = [
        (doc_ids[i], texts[i])
        for i in range(len(doc_ids))
        if doc_ids[i] not in processed_ids
    ]

    # Pass 1: NER extraction to DB
    if unprocessed:
        already_done = len(doc_ids) - len(unprocessed)
        if already_done > 0:
            logger.info(
                f"Auto-resuming: {already_done}/{len(doc_ids)} already processed, "
                f"{len(unprocessed)} remaining"
            )
        else:
            logger.info("Starting fresh processing")
        _extract_entities_to_db(
            [t for _, t in unprocessed],
            [did for did, _ in unprocessed],
            already_done,
            config,
            output_db,
            doc_metadata,
        )
    else:
        logger.info("All documents already processed in Pass 1")

    unique_entity_texts = _get_unique_entity_texts_from_db(output_db)
    logger.info(f"Pass 1 complete: {len(unique_entity_texts)} unique entities found")

    # Free NER models before loading heavier UMLS linker (~300-600 MB reclaimed)
    BiomedicalNER.release_models()

    # Pass 2: UMLS linking + DB update
    logger.info(f"Pass 2: Linking {len(unique_entity_texts)} unique entities to UMLS")
    entity_linker = EntityLinker(config.linker)
    umls_mappings = entity_linker.link_entities_with_cache(
        unique_entity_texts, output_db
    )

    del entity_linker
    gc.collect()

    _update_entities_in_database(umls_mappings, output_db)
    document_entities = _load_results_in_order(doc_ids, output_db)

    logger.info(
        f"Two-pass processing complete: {len(texts)} documents, "
        f"{len(unique_entity_texts)} unique entities"
    )
    return document_entities, umls_mappings


def _extract_entities_to_db(
    texts: list[str],
    doc_ids: list[str],
    already_done: int,
    config: NerConfig,
    output_db: str,
    doc_metadata: dict[str, dict] | None = None,
) -> None:
    """Extract entities from texts and save to DB in batches (100 docs each)."""
    logger.info("Pass 1: Extracting entities (memory efficient, no UMLS linking)")

    ner = BiomedicalNER(config=config)
    docs_pending_save: list[DocumentInternal] = []
    batch_size = 100
    fail_count = 0

    for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
        try:
            entities_by_type = ner.extract_entities(text, doc_id)

            metadata = (doc_metadata or {}).get(doc_id, {})
            doc = DocumentInternal(
                id=doc_id,
                title=metadata.get("title", ""),
                text=text,
                source=metadata.get("source", "processed"),
                pub_year=metadata.get("pub_year"),
                extras=metadata.get("extras", {}),
            )
            doc.entities = [e for ents in entities_by_type.values() for e in ents]
            docs_pending_save.append(doc)

            if len(docs_pending_save) >= batch_size or i == len(texts) - 1:
                save_nlp_results(
                    docs_pending_save.copy(),
                    config,
                    db_path=output_db,
                )
                current_total = already_done + i + 1
                logger.info(
                    f"Saved batch: {len(docs_pending_save)} docs "
                    f"(total processed: {current_total})"
                )
                docs_pending_save.clear()
                gc.collect()

            if (i + 1) % 100 == 0 or i == len(texts) - 1:
                current_total = already_done + i + 1
                logger.info(
                    f"Processed {current_total}/{already_done + len(texts)} documents"
                )

        except Exception as e:
            fail_count += 1
            logger.error(f"Failed to process document {doc_id}: {e}")

    if fail_count:
        logger.warning(f"Pass 1 completed with {fail_count}/{len(texts)} failures")


def _get_processed_doc_ids(config: NerConfig, output_db: str) -> set[str]:
    """Return the set of document IDs already processed in the output DB."""
    try:
        with _db_session(output_db) as session:
            _validate_db_config(session, config)
            return set(session.exec(select(ProcessedDocument.id)).all())
    except ValueError:
        raise
    except Exception as e:
        logger.warning(
            f"Could not read processed doc IDs from {output_db}: {e}. "
            "Treating all documents as unprocessed. "
            "If the DB already has data, duplicate-key errors may follow."
        )
        return set()


def _validate_db_config(session: Session, config: NerConfig) -> None:
    """Raise ValueError if the DB was processed with a different NER config."""
    first_doc = session.exec(select(ProcessedDocument).limit(1)).first()
    if first_doc and first_doc.config_json:
        stored_config = NerConfig.model_validate_json(first_doc.config_json)
        if stored_config != config:
            raise ValueError(
                "Config mismatch! Database was processed with different config.\n"
                "Solution: Use a different database (--output-db) or restore"
                " original config.\n"
                f"Stored config: {stored_config.model_dump_json(indent=2)}\n"
                f"Current config: {config.model_dump_json(indent=2)}"
            )


@contextmanager
def _db_session(output_db: str) -> Generator[Session, None, None]:
    """Create a database session for the given SQLite path (or :memory:)."""
    engine = create_engine(f"sqlite:///{output_db}")
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _update_entities_in_database(
    umls_mappings: dict[str, UMLSMapping],
    output_db: str,
) -> None:
    """Update entity records in DB with UMLS enrichment."""
    if not umls_mappings:
        logger.info("No UMLS mappings to apply")
        return

    try:
        with _db_session(output_db) as session:
            entities = session.exec(select(ExtractedEntity)).all()

            if not entities:
                logger.warning("No entities found in database to update")
                return

            umls_updated_count = 0
            type_corrected_count = 0

            for entity in entities:
                original_type = entity.entity_type

                if entity.text in umls_mappings:
                    mapping = umls_mappings[entity.text]
                    entity.umls_cui = mapping.get("umls_cui")
                    entity.linking_confidence = mapping.get("linking_confidence")
                    entity.umls_preferred_name = mapping.get("umls_preferred_name")

                    semantic_types = mapping.get("umls_semantic_types")
                    entity.umls_semantic_types = semantic_types or None

                    umls_updated_count += 1

                EntityLinker.apply_pattern_correction_inplace(entity)

                if entity.entity_type != original_type:
                    type_corrected_count += 1

            session.commit()
            logger.info(
                f"Updated {umls_updated_count}/{len(entities)} entities with UMLS data "
                f"({umls_updated_count * 100.0 / len(entities):.1f}%)"
            )
            if type_corrected_count > 0:
                logger.info(
                    f"Corrected entity_type for {type_corrected_count} entities "
                    "(pattern corrections)"
                )

            # Sanity check: if we had UMLS mappings but updated 0 entities
            if umls_mappings and umls_updated_count == 0:
                raise RuntimeError(
                    f"UMLS update sanity check failed: {len(umls_mappings)} mappings "
                    "but 0 entities updated"
                )

    except Exception as e:
        logger.error(f"Failed to update entities in database: {e}", exc_info=True)
        raise  # Re-raise instead of silent return


def _get_unique_entity_texts_from_db(output_db: str) -> set[str]:
    """Query all unique entity texts from the database."""
    try:
        with _db_session(output_db) as session:
            results = session.exec(select(ExtractedEntity.text).distinct()).all()
            return set(results)
    except Exception as e:
        logger.warning(f"Failed to query unique entity texts: {e}")
        return set()


def _load_results_in_order(
    doc_ids: list[str], output_db: str
) -> list[dict[str, list[ExtractedEntity]]]:
    """Load entities from DB, grouped by type, preserving doc_ids order."""
    if not doc_ids:
        return []

    try:
        entities_by_doc = load_entities(db_path=output_db)

        result = []
        for doc_id in doc_ids:
            grouped: dict[str, list[ExtractedEntity]] = {}
            for entity in entities_by_doc.get(doc_id, []):
                grouped.setdefault(entity.entity_type, []).append(entity)
            result.append(grouped)

        return result

    except Exception as e:
        raise RuntimeError(
            f"Failed to load results after processing. "
            f"Database may be corrupted or inaccessible. Error: {e}"
        ) from e


def _prepare_doc_ids_and_db_path(
    texts: list[str],
    doc_ids: Optional[list[str]] = None,
    output_db: Optional[str] = None,
) -> tuple[list[str], str]:
    """Generate document IDs if needed and set default database path."""
    if doc_ids is None:
        logger.warning(
            "No doc_ids provided; generating UUIDs. "
            "Resume across runs is disabled (IDs are not stable)."
        )
        doc_ids = [str(uuid.uuid4()) for _ in range(len(texts))]
    elif len(doc_ids) != len(texts):
        raise ValueError(
            f"doc_ids length ({len(doc_ids)}) must match texts length ({len(texts)})"
        )

    if output_db is None:
        output_db = "data/nlp_results.db"

    return doc_ids, output_db
