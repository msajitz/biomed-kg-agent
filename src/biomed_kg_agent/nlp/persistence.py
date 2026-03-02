"""
Persistence functions for intermediate NLP results.

This module handles storing and loading NLP-processed documents and entity mentions
for the disentangled pipeline architecture.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from sqlmodel import Session, SQLModel, select

from biomed_kg_agent.core.database import create_db_engine
from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.nlp.config import NerConfig
from biomed_kg_agent.nlp.models import ExtractedEntity, ProcessedDocument

logger = logging.getLogger(__name__)


def build_processed_document(
    doc: DocumentInternal, config: NerConfig
) -> ProcessedDocument:
    """Construct a ProcessedDocument from a DocumentInternal.

    Now with proper typing since circular imports are resolved.
    """
    # Extract metadata from extras
    extras = doc.extras or {}

    # Remaining extras (source-specific) go to JSON
    whitelist = [
        "journal",
        "authors",
        "doi",
        "document_type",
        "keywords",
        "mesh_terms",
    ]
    remaining_extras = {k: v for k, v in extras.items() if k not in whitelist}

    return ProcessedDocument(
        id=doc.id,
        title=doc.title,
        text=doc.text,
        source=doc.source,
        pub_year=doc.pub_year,
        # Academic publication metadata
        journal=extras.get("journal"),
        authors=extras.get("authors"),
        document_type=extras.get("document_type"),
        doi=extras.get("doi"),
        keywords=extras.get("keywords"),
        mesh_terms=extras.get("mesh_terms"),
        # Remaining extras (for future sources)
        extras_json=json.dumps(remaining_extras) if remaining_extras else None,
        processed_at=datetime.now(),
        config_json=config.model_dump_json(),
        entity_count=len(doc.entities or []),
    )


def save_nlp_results(
    docs: list[DocumentInternal],
    config: NerConfig,
    db_path: str = "data/nlp_results.db",
    database_url: str | None = None,
) -> None:
    """
    Save NLP processing results to intermediate storage.

    Args:
        docs: List of DocumentInternal objects with extracted entities
        config: NER configuration used
        db_path: Path to SQLite database (used if database_url is None)
        database_url: Full database URL (e.g., "postgresql://user:pass@host/db")
    """
    # Create database and tables (even if docs is empty, ensure valid DB structure)
    engine = create_db_engine(db_path=db_path, database_url=database_url)
    SQLModel.metadata.create_all(engine)

    if not docs:
        logger.warning("No documents to save")
        return

    logger.info(f"Saving {len(docs)} processed documents to {db_path}")

    total_entities = 0

    with Session(engine, expire_on_commit=False) as session:
        for doc in docs:
            processed_doc = build_processed_document(doc, config)
            session.add(processed_doc)

            # Store entities (expire_on_commit=False keeps them usable after commit)
            for entity in doc.entities or []:
                session.add(entity)
                total_entities += 1

        session.commit()

    logger.info(f"Saved {len(docs)} documents with {total_entities} entities")


def load_entities(
    db_path: str,
    database_url: str | None = None,
) -> dict[str, list[ExtractedEntity]]:
    """Load entities from database.

    Args:
        db_path: Path to SQLite database
        database_url: Full database URL (overrides db_path)

    Returns:
        Dict mapping doc_id -> list of ExtractedEntity objects
    """
    engine = create_db_engine(db_path=db_path, database_url=database_url)

    with Session(engine, expire_on_commit=False) as session:
        # Query all entities (expire_on_commit=False keeps them usable after session closes)
        query = select(ExtractedEntity)
        entities = session.exec(query).all()

    # Group entities by document (JSON decoding happens automatically)
    entities_by_doc: dict[str, list[ExtractedEntity]] = {}

    for entity in entities:
        if entity.doc_id not in entities_by_doc:
            entities_by_doc[entity.doc_id] = []
        entities_by_doc[entity.doc_id].append(entity)

    return entities_by_doc


def load_processed_documents(
    db_path: str, database_url: str | None = None
) -> list[DocumentInternal]:
    """
    Load processed documents from intermediate storage.

    Args:
        db_path: Path to SQLite database with intermediate storage

    Returns:
        List of DocumentInternal objects with reconstructed entities
    """
    logger.info(f"Loading processed documents from {db_path}")

    engine = create_db_engine(db_path=db_path, database_url=database_url)

    with Session(engine) as session:
        # Load processed documents
        statement = select(ProcessedDocument)
        processed_docs = session.exec(statement).all()

        if not processed_docs:
            logger.warning("No processed documents found")
            return []

    # Load and decode entities
    entities_by_doc = load_entities(db_path=db_path, database_url=database_url)

    # Reconstruct DocumentInternal objects
    internal_docs = []
    total_entities = 0

    for processed_doc in processed_docs:
        entities = entities_by_doc.get(processed_doc.id, [])

        # Create DocumentInternal
        extras = (
            json.loads(processed_doc.extras_json) if processed_doc.extras_json else {}
        )
        doc = DocumentInternal(
            id=processed_doc.id,
            title=processed_doc.title,
            text=processed_doc.text,
            source=processed_doc.source,
            pub_year=processed_doc.pub_year,
            extras=extras,
            entities=entities,
        )

        internal_docs.append(doc)
        total_entities += len(entities)

    logger.info(f"Loaded {len(internal_docs)} documents with {total_entities} entities")
    return internal_docs


def get_processing_info(
    db_path: str, database_url: str | None = None
) -> Optional[dict]:
    """
    Get processing metadata from intermediate storage.

    Args:
        db_path: Path to SQLite database

    Returns:
        Dictionary with processing info or None if no data
    """
    try:
        engine = create_db_engine(db_path=db_path, database_url=database_url)

        with Session(engine) as session:

            statement = select(ProcessedDocument)
            all_docs = session.exec(statement).all()

            if not all_docs:
                return None

            # Parse config from JSON
            # Enforce single-config assumption: all docs must have same config
            configs = {doc.config_json for doc in all_docs}
            if len(configs) > 1:
                raise ValueError(
                    "Database contains documents with "
                    f"{len(configs)} different configs. "
                    "get_processing_info() requires all documents to be "
                    "processed with the same config."
                )

            # Get config and total entities
            first_doc = all_docs[0]
            total_entities = sum(doc.entity_count for doc in all_docs)
            config_dict = json.loads(first_doc.config_json)

            return {
                "document_count": len(all_docs),
                "total_entities": total_entities,
                "config": config_dict,
                "processed_at": first_doc.processed_at,
            }

    except Exception as e:
        logger.error(f"Failed to get processing info: {e}")
        return None


def get_document_by_id(
    doc_id: str, db_path: str | None = None, database_url: str | None = None
) -> Optional[dict]:
    """Fetch a single processed document by ID, returning a dict or None."""
    try:
        engine = create_db_engine(db_path=db_path, database_url=database_url)
        with Session(engine) as session:
            statement = select(ProcessedDocument).where(ProcessedDocument.id == doc_id)
            doc = session.exec(statement).first()
            if doc:
                logger.info(f"Retrieved document {doc_id}: {(doc.title or '')[:50]}...")
                return doc.model_dump()
            else:
                logger.warning(f"Document {doc_id} not found in database")
                return None
    except Exception as e:
        logger.error(f"Error fetching document {doc_id}: {e}")
        return None
