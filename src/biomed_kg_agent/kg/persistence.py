"""
Database persistence functions for knowledge graph data.

This module provides functions to save knowledge graph models to persistent storage.
"""

import logging

from sqlmodel import Session, create_engine

from .models import Cooccurrence, Entity, Mention

logger = logging.getLogger(__name__)


def save_kg_data(
    entities: list[Entity],
    mentions: list[Mention],
    cooccurrences: list[Cooccurrence],
    db_path: str = "data/kg.db",
) -> None:
    """Save all knowledge graph models to database.

    Args:
        entities: Normalized entities
        mentions: Entity mentions
        cooccurrences: Co-occurrence statistics
        db_path: SQLite database path
    """
    database_url = f"sqlite:///{db_path}"
    engine = create_engine(database_url)

    # Create tables
    from .models import SQLModel

    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        # Add entities
        for entity in entities:
            session.merge(entity)

        # Add mentions
        for mention in mentions:
            session.add(mention)

        # Add co-occurrences
        for cooccurrence in cooccurrences:
            session.merge(cooccurrence)

        session.commit()

    logger.info(
        f"Saved {len(entities)} entities, {len(mentions)} mentions, "
        f"{len(cooccurrences)} co-occurrences to {database_url}"
    )
