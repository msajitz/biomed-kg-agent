"""Neo4j integration and migration module.

Provides Neo4j connection management and SQLite to Neo4j migration.
Uses idempotent MERGE-based migration with configurable filtering
and evidence properties (docs_count, sent_count, evidence_sentences).
"""

import json
import logging
from typing import Optional

from neo4j import Driver, GraphDatabase, Transaction
from sqlmodel import Session, select

from biomed_kg_agent.core.database import create_db_engine
from biomed_kg_agent.kg.config import FilterConfig
from biomed_kg_agent.kg.filtering import FilterRelationships
from biomed_kg_agent.kg.models import Cooccurrence, Entity

logger = logging.getLogger(__name__)


def get_neo4j_driver(uri: str, user: str, password: str) -> Driver:
    """Create and return Neo4j driver instance.

    Args:
        uri: Neo4j connection URI (e.g., bolt://localhost:7687)
        user: Neo4j username
        password: Neo4j password

    Returns:
        Connected Neo4j driver instance
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    logger.info(f"Connected to Neo4j at {uri}")
    return driver


def ensure_schema(driver: Driver, database: str) -> None:
    """Ensure Neo4j schema exists: unique constraint and query performance indexes.

    Creates (if not exists):
    - Unique constraint on Entity.id (for idempotent MERGE)
    - Index on Entity.name (for agent name lookups)
    - Index on Entity.entity_type (for type filtering)
    - Index on Entity.umls_cui (for CUI lookups)
    """

    def _ensure_schema_tx(tx: Transaction) -> None:
        # Unique constraint on Entity.id
        tx.run(
            """
            CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
            FOR (e:Entity) REQUIRE e.id IS UNIQUE
            """
        )
        logger.debug("Ensured unique constraint on Entity.id")

        # Query performance indexes
        tx.run(
            """
            CREATE INDEX entity_name_index IF NOT EXISTS
            FOR (e:Entity) ON (e.name)
            """
        )
        tx.run(
            """
            CREATE INDEX entity_type_index IF NOT EXISTS
            FOR (e:Entity) ON (e.entity_type)
            """
        )
        tx.run(
            """
            CREATE INDEX umls_cui_index IF NOT EXISTS
            FOR (e:Entity) ON (e.umls_cui)
            """
        )
        logger.debug("Ensured query performance indexes")

    with driver.session(database=database) as session:
        session.execute_write(_ensure_schema_tx)
    logger.info("Neo4j schema ensured (indexes and constraints)")


def clear_graph(driver: Driver, database: str, batch_size: int = 100) -> None:
    """Delete all nodes and relationships from Neo4j graph in batches.

    Warning: This is a destructive operation. Use with caution.

    Args:
        driver: Neo4j driver instance
        database: Target database name
        batch_size: Number of nodes to delete per transaction (default: 100)
                   Very small batches required for AuraDB compatibility when nodes
                   have many relationships with large properties (evidence_sentences)
    """

    def _count_nodes_tx(tx: Transaction) -> int:
        result = tx.run("MATCH (n) RETURN count(n) as count")
        return result.single()["count"]

    def _delete_batch_tx(tx: Transaction, batch_size: int) -> int:
        result = tx.run(
            """
            MATCH (n)
            WITH n LIMIT $batch_size
            DETACH DELETE n
            RETURN count(n) as deleted
            """,
            batch_size=batch_size,
        )
        return result.single()["deleted"]

    with driver.session(database=database) as session:
        # Count total nodes first
        total_nodes = session.execute_read(_count_nodes_tx)
        logger.info(f"Clearing {total_nodes} nodes from Neo4j graph...")

        # Delete in batches until graph is empty
        total_deleted = 0
        while True:
            deleted = session.execute_write(_delete_batch_tx, batch_size)
            if deleted == 0:
                break
            total_deleted += deleted
            if total_deleted % 10000 == 0 or deleted < batch_size:
                logger.info(f"Deleted {total_deleted}/{total_nodes} nodes...")

    logger.info(f"Neo4j graph cleared ({total_deleted} nodes deleted)")


def migrate_to_neo4j(
    db_path: str,
    neo4j_uri: str,
    neo4j_user: str = "neo4j",
    neo4j_password: str = "",
    neo4j_database: str = "neo4j",
    filter_config: Optional[FilterConfig] = None,
    clear_existing: bool = False,
    # Forward-compatible parameters (MVP ignores with warnings)
    batch_size: Optional[int] = None,
    resume: bool = True,
) -> dict[str, int]:
    """Migrate SQLite KG data to Neo4j with filtering and evidence sentences.

    MVP implementation uses single-pass migration with pre-computed evidence sentences
    from Cooccurrence table. Signature includes batch_size and resume parameters
    for forward compatibility with post-MVP batching.

    Args:
        db_path: Path to SQLite database with entities, mentions, and co-occurrences
        neo4j_uri: Neo4j connection URI (e.g., bolt://localhost:7687)
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        neo4j_database: Neo4j database name
        filter_config: Filter parameters (docs_count, sent_count, stopwords).
                      If None, uses default FilterConfig (docs_count_min=2, sent_count_min=2)
        clear_existing: If True, delete all nodes before migration
        batch_size: MVP: ignored with warning. Post-MVP: chunk size for batched processing
        resume: MVP: no-op with info log. Post-MVP: auto-resume from interruptions

    Returns:
        Statistics dict with keys:
        - entities_created: Number of entity nodes created in Neo4j
        - relationships_created: Number of relationship edges created
        - entities_filtered: Number of orphaned entities (not in any relationship)
        - relationships_filtered: Number of relationships filtered out

    Example:
        >>> stats = migrate_to_neo4j(
        ...     db_path="data/kg.db",
        ...     neo4j_uri="bolt://localhost:7687",
        ...     neo4j_password="password",
        ...     filter_config=FilterConfig(docs_count_min=2, sent_count_min=2),
        ... )
        >>> print(f"Created {stats['entities_created']} entities")
    """
    # Forward-compatibility warnings for MVP
    if batch_size is not None:
        logger.warning(
            f"batch_size parameter is not implemented in MVP (provided: {batch_size}). "
            "Single-pass migration will be used. This parameter is reserved for post-MVP batching."
        )
    if not resume:
        logger.warning(
            "resume parameter is not implemented in MVP (provided: False). "
            "This parameter is reserved for post-MVP progress tracking."
        )

    # Use default filter config if none provided
    if filter_config is None:
        filter_config = FilterConfig()
        logger.info(
            f"Using default filter config: docs_count_min={filter_config.docs_count_min}, "
            f"sent_count_min={filter_config.sent_count_min}"
        )

    # Connect to Neo4j
    driver = get_neo4j_driver(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Optional: clear existing graph
        if clear_existing:
            logger.info("Clearing existing Neo4j graph...")
            clear_graph(driver, neo4j_database)

        # Ensure schema exists (indexes and constraints)
        ensure_schema(driver, neo4j_database)

        # Load and filter relationships
        logger.info(f"Loading relationships from {db_path}...")
        filter_rels = FilterRelationships(db_path, filter_config)
        filtered_relationships = filter_rels.apply_filters()  # loads and filters
        total_relationships = len(filter_rels.load_rows())  # accesses cached result
        filtered_count = total_relationships - len(filtered_relationships)

        logger.info(
            f"Filtered relationships: {len(filtered_relationships)} kept, "
            f"{filtered_count} filtered out (from {total_relationships} total)"
        )

        # Load all entities from SQLite (no filtering - preserve complete entity catalog)
        engine = create_db_engine(db_path)
        with Session(engine) as session:
            entities_to_migrate = list(session.exec(select(Entity)).all())

        logger.info(
            f"Entities to migrate: {len(entities_to_migrate)} "
            f"(all entities, including those without filtered relationships)"
        )

        # Migrate entities to Neo4j (batched UNWIND operations)
        logger.info("Migrating entities to Neo4j...")
        _migrate_entities(driver, neo4j_database, entities_to_migrate)

        # Migrate relationships to Neo4j (batched UNWIND operations)
        logger.info("Migrating relationships to Neo4j...")
        # Load full cooccurrence data for relationships
        # (includes doc_ids_sample and evidence_sentences)
        with Session(engine) as session:
            cooccurrence_map = {
                (c.entity_a_id, c.entity_b_id): c
                for c in session.exec(select(Cooccurrence)).all()
            }

        _migrate_relationships(
            driver, neo4j_database, filtered_relationships, cooccurrence_map
        )

        stats = {
            "entities_created": len(entities_to_migrate),
            "relationships_created": len(filtered_relationships),
            "entities_filtered": 0,  # No longer filtering entities
            "relationships_filtered": filtered_count,
        }

        logger.info(
            f"Migration complete: {stats['entities_created']} entities, "
            f"{stats['relationships_created']} relationships"
        )

        return stats

    finally:
        driver.close()
        logger.info("Neo4j connection closed")


def _migrate_entities(
    driver: Driver, database: str, entities: list[Entity], cypher_batch_size: int = 500
) -> None:
    """Migrate entity nodes to Neo4j using MERGE for idempotency.

    Uses batched UNWIND operations for better performance with large datasets.

    Args:
        driver: Neo4j driver instance
        database: Target database name
        entities: List of Entity objects to migrate
        cypher_batch_size: Number of entities to process per Cypher UNWIND query (default: 500)
    """

    def _merge_entities_batch_tx(tx: Transaction, batch: list[Entity]) -> None:
        # Convert entities to list of dicts for UNWIND
        entity_params = []
        for entity in batch:
            entity_params.append(
                {
                    "id": entity.id,
                    "name": entity.name,
                    "entity_type": entity.entity_type,
                    "umls_cui": entity.umls_cui,
                    "umls_preferred_name": entity.umls_preferred_name,
                    "chebi_id": entity.chebi_id,
                    "go_id": entity.go_id,
                    "mesh_id": entity.mesh_id,
                }
            )

        tx.run(
            """
            UNWIND $entities AS entity
            MERGE (e:Entity {id: entity.id})
            SET e.name = entity.name,
                e.entity_type = entity.entity_type,
                e.umls_cui = entity.umls_cui,
                e.umls_preferred_name = entity.umls_preferred_name,
                e.chebi_id = entity.chebi_id,
                e.go_id = entity.go_id,
                e.mesh_id = entity.mesh_id
            """,
            entities=entity_params,
        )

    # Process entities in batches
    total_migrated = 0
    with driver.session(database=database) as session:
        for i in range(0, len(entities), cypher_batch_size):
            batch = entities[i : i + cypher_batch_size]
            session.execute_write(_merge_entities_batch_tx, batch)
            total_migrated += len(batch)
            if total_migrated % 5000 == 0 or total_migrated == len(entities):
                logger.info(
                    f"Migrated {total_migrated}/{len(entities)} entity nodes "
                    f"({total_migrated/len(entities)*100:.1f}%)"
                )

    logger.info(f"Migrated {len(entities)} entity nodes")


def _migrate_relationships(
    driver: Driver,
    database: str,
    filtered_relationships: list,
    cooccurrence_map: dict,
    cypher_batch_size: int = 500,
) -> None:
    """Migrate co-occurrence relationships to Neo4j using MERGE for idempotency.

    Evidence sentences are parsed directly from Cooccurrence.evidence_sentences,
    which are computed during relation extraction for better performance and
    reusability across different consumers (CLI, notebooks, agent).

    Uses batched UNWIND operations for better performance with large datasets.

    Args:
        driver: Neo4j driver instance
        database: Target database name
        filtered_relationships: List of RelationshipRow objects from FilterRelationships
        cooccurrence_map: Dict mapping (entity_a_id, entity_b_id) to Cooccurrence objects
        cypher_batch_size: Number of relationships to process per Cypher UNWIND query (default: 500)
                          Note: Each relationship creates 2 edges (bidirectional), so actual
                          edges per transaction = cypher_batch_size * 2 = 200 edges.
                          Conservative for AuraDB's 250 MiB limit with evidence_sentences.
    """

    def _merge_relationships_batch_tx(tx: Transaction, batch: list) -> None:
        # Prepare batch parameters
        rel_params = []
        for rel in batch:
            # Get full cooccurrence data
            cooccurrence = cooccurrence_map.get((rel.entity_a_id, rel.entity_b_id))
            if cooccurrence is None:
                logger.warning(
                    f"Skipping relationship {rel.entity_a_id} - {rel.entity_b_id}: "
                    "cooccurrence data not found"
                )
                continue

            # Parse sample document IDs from JSON
            sample_doc_ids = []
            if cooccurrence.doc_ids_sample:
                try:
                    sample_doc_ids = json.loads(cooccurrence.doc_ids_sample)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON in doc_ids_sample for %s - %s",
                        rel.entity_a_id,
                        rel.entity_b_id,
                    )

            # Parse evidence sentences from JSON
            evidence_sentences = []
            if cooccurrence.evidence_sentences:
                try:
                    evidence_sentences = json.loads(cooccurrence.evidence_sentences)
                except json.JSONDecodeError:
                    logger.warning(
                        "Invalid JSON in evidence_sentences for %s - %s",
                        rel.entity_a_id,
                        rel.entity_b_id,
                    )

            rel_params.append(
                {
                    "entity_a_id": rel.entity_a_id,
                    "entity_b_id": rel.entity_b_id,
                    "docs_count": rel.docs_count,
                    "sent_count": rel.sent_count,
                    "sample_doc_ids": sample_doc_ids,
                    "evidence_sentences": evidence_sentences,
                }
            )

        if rel_params:  # Only run if we have valid relationships
            tx.run(
                """
                UNWIND $relationships AS rel
                MATCH (a:Entity {id: rel.entity_a_id})
                MATCH (b:Entity {id: rel.entity_b_id})
                // Create relationship in both directions for symmetric co-occurrence
                MERGE (a)-[r1:CO_OCCURS_WITH]->(b)
                SET r1.docs_count = rel.docs_count,
                    r1.sent_count = rel.sent_count,
                    r1.sample_doc_ids = rel.sample_doc_ids,
                    r1.evidence_sentences = rel.evidence_sentences
                MERGE (b)-[r2:CO_OCCURS_WITH]->(a)
                SET r2.docs_count = rel.docs_count,
                    r2.sent_count = rel.sent_count,
                    r2.sample_doc_ids = rel.sample_doc_ids,
                    r2.evidence_sentences = rel.evidence_sentences
                """,
                relationships=rel_params,
            )

    # Process relationships in batches
    total_migrated = 0
    with driver.session(database=database) as session:
        for i in range(0, len(filtered_relationships), cypher_batch_size):
            batch = filtered_relationships[i : i + cypher_batch_size]
            session.execute_write(_merge_relationships_batch_tx, batch)
            total_migrated += len(batch)
            if total_migrated % 5000 == 0 or total_migrated == len(
                filtered_relationships
            ):
                logger.info(
                    f"Migrated {total_migrated}/{len(filtered_relationships)} relationships "
                    f"({total_migrated/len(filtered_relationships)*100:.1f}%)"
                )

    logger.info(f"Migrated {len(filtered_relationships)} co-occurrence relationships")


def check_connection(uri: str, user: str, password: str) -> bool:
    """Test Neo4j connection.

    Args:
        uri: Neo4j connection URI (e.g., bolt://localhost:7687)
        user: Neo4j username
        password: Neo4j password

    Returns:
        bool: True if connection is successful, False otherwise.
    """
    try:
        driver = get_neo4j_driver(uri, user, password)
        with driver.session() as session:
            # Simple connectivity test
            result = session.run("RETURN 1 as test")
            result.single()
        driver.close()
        logger.info("Neo4j connection test successful")
        return True
    except Exception as e:
        logger.error(f"Neo4j connection test failed: {e}")
        return False
