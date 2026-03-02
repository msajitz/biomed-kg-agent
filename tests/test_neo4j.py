"""Tests for Neo4j migration functionality.

Tests cover:
- Idempotent MERGE-based migration
- Filter-at-load strategy
- Evidence properties (docs_count, sent_count, top_pmids, evidence_sentences)
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
from neo4j import Driver
from sqlmodel import Session, SQLModel

from biomed_kg_agent.core.database import create_db_engine
from biomed_kg_agent.kg.config import FilterConfig
from biomed_kg_agent.kg.models import Cooccurrence, Entity, Mention
from biomed_kg_agent.neo4j import (
    clear_graph,
    ensure_schema,
    get_neo4j_driver,
    migrate_to_neo4j,
)


@pytest.fixture
def test_db_path(tmp_path: Path) -> str:
    """Create a temporary test database with sample data."""
    db_path = str(tmp_path / "test_kg.db")
    engine = create_db_engine(db_path)
    SQLModel.metadata.create_all(engine)

    # Create sample entities
    entities = [
        Entity(
            id="UMLS:C0006142",
            name="breast neoplasms",
            entity_type="Disease",
            umls_cui="C0006142",
        ),
        Entity(
            id="UMLS:C0376545",
            name="BRCA1",
            entity_type="Gene",
            umls_cui="C0376545",
        ),
        Entity(
            id="UMLS:C0000001",
            name="generic stopword",
            entity_type="Other",
            umls_cui="C0000001",
        ),
    ]

    # Create sample mentions with sentence_text
    mentions = [
        # Mentions for BRCA1 and breast neoplasms co-occurring
        Mention(
            doc_id="12345678",
            entity_id="UMLS:C0376545",
            text="BRCA1",
            sentence_id=0,
            sentence_text="BRCA1 mutations increase breast cancer risk significantly.",
            start_pos=0,
            end_pos=5,
            source_label="GENE",
        ),
        Mention(
            doc_id="12345678",
            entity_id="UMLS:C0006142",
            text="breast cancer",
            sentence_id=0,
            sentence_text="BRCA1 mutations increase breast cancer risk significantly.",
            start_pos=29,
            end_pos=42,
            source_label="DISEASE",
        ),
        Mention(
            doc_id="23456789",
            entity_id="UMLS:C0376545",
            text="BRCA1",
            sentence_id=1,
            sentence_text="Hereditary breast cancer often involves BRCA1 gene defects.",
            start_pos=41,
            end_pos=46,
            source_label="GENE",
        ),
        Mention(
            doc_id="23456789",
            entity_id="UMLS:C0006142",
            text="breast cancer",
            sentence_id=1,
            sentence_text="Hereditary breast cancer often involves BRCA1 gene defects.",
            start_pos=11,
            end_pos=24,
            source_label="DISEASE",
        ),
        # Mentions for generic stopword (below threshold)
        Mention(
            doc_id="99999999",
            entity_id="UMLS:C0000001",
            text="stopword",
            sentence_id=0,
            sentence_text="Generic stopword sentence.",
            start_pos=8,
            end_pos=16,
            source_label="OTHER",
        ),
        Mention(
            doc_id="99999999",
            entity_id="UMLS:C0006142",
            text="breast neoplasms",
            sentence_id=0,
            sentence_text="Generic stopword sentence.",
            start_pos=17,
            end_pos=33,
            source_label="DISEASE",
        ),
    ]

    # Create sample relationships
    cooccurrences = [
        Cooccurrence(
            entity_a_id="UMLS:C0006142",
            entity_b_id="UMLS:C0376545",
            docs_count=5,
            sent_count=10,
            doc_ids_sample=json.dumps(["12345678", "23456789", "34567890"]),
            evidence_sentences=json.dumps(
                [
                    "BRCA1 mutations increase breast cancer risk significantly.",
                    "Hereditary breast cancer often involves BRCA1 gene defects.",
                ]
            ),
        ),
        Cooccurrence(
            entity_a_id="UMLS:C0000001",
            entity_b_id="UMLS:C0006142",
            docs_count=1,  # Below threshold
            sent_count=1,
            doc_ids_sample=json.dumps(["99999999"]),
            evidence_sentences=json.dumps(["Generic stopword sentence."]),
        ),
    ]

    with Session(engine) as session:
        for entity in entities:
            session.add(entity)
        for mention in mentions:
            session.add(mention)
        for cooccurrence in cooccurrences:
            session.add(cooccurrence)
        session.commit()

    return db_path


@pytest.fixture
def mock_neo4j_driver() -> Mock:
    """Create a mock Neo4j driver for testing."""
    mock_driver = Mock(spec=Driver)
    mock_session = MagicMock()

    # Ensure session() returns a context-manager-capable mock
    mock_driver.session.return_value = MagicMock()

    # Setup context manager for session
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_driver.session.return_value.__exit__.return_value = None

    # Setup transaction execution
    mock_session.execute_write.return_value = None
    mock_session.execute_read.return_value = []

    return mock_driver


def test_get_neo4j_driver_success() -> None:
    """Test successful Neo4j driver creation."""
    with patch("biomed_kg_agent.neo4j.GraphDatabase.driver") as mock_driver:
        mock_instance = Mock(spec=Driver)
        mock_driver.return_value = mock_instance

        driver = get_neo4j_driver(
            uri="bolt://localhost:7687",
            user="neo4j",
            password="password",
        )

        assert driver == mock_instance
        mock_driver.assert_called_once_with(
            "bolt://localhost:7687",
            auth=("neo4j", "password"),
        )


def test_ensure_schema(mock_neo4j_driver: Mock) -> None:
    """Test Neo4j schema creation (idempotent)."""
    ensure_schema(mock_neo4j_driver, "neo4j_test")

    # Verify session was opened with correct database
    mock_neo4j_driver.session.assert_called_with(database="neo4j_test")

    # Verify execute_write was called (schema creation happens in write transaction)
    mock_session = mock_neo4j_driver.session.return_value.__enter__.return_value
    assert mock_session.execute_write.called


def test_clear_graph(mock_neo4j_driver: Mock) -> None:
    """Test graph clearing functionality with batched deletes."""
    # Mock the session and transaction behavior
    mock_session = mock_neo4j_driver.session.return_value.__enter__.return_value

    # Mock count query returning 2500 nodes
    mock_session.execute_read.return_value = 2500

    # Mock delete queries - delete 1000, then 1000, then 500, then 0 (done)
    mock_session.execute_write.side_effect = [1000, 1000, 500, 0]

    clear_graph(mock_neo4j_driver, "neo4j_test")

    # Verify session was created with correct database
    mock_neo4j_driver.session.assert_called_with(database="neo4j_test")

    # Verify execute_read was called for counting
    mock_session.execute_read.assert_called_once()

    # Verify execute_write was called 4 times (3 delete batches + 1 final check returning 0)
    assert mock_session.execute_write.call_count == 4


def test_migrate_to_neo4j_happy_path(
    test_db_path: str, mock_neo4j_driver: Mock
) -> None:
    """Test successful migration of entities and relationships.

    Verifies:
    - Entity nodes created with correct properties
    - Relationship edges created with evidence properties
    - Stats returned correctly
    """
    with patch(
        "biomed_kg_agent.neo4j.get_neo4j_driver", return_value=mock_neo4j_driver
    ):
        # Capture Cypher statements executed within write transactions
        mock_session = mock_neo4j_driver.session.return_value.__enter__.return_value
        executed_cypher: list[str] = []

        def execute_write_side_effect(func: Any, *args: Any, **kwargs: Any) -> None:
            tx = MagicMock()
            func(tx, *args)  # Pass through additional args (e.g., batch)
            # Collect all cypher strings passed to tx.run
            for call in tx.run.call_args_list:
                if call.args and isinstance(call.args[0], str):
                    executed_cypher.append(call.args[0])
            return None

        mock_session.execute_write.side_effect = execute_write_side_effect

        stats = migrate_to_neo4j(
            db_path=test_db_path,
            neo4j_uri="bolt://localhost:7687",
            neo4j_user="neo4j",
            neo4j_password="password",
            neo4j_database="neo4j",
            filter_config=FilterConfig(docs_count_min=2, sent_count_min=2),
            clear_existing=False,
        )

        # Verify statistics
        assert stats["entities_created"] == 3  # All entities (including orphaned)
        assert stats["relationships_created"] == 1  # Only above threshold
        assert stats["entities_filtered"] == 0  # No longer filtering entities
        assert stats["relationships_filtered"] == 1  # Below threshold

        # Verify driver methods called
        assert mock_neo4j_driver.session.called
        assert mock_neo4j_driver.close.called

        # Verify that co-occurrence edges are created in both directions
        # Find the cypher block that mentions CO_OCCURS_WITH
        cooc_cyphers = [c for c in executed_cypher if "CO_OCCURS_WITH" in c]
        assert any("MERGE (a)-[r1:CO_OCCURS_WITH]->(b)" in c for c in cooc_cyphers)
        assert any("MERGE (b)-[r2:CO_OCCURS_WITH]->(a)" in c for c in cooc_cyphers)


def test_migrate_to_neo4j_filtering(test_db_path: str, mock_neo4j_driver: Mock) -> None:
    """Test that filtering correctly excludes low-quality relationships."""
    with patch(
        "biomed_kg_agent.neo4j.get_neo4j_driver", return_value=mock_neo4j_driver
    ):
        stats = migrate_to_neo4j(
            db_path=test_db_path,
            neo4j_uri="bolt://localhost:7687",
            filter_config=FilterConfig(
                docs_count_min=5,  # Stricter threshold
                sent_count_min=5,
            ),
        )

        # With stricter filtering, fewer relationships pass
        assert stats["relationships_created"] == 1
        assert stats["relationships_filtered"] == 1


def test_migrate_to_neo4j_clear_existing(
    test_db_path: str, mock_neo4j_driver: Mock
) -> None:
    """Test that clear_existing flag triggers graph clearing."""
    # Mock the session behavior for batched clear
    mock_session = mock_neo4j_driver.session.return_value.__enter__.return_value
    mock_session.execute_read.return_value = 100  # Mock 100 nodes to clear
    mock_session.execute_write.side_effect = [
        100,
        0,
    ]  # Delete 100, then 0 (done), then rest is migration

    # Need to add more return values for the migration operations
    # After the clear (100, 0), we need returns for: indexes, entities, relationships
    mock_session.execute_write.side_effect = [
        100,
        0,  # Clear operations
    ] + [
        None,  # Migration operations (no return values)
    ] * 100  # long enough for migration operations

    with patch(
        "biomed_kg_agent.neo4j.get_neo4j_driver", return_value=mock_neo4j_driver
    ):
        migrate_to_neo4j(
            db_path=test_db_path,
            neo4j_uri="bolt://localhost:7687",
            clear_existing=True,
        )

    # Verify clear_graph was executed (execute_read for count, execute_write for deletes)
    assert mock_session.execute_read.called
    assert mock_session.execute_write.called


def test_migrate_to_neo4j_empty_db(tmp_path: Path) -> None:
    """Test migration with empty database."""
    empty_db = str(tmp_path / "empty.db")
    engine = create_db_engine(empty_db)
    SQLModel.metadata.create_all(engine)

    with patch("biomed_kg_agent.neo4j.get_neo4j_driver") as mock_get_driver:
        mock_driver = Mock(spec=Driver)
        mock_get_driver.return_value = mock_driver
        mock_session = MagicMock()
        # Ensure session() returns a context-manager-capable mock
        mock_driver.session.return_value = MagicMock()
        mock_driver.session.return_value.__enter__.return_value = mock_session
        mock_driver.session.return_value.__exit__.return_value = None

        stats = migrate_to_neo4j(
            db_path=empty_db,
            neo4j_uri="bolt://localhost:7687",
        )

        assert stats["entities_created"] == 0
        assert stats["relationships_created"] == 0
        assert stats["entities_filtered"] == 0
        assert stats["relationships_filtered"] == 0


def test_migrate_to_neo4j_batch_size_warning(
    test_db_path: str, mock_neo4j_driver: Mock, caplog: Any
) -> None:
    """Test that batch_size parameter logs warning in MVP (forward-compat)."""
    with patch(
        "biomed_kg_agent.neo4j.get_neo4j_driver", return_value=mock_neo4j_driver
    ):
        migrate_to_neo4j(
            db_path=test_db_path,
            neo4j_uri="bolt://localhost:7687",
            batch_size=5000,  # MVP ignores this
        )

        # Check warning was logged
        assert any(
            "batch_size parameter is not implemented" in record.message
            for record in caplog.records
        )


def test_migrate_to_neo4j_resume_warning(
    test_db_path: str, mock_neo4j_driver: Mock, caplog: Any
) -> None:
    """Test that resume parameter logs info in MVP (forward-compat)."""
    with patch(
        "biomed_kg_agent.neo4j.get_neo4j_driver", return_value=mock_neo4j_driver
    ):
        migrate_to_neo4j(
            db_path=test_db_path,
            neo4j_uri="bolt://localhost:7687",
            resume=False,
        )

        # Check info was logged
        assert any(
            "resume parameter is not implemented" in record.message
            for record in caplog.records
        )


def test_migrate_to_neo4j_idempotent(
    test_db_path: str, mock_neo4j_driver: Mock
) -> None:
    """Test that running migration twice produces consistent results.

    Uses MERGE semantics - no duplicates created.
    """
    with patch(
        "biomed_kg_agent.neo4j.get_neo4j_driver", return_value=mock_neo4j_driver
    ):
        # First migration
        stats1 = migrate_to_neo4j(
            db_path=test_db_path,
            neo4j_uri="bolt://localhost:7687",
        )

        # Second migration (should be idempotent)
        stats2 = migrate_to_neo4j(
            db_path=test_db_path,
            neo4j_uri="bolt://localhost:7687",
        )

        # Stats should be identical
        assert stats1 == stats2
