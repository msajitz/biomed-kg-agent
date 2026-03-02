"""Tests for KG relationship filtering."""

from pathlib import Path

import pytest
from sqlmodel import Session

from biomed_kg_agent.core.database import create_db_engine
from biomed_kg_agent.kg.config import FilterConfig
from biomed_kg_agent.kg.filtering import (
    BIOMEDICAL_STOPWORDS,
    FilterRelationships,
    RelationshipRow,
)
from biomed_kg_agent.kg.models import Cooccurrence, Entity


@pytest.fixture
def sample_entities() -> list[Entity]:
    """Sample entities for testing."""
    return [
        Entity(id="gene:brca1", name="brca1", entity_type="gene", umls_cui="C0376571"),
        Entity(
            id="disease:cancer",
            name="cancer",
            entity_type="disease",
            umls_cui="C0006826",
        ),
        Entity(
            id="chemical:aspirin",
            name="aspirin",
            entity_type="chemical",
            chebi_id="15365",
        ),
        Entity(id="gene:tp53", name="tp53", entity_type="gene", umls_cui="C0080055"),
        Entity(
            id="disease:patients", name="patients", entity_type="disease", umls_cui=None
        ),
    ]


@pytest.fixture
def sample_cooccurrences() -> list[Cooccurrence]:
    """Sample co-occurrences for testing."""
    return [
        Cooccurrence(
            entity_a_id="gene:brca1",
            entity_b_id="disease:cancer",
            docs_count=5,
            sent_count=10,
        ),
        Cooccurrence(
            entity_a_id="gene:brca1",
            entity_b_id="chemical:aspirin",
            docs_count=1,
            sent_count=1,
        ),
        Cooccurrence(
            entity_a_id="gene:tp53",
            entity_b_id="disease:cancer",
            docs_count=3,
            sent_count=4,
        ),
        Cooccurrence(
            entity_a_id="gene:brca1",
            entity_b_id="disease:patients",
            docs_count=10,
            sent_count=15,
        ),
    ]


@pytest.fixture
def test_db(
    tmp_path: Path,
    sample_entities: list[Entity],
    sample_cooccurrences: list[Cooccurrence],
) -> str:
    """Create test database with sample data."""
    db_path = tmp_path / "test.db"
    engine = create_db_engine(db_path=str(db_path))

    # Create tables
    from biomed_kg_agent.kg.models import SQLModel

    SQLModel.metadata.create_all(engine)

    # Insert data
    with Session(engine) as session:
        for entity in sample_entities:
            session.add(entity)
        for cooc in sample_cooccurrences:
            session.add(cooc)
        session.commit()

    return str(db_path)


def test_biomedical_stopwords_present() -> None:
    """Test that stopword list contains expected terms."""
    assert "patients" in BIOMEDICAL_STOPWORDS
    assert "cancer" in BIOMEDICAL_STOPWORDS
    assert "study" in BIOMEDICAL_STOPWORDS
    assert len(BIOMEDICAL_STOPWORDS) > 50  # Reasonably sized list


def test_filter_relationships_init(test_db: str) -> None:
    """Test FilterRelationships initialization."""
    config = FilterConfig(docs_count_min=2, sent_count_min=2)
    filterer = FilterRelationships(test_db, config)

    assert filterer.db_path == test_db
    assert filterer.config.docs_count_min == 2
    assert filterer.config.sent_count_min == 2
    assert filterer.sample_size is None
    assert len(filterer.stopwords) > 0


def test_filter_relationships_load_rows(test_db: str) -> None:
    """Test loading relationships from database."""
    config = FilterConfig()
    filterer = FilterRelationships(test_db, config)

    rows = filterer.load_rows()

    assert len(rows) == 4  # All 4 co-occurrences
    assert all(isinstance(r, RelationshipRow) for r in rows)
    assert rows[0].entity_a_name == "brca1"
    assert rows[0].entity_b_name == "cancer"


def test_filter_relationships_load_rows_with_sampling(test_db: str) -> None:
    """Test sampling behavior when sample_size is set."""
    config = FilterConfig()
    filterer = FilterRelationships(test_db, config, sample_size=2, random_seed=42)

    rows = filterer.load_rows()

    assert len(rows) == 2  # Sampled down to 2
    # Should cache result
    rows2 = filterer.load_rows()
    assert rows == rows2


def test_filter_relationships_apply_filters_docs_count(test_db: str) -> None:
    """Test filtering by docs_count threshold."""
    config = FilterConfig(docs_count_min=3, sent_count_min=1, stopwords_enabled=False)
    filterer = FilterRelationships(test_db, config)

    kept = filterer.apply_filters()

    # Only brca1-cancer (5), tp53-cancer (3), brca1-patients (10) should pass
    assert len(kept) == 3
    assert all(r.docs_count >= 3 for r in kept)


def test_filter_relationships_apply_filters_sent_count(test_db: str) -> None:
    """Test filtering by sent_count threshold."""
    config = FilterConfig(docs_count_min=1, sent_count_min=5, stopwords_enabled=False)
    filterer = FilterRelationships(test_db, config)

    kept = filterer.apply_filters()

    # Only brca1-cancer (10), brca1-patients (15) should pass
    assert len(kept) == 2
    assert all(r.sent_count >= 5 for r in kept)


def test_filter_relationships_apply_filters_stopwords(test_db: str) -> None:
    """Test stopword filtering."""
    config = FilterConfig(docs_count_min=1, sent_count_min=1, stopwords_enabled=True)
    filterer = FilterRelationships(test_db, config)

    kept = filterer.apply_filters()

    # Should filter out relationships with "cancer" and "patients" (both stopwords)
    assert len(kept) == 1
    assert kept[0].entity_a_name == "brca1"
    assert kept[0].entity_b_name == "aspirin"


def test_filter_relationships_apply_filters_entity_type_pairs(test_db: str) -> None:
    """Test entity type pair filtering."""
    config = FilterConfig(
        docs_count_min=1,
        sent_count_min=1,
        stopwords_enabled=False,
        allowed_entity_type_pairs=[("gene", "chemical")],
    )
    filterer = FilterRelationships(test_db, config)

    kept = filterer.apply_filters()

    # Only gene-chemical pairs should pass
    assert len(kept) == 1
    assert kept[0].entity_a_type == "gene"
    assert kept[0].entity_b_type == "chemical"


def test_filter_relationships_apply_and_split(test_db: str) -> None:
    """Test apply_and_split returns kept and removed."""
    config = FilterConfig(docs_count_min=5, sent_count_min=1, stopwords_enabled=False)
    filterer = FilterRelationships(test_db, config)

    kept, removed = filterer.apply_and_split()

    # brca1-cancer (5) and brca1-patients (10) should be kept
    assert len(kept) == 2
    # brca1-aspirin (1) and tp53-cancer (3) should be removed
    assert len(removed) == 2
    assert len(kept) + len(removed) == 4


def test_filter_relationships_removal_reasons(test_db: str) -> None:
    """Test removal_reasons explains why rows are filtered."""
    config = FilterConfig(docs_count_min=3, sent_count_min=5, stopwords_enabled=True)
    filterer = FilterRelationships(test_db, config)

    rows = filterer.load_rows()

    # Find brca1-aspirin (docs=1, sent=1, no stopword)
    aspirin_row = next(r for r in rows if r.entity_b_name == "aspirin")
    reasons = filterer.removal_reasons(aspirin_row)
    assert "docs_count" in reasons
    assert "sent_count" in reasons
    assert "stopword" not in reasons

    # Find brca1-cancer (docs=5, sent=10, stopword="cancer")
    cancer_row = next(
        r for r in rows if r.entity_b_name == "cancer" and r.entity_a_name == "brca1"
    )
    reasons = filterer.removal_reasons(cancer_row)
    assert "docs_count" not in reasons
    assert "sent_count" not in reasons
    assert "stopword" in reasons


def test_filter_relationships_custom_stopwords(test_db: str) -> None:
    """Test using custom stopword set."""
    custom_stopwords = {"aspirin"}  # Only "aspirin" is a stopword
    config = FilterConfig(
        docs_count_min=1,
        sent_count_min=1,
        stopwords_enabled=True,
        stopwords=custom_stopwords,
    )
    filterer = FilterRelationships(test_db, config)

    kept = filterer.apply_filters()

    # Should filter out only brca1-aspirin
    assert len(kept) == 3
    assert all(r.entity_b_name != "aspirin" for r in kept)


def test_relationship_row_model() -> None:
    """Test RelationshipRow Pydantic model."""
    row = RelationshipRow(
        entity_a_id="gene:brca1",
        entity_b_id="disease:cancer",
        docs_count=5,
        sent_count=10,
        entity_a_name="brca1",
        entity_b_name="cancer",
        entity_a_type="gene",
        entity_b_type="disease",
    )

    assert row.entity_a_id == "gene:brca1"
    assert row.docs_count == 5
    assert row.entity_a_type == "gene"


def test_filter_rows_without_loading(test_db: str) -> None:
    """Test filter_rows works independently of load_rows."""
    config = FilterConfig(docs_count_min=3, sent_count_min=1, stopwords_enabled=False)
    filterer = FilterRelationships(test_db, config)

    # Create test rows directly without loading from DB
    test_rows = [
        RelationshipRow(
            entity_a_id="gene:brca1",
            entity_b_id="disease:cancer",
            docs_count=5,
            sent_count=10,
            entity_a_name="brca1",
            entity_b_name="cancer",
            entity_a_type="gene",
            entity_b_type="disease",
        ),
        RelationshipRow(
            entity_a_id="gene:brca1",
            entity_b_id="chemical:aspirin",
            docs_count=1,
            sent_count=1,
            entity_a_name="brca1",
            entity_b_name="aspirin",
            entity_a_type="gene",
            entity_b_type="chemical",
        ),
    ]

    # Filter without loading from DB
    kept = filterer.filter_rows(test_rows)

    # Only the first row should pass (docs_count >= 3)
    assert len(kept) == 1
    assert kept[0].entity_b_name == "cancer"
    assert kept[0].docs_count == 5
