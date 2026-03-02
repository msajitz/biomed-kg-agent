import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument


@pytest.fixture
def temp_db_path() -> Generator[str, None, None]:
    """Provide a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_two_pass_deps() -> Generator[tuple[MagicMock, MagicMock], None, None]:
    """Fixture that provides mocked BiomedicalNER and EntityLinker."""
    with (
        patch("biomed_kg_agent.nlp.two_pass_processor.BiomedicalNER") as mock_ner_class,
        patch(
            "biomed_kg_agent.nlp.two_pass_processor.EntityLinker"
        ) as mock_linker_class,
    ):
        yield mock_ner_class, mock_linker_class


@pytest.fixture
def sample_pubmed_doc() -> PubmedDocument:
    return PubmedDocument(
        pmid="123456",
        title="Sample PubMed Article",
        abstract="This is a sample abstract.",
        journal="Sample Journal",
        year=2024,
        authors="Doe, John; Smith, Jane",
    )
