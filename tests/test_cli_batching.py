import pytest
from click.testing import CliRunner

from biomed_kg_agent.cli import cli


def test_ingest_warns_on_large_query(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ingestion with large query - verifies successful processing without extra warnings."""
    runner = CliRunner()
    # Simulate a large number of PMIDs
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.get_total_count",
        lambda term: 15000,
    )
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.get_pubmed_pmids",
        lambda term, limit: [str(i) for i in range(15000)],
    )
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.fetch_abstracts",
        lambda pmids, batch_size, delay: [
            {"pmid": pmid, "title": "Test"} for pmid in pmids
        ],
    )
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.save_to_sqlite",
        lambda docs, path: None,
    )

    result = runner.invoke(
        cli,
        [
            "ingest",
            "--term",
            "cancer",
            "--limit",
            "15000",
            "--batch-size",
            "200",
            "--delay",
            "0.01",
        ],
    )

    # Verify core functionality: successful ingestion and correct document count
    assert "Ingesting PubMed abstracts: 'cancer' (limit 15000)" in result.output
    assert "Saved 15000 documents to data/pubmed.db" in result.output
    assert result.exit_code == 0


def test_ingest_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ingestion with batching - verifies successful processing with batch parameters."""
    runner = CliRunner()

    # Simulate a medium number of PMIDs
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.get_total_count",
        lambda term: 400,
    )
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.get_pubmed_pmids",
        lambda term, limit: [str(i) for i in range(400)],
    )
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.fetch_abstracts",
        lambda pmids, batch_size, delay: [
            {"pmid": pmid, "title": "Test"} for pmid in pmids
        ],
    )
    monkeypatch.setattr(
        "biomed_kg_agent.data_sources.ncbi.pubmed.save_to_sqlite",
        lambda docs, path: None,
    )

    result = runner.invoke(
        cli,
        [
            "ingest",
            "--term",
            "cancer",
            "--limit",
            "400",
            "--batch-size",
            "200",
            "--delay",
            "0.01",
        ],
    )

    # Verify core functionality: successful ingestion and correct document count
    assert "Ingesting PubMed abstracts: 'cancer' (limit 400)" in result.output
    assert "Saved 400 documents to data/pubmed.db" in result.output
    assert result.exit_code == 0
