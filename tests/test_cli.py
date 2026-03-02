import pytest
from click.testing import CliRunner

from biomed_kg_agent.cli import cli  # main import removed as it's unused


@pytest.mark.unit
def test_cli_help() -> None:
    """Test CLI shows help message."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Biomedical Knowledge Graph CLI" in result.output


@pytest.mark.unit
def test_ingest_command_help() -> None:
    """Test ingest command help text."""
    runner = CliRunner()
    result = runner.invoke(cli, ["ingest", "--help"])
    assert result.exit_code == 0
    assert "Ingest PubMed abstracts" in result.output


@pytest.mark.unit
def test_ingest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test ingest command delegates to orchestration function and formats output."""
    runner = CliRunner()

    # Mock at the orchestration boundary
    mock_result = {
        "doc_count": 5,
        "output_path": "data/test.db",
        "total_available": 10,
    }
    monkeypatch.setattr(
        "biomed_kg_agent.cli.ingest_pubmed_abstracts",
        lambda term, limit, path, fmt, batch_size, delay: mock_result,
    )

    result = runner.invoke(
        cli,
        [
            "ingest",
            "--term",
            "test query",
            "--limit",
            "10",
            "--path",
            "data/test.db",
        ],
    )

    assert result.exit_code == 0
    assert "Ingesting PubMed abstracts: 'test query' (limit 10)" in result.output
    assert "Saved 5 documents to data/test.db" in result.output


@pytest.mark.unit
def test_migrate_to_neo4j_missing_password() -> None:
    """Test migrate-to-neo4j raises error when password is not provided."""
    runner = CliRunner()

    # Ensure NEO4J_PASSWORD env var is not set
    result = runner.invoke(
        cli,
        [
            "migrate-to-neo4j",
            "--input",
            "data/kg.db",
            # No password provided
        ],
        env={"NEO4J_PASSWORD": ""},  # Explicitly clear the env var
    )

    # Should fail with non-zero exit code
    assert result.exit_code != 0
    assert "Neo4j password required" in result.output
