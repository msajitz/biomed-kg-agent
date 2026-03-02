"""CLI for biomed-kg-agent."""

from datetime import datetime

import click
from dotenv import load_dotenv

from biomed_kg_agent.data_sources.ncbi.pubmed import (
    ingest_pubmed_abstracts,
    process_pubmed_documents_from_db,
)
from biomed_kg_agent.kg.config import FilterConfig
from biomed_kg_agent.log import setup_logger
from biomed_kg_agent.neo4j import migrate_to_neo4j as run_neo4j_migration
from biomed_kg_agent.orchestrators import (
    build_knowledge_graph,
    continue_pubmed_pipeline_from_checkpoint,
    run_complete_pubmed_pipeline,
)

load_dotenv()

logger = setup_logger(__name__)


# Pipeline preset configurations
PIPELINE_PRESETS = {
    "quick": {
        "search_term": "breast cancer AND targeted therapy",
        "size": 100,
        "output_prefix": "quick",
    },
    "full": {
        "search_term": "breast cancer AND targeted therapy",
        "size": 20000,
        "output_prefix": "breast_cancer_targeted_therapy",
    },
}


def _show_kg_preview(kg_db_path: str, top_n: int = 3) -> None:
    """Display preview of top entities and relationships for demo."""
    import sqlite3

    try:
        with sqlite3.connect(kg_db_path) as conn:
            cursor = conn.cursor()

            # Top entities by mention count
            cursor.execute(
                """
                SELECT e.name, e.umls_preferred_name, COUNT(m.id)
                FROM entity e JOIN mention m ON e.id = m.entity_id
                GROUP BY e.id ORDER BY COUNT(m.id) DESC LIMIT ?
                """,
                (top_n,),
            )

            click.echo("\nTop Entities:")
            for i, (name, umls, count) in enumerate(cursor.fetchall(), 1):
                label = f"{name} ({umls})" if umls else name
                click.echo(f"   {i}. {label} - {count} mentions")

            # Top relationships by co-occurrence
            cursor.execute(
                """
                SELECT ea.name, ea.umls_preferred_name, eb.name, eb.umls_preferred_name,
                       c.sent_count, c.docs_count
                FROM cooccurrence c
                JOIN entity ea ON c.entity_a_id = ea.id
                JOIN entity eb ON c.entity_b_id = eb.id
                ORDER BY c.sent_count DESC LIMIT ?
                """,
                (top_n,),
            )

            click.echo("\nTop Relationships:")
            for i, (na, ua, nb, ub, sent, docs) in enumerate(cursor.fetchall(), 1):
                click.echo(
                    f"   {i}. {ua or na} <--> {ub or nb} - {sent} co-occurrences in {docs} docs"
                )

    except Exception as e:
        logger.debug(f"Could not generate KG preview: {e}")


@click.group()
def cli() -> None:
    """Biomedical Knowledge Graph CLI."""
    pass


@cli.command()
@click.option("--term", required=True, help="PubMed search term")
@click.option("--limit", default=10, help="Number of results")
@click.option("--out", type=click.Choice(["sqlite", "json"]), default="sqlite")
@click.option("--path", default="data/pubmed.db", help="Output path")
@click.option(
    "--batch-size",
    default=200,
    show_default=True,
    help="Batch size for fetching abstracts",
)
@click.option(
    "--delay",
    default=0.35,
    show_default=True,
    help="Delay (seconds) between API requests",
)
def ingest(
    term: str, limit: int, out: str, path: str, batch_size: int, delay: float
) -> None:
    """Ingest PubMed abstracts."""
    try:
        click.echo(f"Ingesting PubMed abstracts: '{term}' (limit {limit})")
        logger.info(f"Starting PubMed ingestion: term='{term}', limit={limit}")

        result = ingest_pubmed_abstracts(term, limit, path, out, batch_size, delay)

        if result["doc_count"] == 0:
            click.echo("No results found")
            logger.warning(f"No documents found for term '{term}'")
            return

        click.echo(f"Saved {result['doc_count']} documents to {result['output_path']}")
        logger.info(f"Ingestion completed: {result['doc_count']} documents")

    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error(f"Failed to ingest PubMed data: {e}", exc_info=True)
        raise


@cli.command()
@click.option(
    "--input", required=True, help="Input SQLite database with PubMed documents"
)
@click.option(
    "--output", default="data/nlp_results.db", help="Output NLP results database"
)
@click.option(
    "--config-path",
    default=None,
    help="Path to NER config file (default: ner_config.yaml)",
)
def nlp(input: str, output: str, config_path: str | None) -> None:
    """Process documents with NER and entity linking."""
    try:
        click.echo("Running biomedical NER + UMLS linking...\n")
        logger.info(f"Starting NLP processing: input={input}, output={output}")

        result = process_pubmed_documents_from_db(input, output, config_path)

        if result["doc_count"] == 0:
            click.echo("No documents found in database")
            return

        click.echo("\nEntity extraction complete!")
        click.echo(f"   Entities: {result['entity_count']}")
        click.echo(
            f"   UMLS linked: {result['umls_linked']} ({result['umls_rate']:.1f}%)"
        )
        click.echo(f"   Documents: {result['doc_count']}")
        click.echo(f"   Saved to: {output}")

        logger.info(
            f"NLP processing complete: {result['doc_count']} documents, "
            f"{result['entity_count']} entities ({result['umls_linked']} UMLS-linked)"
        )

    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error(f"Failed to process documents with NLP: {e}", exc_info=True)
        raise


@cli.command()
@click.option("--input", required=True, help="Input NLP results database")
@click.option("--output", default="data/kg.db", help="Output knowledge graph database")
def build_kg(input: str, output: str) -> None:
    """Build knowledge graph from NLP-processed documents.

    Extracts entities, mentions, and co-occurrence relationships from processed documents.
    Uses sentence-level co-occurrence for relationship detection.
    """
    try:
        click.echo("Building knowledge graph (co-occurrence extraction)...\n")
        logger.info(f"Starting KG construction: input={input}, output={output}")

        result = build_knowledge_graph(input, output)

        if result["entities"] == 0:
            click.echo("No entities found to build knowledge graph")
            return

        click.echo("\nKnowledge graph built!")
        click.echo(f"   Entities: {result['entities']}")
        click.echo(f"   Relationships: {result['cooccurrences']}")
        click.echo(f"   Mentions: {result['mentions']}")
        click.echo(f"   Saved to: {output}")

        logger.info(
            f"Knowledge graph complete: {result['entities']} entities, "
            f"{result['cooccurrences']} relationships"
        )

    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error(f"Failed to build knowledge graph: {e}", exc_info=True)
        raise


@cli.command()
@click.option(
    "--input", required=True, help="Input knowledge graph SQLite database (kg.db)"
)
@click.option(
    "--neo4j-uri",
    envvar="NEO4J_URI",
    default="bolt://localhost:7687",
    show_default=True,
    help="Neo4j connection URI",
)
@click.option(
    "--neo4j-user",
    envvar="NEO4J_USER",
    default="neo4j",
    show_default=True,
    help="Neo4j username",
)
@click.option(
    "--neo4j-password",
    envvar="NEO4J_PASSWORD",
    default="",
    help="Neo4j password (or use NEO4J_PASSWORD env var)",
)
@click.option(
    "--neo4j-database",
    envvar="NEO4J_DATABASE",
    default="neo4j",
    show_default=True,
    help="Neo4j database name",
)
@click.option(
    "--docs-count-min",
    default=2,
    show_default=True,
    help="Minimum document count for relationships",
)
@click.option(
    "--sent-count-min",
    default=2,
    show_default=True,
    help="Minimum sentence count for relationships",
)
@click.option(
    "--stopwords-enabled/--no-stopwords",
    default=True,
    show_default=True,
    help="Enable biomedical stopword filtering",
)
@click.option(
    "--clear-existing",
    is_flag=True,
    help="Clear existing Neo4j graph before migration (destructive)",
)
@click.option(
    "--batch-size",
    type=int,
    default=None,
    help="[MVP: ignored] Batch size for chunked migration (post-MVP feature)",
)
def migrate_to_neo4j(
    input: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    neo4j_database: str,
    docs_count_min: int,
    sent_count_min: int,
    stopwords_enabled: bool,
    clear_existing: bool,
    batch_size: int | None,
) -> None:
    """Migrate SQLite knowledge graph to Neo4j with filtering.

    Applies filtering during migration to fit within AuraDB free tier constraints
    (~300K relationships). SQLite database retains full fidelity as source-of-truth.
    Migration is idempotent (uses MERGE) and parameterized for safety.

    Neo4j connection can be configured via environment variables:
    - NEO4J_URI (default: bolt://localhost:7687)
    - NEO4J_USER (default: neo4j)
    - NEO4J_PASSWORD (required)
    - NEO4J_DATABASE (default: neo4j)

    Example:
        $ export NEO4J_PASSWORD=your_password
        $ biomed_kg_agent.cli migrate-to-neo4j \\
            --input data/kg.db \\
            --docs-count-min 2 \\
            --sent-count-min 2 \\
            --stopwords-enabled
    """
    try:
        if not neo4j_password:
            raise click.UsageError(
                "Neo4j password required (use --neo4j-password or NEO4J_PASSWORD env var)"
            )

        if clear_existing:
            click.echo("WARNING: --clear-existing will delete all Neo4j data")
            click.confirm("Continue?", abort=True)

        click.echo(f"Migrating to Neo4j: {neo4j_uri} (database: {neo4j_database})")
        logger.info(f"Starting Neo4j migration: input={input}, uri={neo4j_uri}")

        filter_config = FilterConfig(
            docs_count_min=docs_count_min,
            sent_count_min=sent_count_min,
            stopwords_enabled=stopwords_enabled,
        )

        stats = run_neo4j_migration(
            db_path=input,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            neo4j_database=neo4j_database,
            filter_config=filter_config,
            clear_existing=clear_existing,
            batch_size=batch_size,
        )

        click.echo(
            f"Migration complete: {stats['entities_created']:,} entities, "
            f"{stats['relationships_created']:,} relationships"
        )
        logger.info(f"Neo4j migration completed: {stats}")

    except Exception as e:
        click.echo(f"Error: {e}")
        logger.error(f"Failed to migrate to Neo4j: {e}", exc_info=True)
        raise


@cli.command()
@click.option(
    "--preset",
    type=click.Choice(["quick", "full"]),
    help="Predefined pipeline configuration (quick=100 abstracts, full=20000)",
)
@click.option("--search-term", help="PubMed search query (overrides preset)")
@click.option("--size", type=int, help="Number of abstracts (overrides preset)")
@click.option(
    "--output-dir",
    type=click.Path(),
    help="Output directory (auto-generated from preset if not provided)",
)
@click.option(
    "--migrate", is_flag=True, help="Migrate to Neo4j after building knowledge graph"
)
@click.option(
    "--clear-neo4j",
    is_flag=True,
    help="Clear existing Neo4j data before migration (requires --migrate, destructive)",
)
@click.option(
    "--ner-config",
    type=click.Path(exists=True),
    help="Path to custom NER config file (default: ner_config.yaml)",
)
def run_pipeline(
    preset: str | None,
    search_term: str | None,
    size: int | None,
    output_dir: str | None,
    migrate: bool,
    clear_neo4j: bool,
    ner_config: str | None,
) -> None:
    """Run complete pipeline: ingest -> NLP -> KG -> (optional) Neo4j."""
    try:
        if not preset and not (search_term and size):
            raise click.UsageError("Provide --preset OR both --search-term and --size")
        if clear_neo4j and not migrate:
            raise click.UsageError("--clear-neo4j requires --migrate")

        # Resolve configuration
        if preset:
            config = PIPELINE_PRESETS[preset].copy()
            if search_term:
                config["search_term"] = search_term
            if size:
                config["size"] = size
        else:
            config = {
                "search_term": search_term,
                "size": size,
                "output_prefix": "custom",
            }

        # Generate output directory
        if not output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"data/{config['output_prefix']}_{timestamp}"

        click.echo(
            f"Running pipeline: '{config['search_term']}' ({config['size']} abstracts)"
        )
        logger.info(
            f"Starting pipeline: term='{config['search_term']}', size={config['size']}"
        )

        result = run_complete_pubmed_pipeline(
            search_term=str(config["search_term"]),
            size=config["size"],  # type: ignore[arg-type]
            output_dir=output_dir,
            ner_config=ner_config,
            migrate_to_neo4j=migrate,
            clear_neo4j=clear_neo4j,
        )

        click.echo(
            f"\nPipeline complete: "
            f"{result['ingest']['doc_count']} docs, "
            f"{result['nlp']['entity_count']} entities, "
            f"{result['kg']['cooccurrences']} relationships"
        )
        click.echo(f"   Results in {result['output_dir']}")

        # Show quick preview of results
        _show_kg_preview(result["kg"]["output_path"])

        logger.info(f"Pipeline completed: {result['output_dir']}")

    except click.UsageError:
        raise
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


@cli.command()
@click.option(
    "--dir",
    "directory",
    required=True,
    type=click.Path(exists=True),
    help="Existing pipeline directory to continue from",
)
@click.option(
    "--migrate", is_flag=True, help="Migrate to Neo4j after completing pipeline"
)
@click.option(
    "--clear-neo4j",
    is_flag=True,
    help="Clear existing Neo4j data before migration (requires --migrate, destructive)",
)
@click.option(
    "--ner-config",
    type=click.Path(exists=True),
    help="Path to custom NER config file (only used if NLP step is missing)",
)
def continue_pipeline(
    directory: str,
    migrate: bool,
    clear_neo4j: bool,
    ner_config: str | None,
) -> None:
    """Continue pipeline from existing directory with checkpoint detection."""
    try:
        if clear_neo4j and not migrate:
            raise click.UsageError("--clear-neo4j requires --migrate")

        click.echo(f"Continuing pipeline from {directory}")
        logger.info(f"Continue-pipeline: directory={directory}")

        result = continue_pubmed_pipeline_from_checkpoint(
            directory=directory,
            ner_config=ner_config,
            migrate_to_neo4j=migrate,
            clear_neo4j=clear_neo4j,
        )

        if result["steps_completed"] == 0:
            click.echo("All steps complete! Nothing to do.")
            return

        click.echo(
            f"Completed {result['steps_completed']} steps: {', '.join(result['steps_needed'])}"
        )
        click.echo(f"Results in {result['directory']}")

        # Show preview if KG was built
        if "kg" in result.get("steps_needed", []) and "kg" in result:
            _show_kg_preview(result["kg"]["output_path"])

        logger.info(f"Continue-pipeline completed: {result['directory']}")

    except (click.UsageError, click.ClickException, FileNotFoundError):
        raise
    except Exception as e:
        logger.error(f"Continue-pipeline failed: {e}", exc_info=True)
        raise click.ClickException(str(e))


def main() -> None:
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
