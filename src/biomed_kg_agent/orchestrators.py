"""
Pipeline orchestration functions.

This module provides high-level orchestration functions that coordinate
multiple domain modules to execute complete pipelines.

Functions:
- build_knowledge_graph(): Orchestrate KG construction from NLP results
- run_complete_pubmed_pipeline(): Orchestrate full PubMed pipeline (ingest -> NLP -> KG -> Neo4j)
- continue_pubmed_pipeline_from_checkpoint(): Resume PubMed pipeline from checkpoint

Note: These functions are currently PubMed-specific.
"""

import logging
import os
from pathlib import Path
from typing import Any

from biomed_kg_agent.data_sources.ncbi.pubmed import (
    ingest_pubmed_abstracts,
    process_pubmed_documents_from_db,
)
from biomed_kg_agent.kg.config import FilterConfig
from biomed_kg_agent.kg.persistence import save_kg_data
from biomed_kg_agent.kg.relations import extract_cooccurrences
from biomed_kg_agent.kg.transforms import extract_entities_and_mentions
from biomed_kg_agent.neo4j import migrate_to_neo4j as run_neo4j_migration
from biomed_kg_agent.nlp.persistence import load_processed_documents

logger = logging.getLogger(__name__)


def build_knowledge_graph(
    input_db_path: str,
    output_db_path: str,
) -> dict:
    """Extract co-occurrence relations and build KG from NLP results.

    Orchestrates the complete knowledge graph construction pipeline:
    1. Load processed documents from NLP database
    2. Extract entities and mentions
    3. Extract co-occurrence relationships (sentence-level)
    4. Save complete KG to output database

    Uses sentence-level co-occurrence for relationship detection.

    Args:
        input_db_path: Path to SQLite database containing NLP results
        output_db_path: Path where knowledge graph will be saved

    Returns:
        dict with keys:
            - entities: Number of entities in the graph
            - mentions: Number of entity mentions
            - cooccurrences: Number of co-occurrence relationships
            - output_path: Path where KG was saved
            - method: Extraction method used ("cooccurrence")

    """

    # Load processed documents
    processed_docs = load_processed_documents(input_db_path)
    if not processed_docs:
        logger.warning(f"No processed documents found in {input_db_path}")
        return {
            "entities": 0,
            "mentions": 0,
            "cooccurrences": 0,
            "output_path": output_db_path,
            "method": "cooccurrence",
        }

    logger.info(
        f"Building knowledge graph from {len(processed_docs)} documents "
        f"using co-occurrence method"
    )

    # Extract and save
    entities, mentions = extract_entities_and_mentions(processed_docs)
    cooccurrences = extract_cooccurrences(mentions)
    save_kg_data(entities, mentions, cooccurrences, output_db_path)

    logger.info(
        f"Knowledge graph complete: {len(entities)} entities, "
        f"{len(mentions)} mentions, {len(cooccurrences)} co-occurrences "
        f"saved to {output_db_path}"
    )

    return {
        "entities": len(entities),
        "mentions": len(mentions),
        "cooccurrences": len(cooccurrences),
        "output_path": output_db_path,
        "method": "cooccurrence",
    }


def _get_neo4j_config() -> tuple[str, str, str, str]:
    """Get Neo4j configuration from environment variables.

    Returns:
        Tuple of (uri, user, password, database)
    """
    return (
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        os.getenv("NEO4J_USER", "neo4j"),
        os.getenv("NEO4J_PASSWORD", ""),
        os.getenv("NEO4J_DATABASE", "neo4j"),
    )


def run_complete_pubmed_pipeline(
    search_term: str,
    size: int,
    output_dir: str,
    ner_config: str | None = None,
    migrate_to_neo4j: bool = False,
    clear_neo4j: bool = False,
    filter_config: FilterConfig | None = None,
) -> dict:
    """Run complete PubMed pipeline: ingest -> NLP -> KG -> optional Neo4j.

    Args:
        search_term: PubMed search query
        size: Number of abstracts to fetch
        output_dir: Output directory path
        ner_config: Optional path to NER config file
        migrate_to_neo4j: Whether to migrate to Neo4j after KG construction
        clear_neo4j: Whether to clear existing Neo4j data (requires migrate_to_neo4j=True)
        filter_config: Optional FilterConfig for Neo4j migration. Defaults to
            docs_count_min=2, sent_count_min=2, stopwords_enabled=True

    Returns:
        dict with keys:
            - output_dir: Path to output directory
            - ingest: Dict with ingestion results
            - nlp: Dict with NLP processing results
            - kg: Dict with KG construction results
            - neo4j: Optional dict with Neo4j migration results
            - steps_completed: Number of steps completed

    Example:
        >>> result = run_complete_pubmed_pipeline(
        ...     search_term="cancer treatment",
        ...     size=100,
        ...     output_dir="data/output"
        ... )
        >>> print(f"Completed {result['steps_completed']} steps")
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    corpus_db = str(output_path / "corpus.db")
    nlp_db = str(output_path / "nlp.db")
    kg_db = str(output_path / "kg.db")

    logger.info(
        f"Starting complete pipeline: term='{search_term}', size={size}, output={output_dir}"
    )

    # Step 1: Ingest
    logger.info("Pipeline step 1/4: Ingesting PubMed abstracts")
    ingest_result = ingest_pubmed_abstracts(
        term=search_term,
        limit=size,
        output_path=corpus_db,
        output_format="sqlite",
        batch_size=200,
        delay=0.35,
    )

    # Step 2: NLP processing
    logger.info("Pipeline step 2/4: NLP processing")
    nlp_result = process_pubmed_documents_from_db(corpus_db, nlp_db, ner_config)

    # Step 3: Knowledge graph extraction
    logger.info("Pipeline step 3/4: Building knowledge graph")
    kg_result = build_knowledge_graph(nlp_db, kg_db)

    result = {
        "output_dir": output_dir,
        "ingest": ingest_result,
        "nlp": nlp_result,
        "kg": kg_result,
        "steps_completed": 3,
    }

    # Step 4: Optional Neo4j migration
    if migrate_to_neo4j:
        neo4j_uri, neo4j_user, neo4j_password, neo4j_database = _get_neo4j_config()
        if not neo4j_password:
            logger.warning("NEO4J_PASSWORD not set, skipping Neo4j migration")
            result["neo4j"] = {"skipped": True, "reason": "NEO4J_PASSWORD not set"}
        else:
            logger.info("Pipeline step 4/4: Migrating to Neo4j")
            # Use provided filter_config or defaults
            if filter_config is None:
                filter_config = FilterConfig(
                    docs_count_min=2,
                    sent_count_min=2,
                    stopwords_enabled=True,
                )

            neo4j_stats = run_neo4j_migration(
                db_path=kg_db,
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                filter_config=filter_config,
                clear_existing=clear_neo4j,
                batch_size=None,
            )
            result["neo4j"] = neo4j_stats
            result["steps_completed"] = 4

    logger.info(
        f"Complete pipeline finished: {result['steps_completed']} steps completed"
    )
    return result


def continue_pubmed_pipeline_from_checkpoint(
    directory: str,
    ner_config: str | None = None,
    migrate_to_neo4j: bool = False,
    clear_neo4j: bool = False,
    filter_config: FilterConfig | None = None,
) -> dict:
    """Continue PubMed pipeline from existing directory with checkpoint detection.

    Automatically detects which steps are complete and runs only the missing ones.

    Args:
        directory: Existing pipeline directory path
        ner_config: Optional path to NER config file (used if NLP step missing)
        migrate_to_neo4j: Whether to migrate to Neo4j
        clear_neo4j: Whether to clear existing Neo4j data (requires migrate_to_neo4j=True)
        filter_config: Optional FilterConfig for Neo4j migration. Defaults to
            docs_count_min=2, sent_count_min=2, stopwords_enabled=True

    Returns:
        dict with keys:
            - directory: Pipeline directory path
            - steps_needed: List of step names that were needed
            - steps_completed: Number of steps completed
            - nlp: Optional dict with NLP results (if step was run)
            - kg: Optional dict with KG results (if step was run)
            - neo4j: Optional dict with Neo4j results (if step was run)

    Raises:
        FileNotFoundError: If corpus.db not found in directory

    Example:
        >>> result = continue_pubmed_pipeline_from_checkpoint(
        ...     directory="data/quick_20241114_120000",
        ...     migrate_to_neo4j=True
        ... )
        >>> print(f"Ran {len(result['steps_needed'])} missing steps")
    """
    dir_path = Path(directory)
    corpus_db = dir_path / "corpus.db"
    nlp_db = dir_path / "nlp.db"
    kg_db = dir_path / "kg.db"

    if not corpus_db.exists():
        raise FileNotFoundError(f"corpus.db not found in {directory}")

    # Detect what needs to be done
    steps_needed = []
    if not nlp_db.exists():
        steps_needed.append("nlp")
    if not kg_db.exists():
        steps_needed.append("kg")
    if migrate_to_neo4j:
        steps_needed.append("neo4j")

    if not steps_needed:
        logger.info(f"Continue-pipeline: all steps complete in {directory}")
        return {
            "directory": directory,
            "steps_needed": [],
            "steps_completed": 0,
        }

    logger.info(
        f"Continue-pipeline: running {len(steps_needed)} missing steps: {steps_needed}"
    )

    result: dict[str, Any] = {
        "directory": directory,
        "steps_needed": steps_needed,
        "steps_completed": 0,
    }

    # Run NLP if needed
    if "nlp" in steps_needed:
        logger.info("Continue-pipeline: running NLP processing")
        nlp_result = process_pubmed_documents_from_db(
            str(corpus_db), str(nlp_db), ner_config
        )
        result["nlp"] = nlp_result
        result["steps_completed"] = int(result["steps_completed"]) + 1

    # Run KG extraction if needed
    if "kg" in steps_needed:
        logger.info("Continue-pipeline: building knowledge graph")
        kg_result = build_knowledge_graph(str(nlp_db), str(kg_db))
        result["kg"] = kg_result
        result["steps_completed"] = int(result["steps_completed"]) + 1

    # Run Neo4j migration if requested
    if "neo4j" in steps_needed:
        neo4j_uri, neo4j_user, neo4j_password, neo4j_database = _get_neo4j_config()
        if not neo4j_password:
            logger.warning("NEO4J_PASSWORD not set, skipping Neo4j migration")
            result["neo4j"] = {"skipped": True, "reason": "NEO4J_PASSWORD not set"}
        else:
            logger.info("Continue-pipeline: migrating to Neo4j")
            # Use provided filter_config or defaults
            if filter_config is None:
                filter_config = FilterConfig(
                    docs_count_min=2,
                    sent_count_min=2,
                    stopwords_enabled=True,
                )

            neo4j_stats = run_neo4j_migration(
                db_path=str(kg_db),
                neo4j_uri=neo4j_uri,
                neo4j_user=neo4j_user,
                neo4j_password=neo4j_password,
                neo4j_database=neo4j_database,
                filter_config=filter_config,
                clear_existing=clear_neo4j,
                batch_size=None,
            )
            result["neo4j"] = neo4j_stats
            result["steps_completed"] = int(result["steps_completed"]) + 1

    logger.info(f"Continue-pipeline completed: {result['steps_completed']} steps run")
    return result
