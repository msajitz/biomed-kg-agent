"""
Core query methods for knowledge graph agent.

Structured Cypher queries for biomedical relationship discovery using
UMLS CUI-based matching (preferred) with CONTAINS name fallback when
CUI resolution is unavailable.
All operations are read-only with mandatory provenance.

Thin wrappers around static Cypher queries - no business logic to unit test.
Validate manually via scripts/verify_agent_queries.py against real Neo4j.
"""

import logging
from typing import Any, Optional

from neo4j import Driver

logger = logging.getLogger(__name__)


def _get_entity_cui(entity: str, linker: Optional[Any]) -> list[str]:
    """Get UMLS CUI(s) for entity text via entity linker.

    Args:
        entity: Entity text to link (e.g., "breast cancer", "BRCA1")
        linker: Optional EntityLinker instance

    Returns:
        List of UMLS CUIs (empty if linking unavailable or fails)
    """
    if not linker:
        return []

    try:
        result = linker.link_entities({entity})
        cui_val = result.get(entity, {}).get("umls_cui")
        if not cui_val:
            return []

        # Normalize: handle single/multiple CUI (currently only single CUI from linker)
        cuis = [cui_val] if isinstance(cui_val, str) else list(cui_val)
        logger.info(f"   Linked '{entity}' -> CUI(s): {cuis}")
        return cuis
    except Exception as e:
        logger.warning(f"Entity linking failed for '{entity}': {e}")

    return []


def _build_entity_match(
    var: str, param: str, value: str, cuis: list[str]
) -> tuple[str, dict[str, Any]]:
    """Build WHERE clause for entity matching: CUI-only or CONTAINS fallback.

    When CUIs are available, matches by CUI only for precision. Falls back
    to CONTAINS substring matching only when CUI resolution fails. The
    CONTAINS path can match multiple source entities, so queries using this
    helper still aggregate by neighbor node to handle that case.

    Args:
        var: Cypher variable name (e.g., "d", "g", "e")
        param: Parameter name (e.g., "disease", "gene", "entity")
        value: Entity search value
        cuis: UMLS CUIs to match (empty if unavailable)

    Returns:
        Tuple of (where_clause, params_dict)
    """
    if cuis:
        where = f"({var}.umls_cui IN ${param}_cuis)"
        params: dict[str, Any] = {f"{param}_cuis": cuis}
    else:
        where = f"toLower({var}.name) CONTAINS toLower(${param})"
        params = {param: value}
    return where, params


def query_disease_genes(
    driver: Driver,
    disease: str,
    min_evidence: int = 5,
    limit: int = 20,
    database: str = "neo4j",
    entity_linker: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Find genes associated with a disease.

    Args:
        driver: Neo4j driver instance
        disease: Disease name or substring
        min_evidence: Minimum document count for relationships
        limit: Maximum results to return
        database: Neo4j database name
        entity_linker: Optional EntityLinker for UMLS CUI resolution

    Returns:
        List of dicts with neighbor_name, disease_name, docs_count, sent_count,
        evidence_sentences, sample_doc_ids
    """
    cuis = _get_entity_cui(disease, entity_linker)
    where_clause, entity_params = _build_entity_match("d", "disease", disease, cuis)
    params = {**entity_params, "min_evidence": min_evidence, "limit": limit}

    # Aggregate by neighbor: ORDER BY + head(collect()) ensures row-coherent
    # results when CONTAINS matches multiple source entities.
    query = f"""
    MATCH (d:Entity {{entity_type: 'disease'}})-[r:CO_OCCURS_WITH]-
        (g:Entity {{entity_type: 'gene'}})
    WHERE {where_clause}
      AND r.docs_count >= $min_evidence
    WITH g, d, r
    ORDER BY r.docs_count DESC
    WITH g,
         head(collect(r.docs_count)) AS docs_count,
         head(collect(r.sent_count)) AS sent_count,
         head(collect(r.evidence_sentences)) AS evidence_sentences,
         head(collect(r.sample_doc_ids[0..5])) AS sample_doc_ids,
         head(collect(d.name)) AS disease_name
    RETURN
      g.name AS neighbor_name,
      disease_name,
      docs_count, sent_count, evidence_sentences, sample_doc_ids
    ORDER BY docs_count DESC
    LIMIT $limit
    """

    with driver.session(database=database) as session:
        result = session.run(query, **params)
        records = result.data()

    logger.info(
        f"query_disease_genes('{disease}', min_evidence={min_evidence}): "
        f"found {len(records)} relationships (CUI: {bool(cuis)})"
    )

    return records


def query_gene_diseases(
    driver: Driver,
    gene: str,
    min_evidence: int = 5,
    limit: int = 20,
    database: str = "neo4j",
    entity_linker: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Find diseases associated with a gene.

    Args:
        driver: Neo4j driver instance
        gene: Gene name or substring
        min_evidence: Minimum document count for relationships
        limit: Maximum results to return
        database: Neo4j database name
        entity_linker: Optional EntityLinker for UMLS CUI resolution

    Returns:
        List of dicts with neighbor_name, gene_name, docs_count, sent_count,
        evidence_sentences, sample_doc_ids
    """
    cuis = _get_entity_cui(gene, entity_linker)
    where_clause, entity_params = _build_entity_match("g", "gene", gene, cuis)
    params = {**entity_params, "min_evidence": min_evidence, "limit": limit}

    # Aggregate by neighbor: ORDER BY + head(collect()) ensures row-coherent
    # results when CONTAINS matches multiple source entities.
    query = f"""
    MATCH (g:Entity {{entity_type: 'gene'}})-[r:CO_OCCURS_WITH]-
        (d:Entity {{entity_type: 'disease'}})
    WHERE {where_clause}
      AND r.docs_count >= $min_evidence
    WITH g, d, r
    ORDER BY r.docs_count DESC
    WITH d,
         head(collect(r.docs_count)) AS docs_count,
         head(collect(r.sent_count)) AS sent_count,
         head(collect(r.evidence_sentences)) AS evidence_sentences,
         head(collect(r.sample_doc_ids[0..5])) AS sample_doc_ids,
         head(collect(g.name)) AS gene_name
    RETURN
      d.name AS neighbor_name,
      gene_name,
      docs_count, sent_count, evidence_sentences, sample_doc_ids
    ORDER BY docs_count DESC
    LIMIT $limit
    """

    with driver.session(database=database) as session:
        result = session.run(query, **params)
        records = result.data()

    logger.info(
        f"query_gene_diseases('{gene}', min_evidence={min_evidence}): "
        f"found {len(records)} relationships (CUI: {bool(cuis)})"
    )

    return records


def query_entity_neighbors(
    driver: Driver,
    entity: str,
    entity_type: Optional[str] = None,
    min_evidence: int = 3,
    limit: int = 20,
    database: str = "neo4j",
    entity_linker: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Find entities that co-occur with a given entity.

    Args:
        driver: Neo4j driver instance
        entity: Entity name or substring
        entity_type: Optional filter by entity type (e.g., 'gene', 'disease')
        min_evidence: Minimum document count for relationships
        limit: Maximum results to return
        database: Neo4j database name
        entity_linker: Optional EntityLinker for UMLS CUI resolution

    Returns:
        List of dicts with entity_name, neighbor_name, neighbor_type, docs_count,
        sent_count, evidence_sentences, sample_doc_ids
    """
    cuis = _get_entity_cui(entity, entity_linker)
    where_clause, entity_params = _build_entity_match("e", "entity", entity, cuis)
    params = {**entity_params, "min_evidence": min_evidence, "limit": limit}

    # Optional neighbor type filter
    type_filter = "AND neighbor.entity_type = $entity_type" if entity_type else ""
    if entity_type:
        params["entity_type"] = entity_type

    # Aggregate by neighbor: ORDER BY + head(collect()) ensures row-coherent
    # results when CONTAINS matches multiple source entities.
    query = f"""
    MATCH (e:Entity)-[r:CO_OCCURS_WITH]-(neighbor:Entity)
    WHERE {where_clause}
      AND r.docs_count >= $min_evidence
      {type_filter}
    WITH e, neighbor, r
    ORDER BY r.docs_count DESC
    WITH neighbor,
         head(collect(r.docs_count)) AS docs_count,
         head(collect(r.sent_count)) AS sent_count,
         head(collect(r.evidence_sentences)) AS evidence_sentences,
         head(collect(r.sample_doc_ids[0..5])) AS sample_doc_ids,
         head(collect(e.name)) AS entity_name
    RETURN
      entity_name,
      neighbor.name AS neighbor_name,
      neighbor.entity_type AS neighbor_type,
      docs_count, sent_count, evidence_sentences, sample_doc_ids
    ORDER BY docs_count DESC
    LIMIT $limit
    """

    with driver.session(database=database) as session:
        result = session.run(query, **params)
        records = result.data()

    logger.info(
        f"query_entity_neighbors('{entity}', type={entity_type}, min_evidence={min_evidence}): "
        f"found {len(records)} neighbors (CUI: {bool(cuis)})"
    )

    return records


def explain_relationship(
    driver: Driver,
    entity_a: str,
    entity_b: str,
    database: str = "neo4j",
    entity_linker: Optional[Any] = None,
) -> dict[str, Any]:
    """Get detailed evidence for a relationship between two entities.

    Args:
        driver: Neo4j driver instance
        entity_a: First entity name or substring
        entity_b: Second entity name or substring
        database: Neo4j database name
        entity_linker: Optional EntityLinker for UMLS CUI resolution

    Returns:
        Dict with entity_a_name, entity_b_name, entity_a_type, entity_b_type,
        docs_count, sent_count, evidence_sentences, sample_doc_ids, found
    """
    cuis_a = _get_entity_cui(entity_a, entity_linker)
    cuis_b = _get_entity_cui(entity_b, entity_linker)

    where_a, params_a = _build_entity_match("a", "entity_a", entity_a, cuis_a)
    where_b, params_b = _build_entity_match("b", "entity_b", entity_b, cuis_b)
    params = {**params_a, **params_b}

    query = f"""
    MATCH (a:Entity)-[r:CO_OCCURS_WITH]-(b:Entity)
    WHERE {where_a}
      AND {where_b}
    RETURN
      a.name as entity_a_name,
      b.name as entity_b_name,
      a.entity_type as entity_a_type,
      b.entity_type as entity_b_type,
      r.docs_count as docs_count,
      r.sent_count as sent_count,
      r.evidence_sentences[0..5] as evidence_sentences,
      r.sample_doc_ids[0..5] as sample_doc_ids
    ORDER BY r.docs_count DESC
    LIMIT 1
    """

    with driver.session(database=database) as session:
        result = session.run(query, **params)
        record = result.single()

        if record:
            explanation = dict(record)
            explanation["found"] = True
            logger.info(
                f"explain_relationship('{entity_a}', '{entity_b}'): "
                f"found {explanation['docs_count']} docs (CUI: {bool(cuis_a or cuis_b)})"
            )
        else:
            explanation = {
                "entity_a_name": entity_a,
                "entity_b_name": entity_b,
                "found": False,
                "docs_count": 0,
                "sent_count": 0,
                "evidence_sentences": [],
                "sample_doc_ids": [],
            }
            logger.info(f"explain_relationship('{entity_a}', '{entity_b}'): not found")

    return explanation


def query_shared_neighbors(
    driver: Driver,
    entity_a: str,
    entity_b: str,
    neighbor_type: Optional[str] = None,
    min_evidence: int = 3,
    limit: int = 20,
    database: str = "neo4j",
    entity_linker: Optional[Any] = None,
) -> list[dict[str, Any]]:
    """Find entities that co-occur with BOTH input entities.

    This performs a graph intersection query: entities in the neighborhood
    of both entity_a AND entity_b. Useful for finding commonalities like
    "genes implicated in both breast and ovarian cancer".

    Args:
        driver: Neo4j driver instance
        entity_a: First entity name or substring
        entity_b: Second entity name or substring
        neighbor_type: Optional filter by entity type (e.g., 'gene', 'disease')
        min_evidence: Minimum document count for relationships
        limit: Maximum results to return
        database: Neo4j database name
        entity_linker: Optional EntityLinker for UMLS CUI resolution

    Returns:
        List of dicts with neighbor_name, neighbor_type, entity_a_docs,
        entity_b_docs, total_evidence, docs_count, sent_count,
        evidence_sentences, sample_doc_ids
    """
    cuis_a = _get_entity_cui(entity_a, entity_linker)
    cuis_b = _get_entity_cui(entity_b, entity_linker)

    where_a, params_a = _build_entity_match("a", "entity_a", entity_a, cuis_a)
    where_b, params_b = _build_entity_match("b", "entity_b", entity_b, cuis_b)
    params = {
        **params_a,
        **params_b,
        "min_evidence": min_evidence,
        "limit": limit,
    }

    # Optional neighbor type filter
    type_filter = "AND n.entity_type = $neighbor_type" if neighbor_type else ""
    if neighbor_type:
        params["neighbor_type"] = neighbor_type

    # Aggregate by neighbor: ORDER BY + head(collect()) ensures row-coherent
    # results when CONTAINS matches multiple source entities.
    query = f"""
    MATCH (a:Entity)-[r1:CO_OCCURS_WITH]-(n:Entity)-[r2:CO_OCCURS_WITH]-(b:Entity)
    WHERE {where_a}
      AND {where_b}
      AND r1.docs_count >= $min_evidence
      AND r2.docs_count >= $min_evidence
      {type_filter}
      AND a <> b
    WITH a, b, n, r1, r2
    ORDER BY r1.docs_count + r2.docs_count DESC
    WITH n,
        head(collect(r1.docs_count)) AS entity_a_docs,
        head(collect(r2.docs_count)) AS entity_b_docs,
        head(collect(r1.sent_count)) AS entity_a_sents,
        head(collect(r2.sent_count)) AS entity_b_sents,
        head(collect(r1.evidence_sentences[0..2])) AS ev_a,
        head(collect(r2.evidence_sentences[0..2])) AS ev_b,
        head(collect(r1.sample_doc_ids[0..3])) AS docs_a,
        head(collect(r2.sample_doc_ids[0..3])) AS docs_b
    RETURN
        n.name AS neighbor_name,
        n.entity_type AS neighbor_type,
        entity_a_docs,
        entity_b_docs,
        entity_a_docs + entity_b_docs AS total_evidence,
        entity_a_docs + entity_b_docs AS docs_count,  // generic key for _format_results
        entity_a_sents + entity_b_sents AS sent_count,
        ev_a + ev_b AS evidence_sentences,
        docs_a + docs_b AS sample_doc_ids
    ORDER BY total_evidence DESC
    LIMIT $limit
    """

    with driver.session(database=database) as session:
        result = session.run(query, **params)
        records = result.data()

    logger.info(
        f"query_shared_neighbors('{entity_a}', '{entity_b}', type={neighbor_type}, "
        f"min_evidence={min_evidence}): found {len(records)} shared neighbors "
        f"(CUI: {bool(cuis_a or cuis_b)})"
    )

    return records
