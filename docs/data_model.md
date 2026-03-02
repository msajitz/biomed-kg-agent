# Knowledge Graph Data Model

This document describes the data model for the biomedical knowledge graph (KG), including entity structure, relationships, type classification, and Neo4j implementation details.

## Entity

Represents biomedical entities (genes, diseases, chemicals, etc.) extracted from biomedical literature.

**Properties**:
- `id` (string): Unique identifier (e.g., "UMLS:C0006142")
- `name` (string): Human-readable entity name (e.g., "breast neoplasms")
- `entity_type` (string): Biomedical category (see [Entity Classification](#entity-classification))
- `umls_cui` (string, optional): UMLS Concept Unique Identifier
- `umls_preferred_name` (string, optional): Canonical UMLS preferred term

**Example**:
```cypher
CREATE (e:Entity {
  id: 'UMLS:C0006142',
  name: 'breast neoplasms',
  entity_type: 'disease',
  umls_cui: 'C0006142',
  umls_preferred_name: 'Malignant neoplasm of breast'
})
```

## CO_OCCURS_WITH

Represents co-occurrence relationships between entities in biomedical literature.

**Properties**:
- `docs_count` (integer): Number of documents containing both entities
- `sent_count` (integer): Number of sentences containing both entities
- `sample_doc_ids` (list[string]): Sample document IDs for provenance (up to 5)
- `evidence_sentences` (list[string]): Up to 5 diverse evidence sentences

**Storage**: SQLite preserves all relationships as the source of truth. Neo4j exports apply quality filters (minimum document frequency, stopword removal) and sample evidence (up to 5 document IDs and sentences per relationship).

**Example**:
```cypher
CREATE (e1)-[:CO_OCCURS_WITH {
  docs_count: 15,
  sent_count: 23,
  sample_doc_ids: ['12345678', '23456789', '34567890', '45678901', '56789012'],
  evidence_sentences: [
    'BRCA1 mutations increase breast cancer risk significantly.',
    'Hereditary breast cancer often involves BRCA1 gene defects.',
    'BRCA1 protein dysfunction leads to DNA repair deficiency.',
    'Clinical studies show BRCA1 testing improves outcomes.',
    'BRCA1 carriers have elevated breast neoplasm incidence.'
  ]
}]->(e2)
```

## Entity Classification

Entity types are assigned by the NER model ensemble (BC5CDR, BioNLP13CG, CRAFT) during extraction. Each entity receives a single primary type based on its NER label (e.g., `GENE_OR_GENE_PRODUCT` -> `gene`, `CHEMICAL` -> `chemical`).

**Pattern-based corrections** handle compound noun phrases where the head noun changes the semantic category: "CDK4/6 Inhibition" is labeled `gene` by NER (it contains a gene name) but corrected to `chemical` because the suffix "inhibition" indicates a drug class. Patterns: inhibitor/inhibition -> chemical, antibody/antibodies -> chemical, syndrome -> disease.

**Entity type values**: `chemical`, `gene`, `disease`, `cell_type`, `anatomy`, `biological_process`, `organism`, `cellular_component`, `sequence_feature`

Each entity receives a single primary type to avoid result duplication in queries. UMLS semantic types are collected during NLP processing but are not exported to the graph (see [Known Limitations](known_limitations.md#metadata-not-in-graph)).

## Neo4j Implementation

### Query Patterns

The following examples demonstrate common query patterns. Replace entity names with your search terms.

**Find Entity Relationships with Evidence**:
```cypher
MATCH (e1:Entity)-[r:CO_OCCURS_WITH]-(e2:Entity)
WHERE e1.name CONTAINS 'breast cancer'
RETURN e1.name, e2.name, r.docs_count, r.evidence_sentences
ORDER BY r.docs_count DESC
LIMIT 10
```

**Find Gene-Disease Associations**:
```cypher
MATCH (gene:Entity {entity_type: 'gene'})-[r:CO_OCCURS_WITH]-(disease:Entity {entity_type: 'disease'})
WHERE r.docs_count >= 5
RETURN gene.name, disease.name, r.evidence_sentences, r.sample_doc_ids
ORDER BY r.docs_count DESC
LIMIT 20
```

**Explain Relationship Between Two Entities**:
```cypher
MATCH (e1:Entity)-[r:CO_OCCURS_WITH]-(e2:Entity)
WHERE e1.name = 'BRCA1' AND e2.name = 'breast cancer'
RETURN e1.name, e2.name, r.docs_count, r.sent_count, r.sample_doc_ids, r.evidence_sentences
```

### Schema

The migration script automatically creates the following indexes and constraints:

```cypher
// Unique constraint (required for idempotent MERGE)
CREATE CONSTRAINT entity_id_unique FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Query performance indexes
CREATE INDEX entity_name_index FOR (e:Entity) ON (e.name);
CREATE INDEX entity_type_index FOR (e:Entity) ON (e.entity_type);
CREATE INDEX umls_cui_index FOR (e:Entity) ON (e.umls_cui);
```

## See Also

- [`scripts/verify_agent_queries.py`](../scripts/verify_agent_queries.py): Runnable examples of agent query patterns
- [Validation](validation.md): NER performance metrics and quality assessment
