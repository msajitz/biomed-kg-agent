# Known Limitations

## Agent

### Query Patterns

The agent uses pre-defined query tools and cannot perform ad-hoc multi-hop graph exploration. See [Agent](agent.md) for available tools.

### Entity-Light Design

The KG stores relationships with evidence but not standalone entity descriptions. For entity-only questions (e.g., "What is BRCA1?"), the agent synthesizes answers from relationship context.

### Co-occurrence Relationships

Relationships are based on sentence-level co-occurrence, not typed predicates (e.g., "treats", "causes"). The agent interprets evidence sentences to characterize relationships.

### Entity Name Variations

Neither matching method covers all cases (see also [Entity Name Fragmentation](#entity-name-fragmentation)):
- CUI matching misses variants with different CUIs (e.g., "p53" vs "p53-mutant")
- Substring fallback misses synonyms (e.g., "aspirin" vs "acetylsalicylic acid")

### Evidence Sampling

The KG stores up to 5 representative evidence sentences and sample document IDs per relationship (see [Data Model](data_model.md)). The agent's explanations are based on this sample, not the full evidence corpus. For relationships with many supporting documents, the sample may not capture all nuances. Full document counts and sample document IDs (up to 5) are provided for independent verification.

## NER Pipeline

### Memory Scaling

Two-pass processing (NER extraction -> UMLS linking) is memory-efficient during Pass 1
(entities saved to DB in batches, NER models released between passes). However, Pass 2
and result loading still use `.all()` queries that load all entities into memory:

- `_update_entities_in_database`: loads all entities into one ORM session for UMLS updates
- `_load_results_in_order` / `load_entities`: loads all entities to build the return value
- `process_documents` (pipeline.py): holds all `DocumentInternal` objects in memory

**Practical limits (estimated):**
- ~100K abstracts: ~2.5M entities, ~2 GB for entity objects. Works on 16 GB RAM.
- ~500K abstracts: ~12.5M entities, ~10 GB for entity objects. Needs ~32 GB RAM.
- Beyond 500K: likely requires streaming refactor (`yield_per` / batched updates).

## Data Quality

### UMLS Coverage

UMLS linking coverage varies by domain, with lower accuracy for low-frequency mentions. Entities without CUIs rely on NER labels only and are not deduplicated across text variants (~35% of unique entities lack CUIs; measured on a 1,000-abstract demo corpus). String-based deduplication is possible but out of scope for this release.

### UMLS Linking Accuracy

Entity linking can misassign CUIs when names are similar across related but distinct concepts (for example, members of the same receptor or gene family). This may create multiple KG nodes for the same biological entity and lead to duplicate results.

### Entity Type Classification

Each entity is assigned exactly one `entity_type` (e.g., `gene`, `chemical`, `disease`), but most molecular biomedical entities are inherently multi-typed — both across ontologies (PD-L1 is a gene in genomic databases, a protein in UniProt) and within them (BRCA1 carries UMLS types "Gene or Genome" and "Amino Acid, Peptide, or Protein" simultaneously). The assigned type reflects NER model consensus and may not match the user's intended category. Pattern-based corrections handle common compound nouns (e.g., "CDK4/6 Inhibition" gene -> chemical) but assume pharmacological context, and models with limited label sets (e.g., BC5CDR: only `CHEMICAL`/`DISEASE`) may mistype entities when no better label is available.

### Entity Name Fragmentation

Entities without UMLS CUIs cannot be normalized across text variants. Related concepts may appear as separate results — for example, "PARPi" and "poly" are an abbreviation and an NER-split fragment of "poly(ADP-ribose) polymerase" that lack CUIs and are stored as distinct entities instead of being unified. UMLS linking reduces this fragmentation but does not eliminate it (see [UMLS Coverage](#umls-coverage)).

## Metadata Not in Graph

UMLS semantic types (`umls_semantic_types`) are collected during entity linking and stored in the NLP database but are not exported to the knowledge graph. They were originally used for priority-based type assignment, but UMLS concepts often carry multiple semantic types (e.g., BRCA1 maps to both "Gene or Genome" and "Amino Acid, Peptide, or Protein"), which caused systematic misclassification of gene products as chemicals. The types are retained for potential future use.

The entity model also defines fields for cross-ontology identifiers (`chebi_id`, `go_id`, `mesh_id`, `parent_classes`) that are not populated by the current pipeline. These are schema placeholders for future ontology enrichment.

Document-level metadata (journal, authors, DOI, MeSH terms, keywords, document type) is collected from PubMed/PMC and stored in the NLP processing database, but is not represented in the knowledge graph. The graph contains entities and co-occurrence relationships only — there are no document nodes.
