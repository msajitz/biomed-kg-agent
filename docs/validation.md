# Validation

This document describes the validation methodology and observed quality metrics for the biomedical knowledge graph (KG) pipeline. Metrics below reflect a single validation run; re-run the [validation notebook](../notebooks/validation_metrics.ipynb) after pipeline or dependency changes.

## Summary Metrics

From a 1,000-abstract validation corpus ("breast cancer AND targeted therapy", 2025-11-18), producing 5,104 unique entities and 31,955 co-occurrence relationships:

| Metric | Value | Notes |
|--------|-------|-------|
| UMLS mention-level coverage | 82.9% | Primary coverage metric |
| UMLS text-level coverage | 76.2% | Lower due to rare/novel terms |
| Entity normalization ratio | 1.72x | Text variants per UMLS concept |
| Multi-paper relationships | 10.4% | Observed in ≥2 independent papers |

**Note on evidence strength:** 89.6% of relationships are observed in a single paper only. Downstream consumers should filter by `docs_count` for higher-confidence results.

## Observed Baselines

The metrics above serve as baselines for this corpus and pipeline version. Significant deviations on a different corpus or after pipeline changes warrant investigation:

- **UMLS mention-level coverage** was 82.9%. A substantial drop may indicate a linking-model or corpus mismatch.
- **UMLS text-level coverage** was 76.2%. Domain-specialized corpora with more novel terminology will produce lower values.
- **Multi-paper relationships** were 10.4%. Lower values indicate insufficient corpus overlap for reliable evidence.

## What This Does Not Measure

These metrics assess linking coverage and co-occurrence volume, not end-to-end accuracy. Specifically, this validation does not evaluate:

- **NER precision/recall** - no held-out annotation set; published scispaCy benchmarks are for different model sizes than those used here.
- **Linking precision** - whether entities map to the *correct* CUI (the linker may over-consolidate compositional phrases).
- **Relationship quality** - no comparison against curated databases (e.g., DrugBank, DisGeNET); co-occurrence counts do not distinguish positive from negative assertions.

## NER Models

The pipeline uses three scispaCy models for biomedical named entity recognition:

| Model | Dataset | Entity Types |
|-------|---------|--------------|
| `en_ner_bc5cdr_md` | BC5CDR | Chemicals, diseases |
| `en_ner_bionlp13cg_md` | BioNLP13CG | Genes, cells, anatomy, organisms, chemicals |
| `en_ner_craft_md` | CRAFT | Genes, proteins, chemicals, taxa, cells, sequences |

**Source:** scispaCy v0.5.4 ([Neumann et al., 2019](https://aclanthology.org/W19-5034/)). See paper for benchmark details.

The pipeline combines all three models with configurable priority. Overlapping extractions are deduplicated based on model priority (see `ner_config.yaml`).

## Entity Filtering

Entities with ≤2 characters are excluded during KG construction. This filters ~2% of extracted entities (acronym fragments, punctuation artifacts).

**Trade-off**: This may filter legitimate short entities (element symbols like "K"), but significantly improves KG quality.

## UMLS Entity Linking

The pipeline links extracted entities to UMLS concepts for standardization. Two coverage metrics:

- **Mention-level coverage**: Percentage of total entity mentions that link to UMLS. High-frequency terms link more reliably.
- **Text-level coverage**: Percentage of unique entity strings that link. Novel/rare terms link less often.

Multiple name variations consolidate to a single canonical entity (e.g., "p53", "TP53" -> CUI C0079419), enabling cross-dataset integration.

## Custom Entities

Entities prefixed with `CUSTOM:` do not link to UMLS. Common reasons:
- Novel terminology not yet in UMLS (e.g., "TNBC", "T-DXd")
- Complex multi-word phrases
- Confidence score below threshold (default: 0.7)

The pipeline retains both UMLS-linked and custom entities.

## Reproducibility

The pipeline is deterministic: identical inputs and configuration produce identical outputs. The only variables are literature source queries, which reflect database state at query time.

To replicate, use the same source document IDs, package versions (`poetry.lock`), and NER configuration.

## Limitations

See [`known_limitations.md`](known_limitations.md) for system limitations.

## References

1. **scispaCy**: Neumann, M., et al. (2019). "ScispaCy: Fast and Robust Models for Biomedical Natural Language Processing." [ACL W19-5034](https://aclanthology.org/W19-5034/)
2. **UMLS**: Bodenreider, O. (2004). "The Unified Medical Language System (UMLS): integrating biomedical terminology." [Nucleic Acids Research, 32(suppl_1), D267–D270](https://doi.org/10.1093/nar/gkh061).
