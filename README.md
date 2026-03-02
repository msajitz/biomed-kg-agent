# Biomedical Knowledge Graph Builder with LLM Agent

[![CI](https://github.com/msajitz/biomed-kg-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/msajitz/biomed-kg-agent/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Build knowledge graphs from biomedical literature and query them with an LLM agent.

Synthesizing findings across thousands of biomedical papers is a manual bottleneck. This tool extracts entities and co-occurrence relationships from PubMed abstracts, then exposes the resulting knowledge graph through an LLM agent that answers questions grounded in cited literature with mandatory document provenance.

**TL;DR:** PubMed -> scispaCy NER + UMLS linking -> co-occurrence KG -> Neo4j -> agent query with cited sources.

**Stack:** Python 3.12, scispaCy, UMLS, Neo4j, Claude (Anthropic), Streamlit, SQLite

![Biomedical KG Agent UI](docs/assets/ui-screenshot.png)
*Agent UI: query results with cited evidence from a 20K-abstract knowledge graph*

## Example Agent Output

*Query: "What genes are implicated in both breast cancer and ovarian cancer?"*

<details>
<summary>Full agent response (click to expand)</summary>

**BRIEF SUMMARY**

Several genes are implicated in both breast cancer and ovarian cancer, with the strongest associations being HER2, BRCA1, BRCA2, and TP53 (p53). These genes show combined evidence across 1,500+ documents, with HER2 being the most frequently co-associated gene across both cancer types.

**SELECTION RATIONALE**

Selecting genes with >50 documents of evidence linking them to both breast and ovarian cancer. These represent the most well-established shared genetic factors between the two malignancies. Excluded lower-evidence results (<50 docs) to focus on genes with robust clinical and research documentation.

**DETAILED RESULTS**

- **HER2** — 1,074 documents; HER2 amplification or overexpression occurs in approximately 20-30% of breast cancers and is measured in HER2-overexpressing breast and ovarian cancer cells and tumor xenografts. HER2 serves as an important drug target in both cancer types [30725819, 34957911, 35335736, 35435784, 39600235, 40147045]

- **BRCA1** — 94 documents; BRCA1 germline mutations are associated with increased breast and ovarian cancer risk. BRCA1 and BRCA2 mutations play important roles in the development of both breast and ovarian cancer, and BRCA1-mutant breast cancers carry higher propensity for CNS metastasis [34044732, 35340889, 35343191, 35772246, 36928661, 37334492]

- **BRCA2** — 57 documents; BRCA1 and BRCA2 associated breast cancers carry a remarkably higher propensity of CNS metastasis. Germline variants in BRCA1 and BRCA2 play key roles in the development of both breast and ovarian cancer, with studies showing varying risks based on mutation type [34044732, 35343191, 35764927, 35772246, 39107554, 39461277]

- **p53** — 92 documents; TP53 mutations are enriched in both triple-negative breast cancer and high-grade serous ovarian cancer. Tumors with germline BRCA1/2 mutations show consistent enrichment of TP53 mutations across both cancer types [35393784, 35658233, 35870568, 39001541, 40019487, 40435111]

- **BRCA** — 163 documents; BRCA mutations are associated with both breast and ovarian cancer development. PARP inhibitors have drastically changed the treatment landscape for advanced ovarian tumors with BRCA mutations and are also used in BRCA-mutated breast cancer [34907091, 35340889, 35367197, 35641483, 35897700, 37148685]

- **PARP** — 156 documents; PARP1 shows copy number amplification in approximately 22.8% of breast cancers. PARP inhibitors (olaparib, rucaparib, niraparib) are approved for epithelial ovarian cancer and are also used for breast cancer treatment [35367197, 35442700, 35466854, 35641483, 35670707, 35834102]

</details>

## Use Cases

- **Literature review**: Query relationships across thousands of abstracts
  *"What genes are implicated in both breast cancer and ovarian cancer?"*
- **Pattern exploration**: Explore co-occurrence patterns across genes, diseases, and chemicals
  *"What drugs are associated with triple-negative breast cancer?"*
- **Relationship evidence**: Inspect the literature behind a specific relationship
  *"Explain the relationship between BRCA1 and PARP inhibitors"*

> **Approach:** This is a research exploration tool for biomedical literature analysis. The knowledge graph captures sentence-level co-occurrence relationships.

## Features

- **Literature ingestion** (PubMed) with MeSH terms, DOIs, and author keywords
- **Biomedical NER** via multi-model scispaCy ensemble (chemicals, diseases, genes, anatomy)
- **Co-occurrence extraction** with sentence-level evidence and frequency statistics
- **Knowledge graph** storage in SQL and Neo4j
- **LLM agent** for natural language research queries
- **Streamlit web UI** for interactive querying

All agent queries are pre-validated with mandatory document provenance - no dynamic Cypher generation. See [Agent](docs/agent.md) for details.

## Architecture

```mermaid
graph TD
    A[PubMed API<br/>Fetch abstracts] --> B[(corpus.db<br/>raw abstracts + metadata)]
    B --> C[NER Pipeline<br/>scispaCy multi-model ensemble<br/>+ UMLS entity linking]
    C --> D[(nlp.db<br/>extracted entities + UMLS links)]
    D --> E[KG Builder<br/>co-occurrence extraction<br/>+ relationship statistics]
    E --> F[(kg.db<br/>knowledge graph)]
    F --> G[Migration<br/>filtering + Neo4j import]
    G --> H[(Neo4j<br/>graph database)]
    H --> I[Agent + UI<br/>LLM-powered queries + Streamlit]
```

## Prerequisites

- **Python 3.12**, [Poetry](https://python-poetry.org/)
- **Neo4j 5.x+**: Self-hosted or [Neo4j Aura](https://neo4j.com/cloud/aura/) (required for agent + UI)
- **ANTHROPIC_API_KEY**: Required for the LLM agent (optional for CLI-only pipeline)
- **UMLS**: Entity linking uses scispaCy's built-in UMLS linker - no API key or UMLS license needed

### System Requirements

- **RAM**: 16 GB minimum. The pipeline peaks at ~10 GB for the default 20K-abstract build,
  dominated by the UMLS linker model (~6 GB). Entity data adds ~0.5 GB per 20K abstracts.
  For larger corpora (100K+), 32 GB is recommended. CPU-only; no GPU required.
- **Disk**: ~2 GB one-time download (scispaCy models + UMLS knowledge base),
  plus ~750 MB per 20K abstracts (SQLite databases).

## Getting Started

1. Clone and install:
```bash
git clone https://github.com/msajitz/biomed-kg-agent.git
cd biomed-kg-agent
poetry install
```

2. Run a demo pipeline (~5 min, no Neo4j needed):
```bash
make quick
```
This fetches 100 PubMed abstracts, runs multi-model NER with UMLS entity linking, and builds a co-occurrence knowledge graph - all as SQLite databases in `data/`.

> **Data scale**: For meaningful exploration, use `make build` for a 20K-abstract full pipeline (~2 hours).

3. (Optional) Set up Neo4j + agent:
```bash
cp .env.example .env   # edit with your Neo4j credentials and Anthropic API key
make migrate DIR=<output_dir>
poetry run streamlit run src/biomed_kg_agent/ui/app.py
```
Replace `<output_dir>` with the directory printed by the pipeline (e.g., `data/quick_20250225_143000`).

For step-by-step control and all CLI options: `poetry run biomed-agent --help`

## Makefile Shortcuts

For convenience, the Makefile wraps the most common workflows:

| Command | Description |
|---------|-------------|
| `make quick` | Full pipeline, 100 abstracts (~5 min) |
| `make build` | Full pipeline + Neo4j migration, 20K abstracts (~2 hours) |
| `make demo SEARCH_TERM="diabetes"` | Custom topic demo, 100 abstracts (requires SEARCH_TERM) |
| `make continue DIR=data/...` | Resume from existing run directory |
| `make migrate DIR=data/...` | Migrate a KG to Neo4j |
| `make test` | Run pytest + pre-commit checks |
| `make clean` | Remove generated data |

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NEO4J_URI` | For agent/UI | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | For agent/UI | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | For agent/UI | - | Neo4j password |
| `ANTHROPIC_API_KEY` | For agent/UI | - | Anthropic API key (model config: [Agent docs](docs/agent.md#configuration)) |
| `ANTHROPIC_MODEL` | No | `claude-haiku-4-5-20251001` | LLM model for agent (see [Agent docs](docs/agent.md#configuration)) |
| `PUBMED_EMAIL` | No | - | Recommended for bulk PubMed downloads |
| `SQLITE_DB_PATH` | No | `data/nlp.db` | SQLite path for cited abstract lookup |

See [`.env.example`](.env.example) for a quick-start template.

## Development

```bash
poetry install --with dev
make test                      # pytest + pre-commit
poetry run pre-commit run -a   # lint/format checks only
```

211 tests, 85%+ line coverage on pipeline, NER, and KG modules. Agent and UI modules require live services and are validated separately ([agent validation](docs/agent.md#validation)).

## Documentation

- [Agent](docs/agent.md) - Agent architecture, tools, and customization
- [Data Model](docs/data_model.md) - Knowledge graph schema and query patterns
- [Known Limitations](docs/known_limitations.md) - Current limitations and workarounds
- [Validation](docs/validation.md) - NER performance metrics and reproducibility
- [Scripts](scripts/README.md) - Benchmarking, manual verification, and utility scripts
- [Validation Metrics](notebooks/validation_metrics.ipynb) - NER/linking coverage results (see [methodology](docs/validation.md))
- [Agent Tool Routing & Grounding Validation](notebooks/agent_tool_routing_grounding_validation.ipynb) - Tool-routing validation and hallucination spot-checks

To run notebooks: `poetry install --with dev,notebook`

## Acknowledgments

- [scispaCy](https://allenai.github.io/scispacy/) - Neumann et al., 2019. Biomedical NER and entity linking.
- [UMLS](https://www.nlm.nih.gov/research/umls/) - Unified Medical Language System (NLM/NIH)
- [Neo4j](https://neo4j.com/) - Graph database
- [Anthropic Claude](https://www.anthropic.com/) - LLM agent backbone

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
