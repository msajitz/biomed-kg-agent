# Scripts Directory

Utility scripts for the biomedical knowledge graph system.

## Analysis

### `benchmark_scispacy_models.py`
Benchmarks scispaCy models for speed, memory, and entity extraction coverage.

```bash
poetry run python scripts/benchmark_scispacy_models.py
```

**Output:** JSON file with timing, memory usage, and entity counts per model.

## Manual Verification

Scripts for testing against live infrastructure (Neo4j, LLMs). These are not automated tests.

**Note:** Test entities (breast cancer, BRCA1, HER2, trastuzumab) are examples from a breast cancer-focused KG. Modify entity parameters in each script to match your KG's data.

### `verify_agent_queries.py`
Tests low-level Cypher query functions against live Neo4j database. Validates that agent query methods (disease -> genes, gene -> diseases, neighbors, explain relationships) work correctly against real Neo4j schema.

**Requires:** Neo4j with data, env vars `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`

**Usage:**
```bash
export NEO4J_URI=bolt://localhost:7687
export NEO4J_USER=neo4j
export NEO4J_PASSWORD=your_password
poetry run python scripts/verify_agent_queries.py
```

**Output:** Query results with entity counts, relationship counts, and sample evidence sentences.

### `verify_agent.py`
Tests high-level LangChain agent with LLM reasoning and tool routing. Validates that `BiomedKGAgent` correctly routes questions to appropriate tools and generates quality answers.

**Requires:** Neo4j with data, Anthropic API key, env vars `NEO4J_*`, `ANTHROPIC_API_KEY`

**Usage:**
```bash
export ANTHROPIC_API_KEY=your_key
poetry run python scripts/verify_agent.py
poetry run python scripts/verify_agent.py --model "claude-haiku-4-5-20251001"
```

**Output:** Agent responses with tool routing decisions, query results, and cited provenance.

### `verify_model_entity_types.py`
Validates entity types produced by each scispaCy model. Quick verification that model behavior matches descriptions in `docs/validation.md`.

**Requires:** scispaCy models installed (bc5cdr, bionlp13cg, craft)

**Usage:**
```bash
poetry run python scripts/verify_model_entity_types.py
```

**Output:** Entity labels and examples from each of the 3 NER models, plus overlap analysis.

## Utility

### `load_env.sh`
Loads environment variables from `.env` file into current shell.

**Usage:**
```bash
# Load default .env
source scripts/load_env.sh

# Load custom file
source scripts/load_env.sh path/to/custom.env
```

**Note:** Must be sourced (not executed) to affect current shell environment.
