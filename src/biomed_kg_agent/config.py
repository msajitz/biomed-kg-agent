"""Configuration management for biomed-kg-agent.

Workflow requirements:
  Web UI:        ANTHROPIC_API_KEY, NEO4J_PASSWORD (+ defaults below)
  Neo4j migrate: NEO4J_PASSWORD (URI/USER via CLI or defaults)
  CLI tools:     None (ingest/nlp/kg-relations work with defaults)

Defaults:
  NEO4J_URI=bolt://localhost:7687, NEO4J_USER=neo4j (local Docker; override for AuraDB/cloud)
  SQLITE_DB_PATH=data/nlp.db (local file storage)
  PUBMED_API_BASE=https://eutils.ncbi.nlm.nih.gov/... (NCBI public API)

Optional:
  PUBMED_EMAIL - Recommended for bulk PubMed downloads
  DATABASE_URL - PostgreSQL alternative to SQLite
  ANTHROPIC_MODEL - LLM model for agent (default: claude-haiku-4-5-20251001)
  NEO4J_DATABASE - Neo4j database name (default: neo4j)
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="forbid",
    )

    # Agent/LLM Configuration
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-haiku-4-5-20251001"  # Default model for agent
    ANTHROPIC_MAX_TOKENS: int = (
        4096  # LangChain default (1024) truncates longer responses
    )

    # Neo4j (defaults for local Docker; password required)
    NEO4J_URI: str = "bolt://localhost:7687"
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = ""  # Must set for any deployment
    NEO4J_DATABASE: str = "neo4j"

    # PubMed
    PUBMED_EMAIL: str = ""  # Recommended for bulk downloads to avoid throttling
    PUBMED_API_BASE: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Database (PostgreSQL overrides SQLite if provided)
    DATABASE_URL: str | None = None
    SQLITE_DB_PATH: str = "data/nlp.db"

    # Logging
    LOG_DIR: Path = PROJECT_ROOT / "logs"


settings = Settings()
