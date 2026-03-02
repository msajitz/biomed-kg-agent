"""
Core filtering infrastructure for co-occurrence relationships.

Shared by notebooks (exploration) and CLI (production migration).
"""

from __future__ import annotations

import logging
import random

from pydantic import BaseModel
from sqlmodel import Session, select

from biomed_kg_agent.core.database import create_db_engine

from .config import FilterConfig
from .models import Cooccurrence, Entity

logger = logging.getLogger(__name__)


BIOMEDICAL_STOPWORDS: list[str] = [
    # --- Demographics / Study populations ---
    "patients",
    "patient",
    "women",
    "men",
    "subjects",
    "subject",
    "individuals",
    "individual",
    "participants",
    "participant",
    # --- Publication artifacts ---
    "study",
    "studies",
    "analysis",
    "results",
    "conclusion",
    "background",
    "method",
    "methods",
    # --- Generic clinical terms ---
    "treatment",
    "therapy",
    # --- Generic biological material / biofluids ---
    "tissue",
    "tissues",
    "blood",
    "plasma",
    # --- Generic disease hubs ---
    "cancer",
    "cancers",
    "tumor",
    "tumor cells",
    "cancer cells",
    "disease",
    "diseases",
    # --- Generic species / model descriptors ---
    "human",
    "animal",
    "mouse",
    # --- Overly generic biological classes ---
    "genes",
    "protein",
    "proteins",
    "cells",
    "cell",
    "cell type",
    "cell types",
    "cell line",
    "cell lines",
    "cellular",
    "enzyme",
    "enzymes",
    "proteins",
    "enzymatic",
    "organism",
    "organisms",
    # --- Overly generic chemical / molecular classes ---
    "molecule",
    "compounds",
    "drug",
    "drugs",
    "lipid",
    "lipids",
    "fatty acids",
    "amino acids",
    "metabolite",
    "metabolites",
    "molecule",
    "molecules",
    "molecular",
    # --- Generic processes / catch-alls ---
    "metabolism",
    "metabolics",
    "metabolisms",
    "inhibitor",
    "inhibitors",
    "activator",
    "activators",
    "pathway",
    "pathways",
    "levels",
    "expression",
    "expressions",
    "signaling",
    "regulation",
]


class RelationshipRow(BaseModel):
    """Relationship data for filtering and display."""

    entity_a_id: str
    entity_b_id: str
    docs_count: int
    sent_count: int
    entity_a_name: str
    entity_b_name: str
    entity_a_type: str
    entity_b_type: str


class FilterRelationships:
    """Apply filtering to co-occurrence relationships.

    Shared by notebooks (exploration) and CLI (production migration).
    """

    def __init__(
        self,
        db_path: str,
        config: FilterConfig,
        sample_size: int | None = None,
        random_seed: int | None = None,
    ) -> None:
        """Initialize relationship filter.

        Args:
            db_path: Path to SQLite database with entities and co-occurrences
            config: Filter configuration (stopwords, thresholds, entity type pairs)
            sample_size: If provided, randomly sample this many relationships (for notebooks)
            random_seed: Random seed for reproducible sampling
        """
        self.db_path = db_path
        self.config = config
        self.sample_size = sample_size
        self._engine = create_db_engine(db_path=db_path)
        self._sample_cache: list[RelationshipRow] | None = None
        self._rng: random.Random | None = (
            random.Random(random_seed) if random_seed is not None else None
        )

        # Resolve stopwords: use config's custom set, or default to BIOMEDICAL_STOPWORDS
        self._stopwords: set[str] = (
            config.stopwords
            if config.stopwords is not None
            else set(BIOMEDICAL_STOPWORDS)
        )

        # Resolve allowed entity type pairs
        self._allowed_type_pairs: set[tuple[str, str]] | None = (
            set(config.allowed_entity_type_pairs)
            if config.allowed_entity_type_pairs
            else None
        )

    def load_rows(self) -> list[RelationshipRow]:
        """Load relationships from database (full or sampled).

        Returns cached result if already loaded.
        """
        if self._sample_cache is not None:
            return self._sample_cache

        with Session(self._engine) as session:
            # Load all entities into memory (40K scale fits in memory comfortably)
            entities = {e.id: e for e in session.exec(select(Entity)).all()}

            # Load relationships, then down-sample if requested
            rels = session.exec(select(Cooccurrence)).all()
            if self.sample_size is not None and len(rels) > self.sample_size:
                if self._rng is None:
                    rels = random.sample(rels, self.sample_size)
                else:
                    rels = self._rng.sample(rels, self.sample_size)

            rows: list[RelationshipRow] = []
            for r in rels:
                ea = entities.get(r.entity_a_id)
                eb = entities.get(r.entity_b_id)
                if ea is None or eb is None:
                    logger.warning(
                        "Skipping cooccurrence with missing entity: %s (%s) - %s (%s)",
                        r.entity_a_id,
                        "missing" if ea is None else "ok",
                        r.entity_b_id,
                        "missing" if eb is None else "ok",
                    )
                    continue
                rows.append(
                    RelationshipRow(
                        entity_a_id=r.entity_a_id,
                        entity_b_id=r.entity_b_id,
                        docs_count=r.docs_count,
                        sent_count=r.sent_count,
                        entity_a_name=ea.name,
                        entity_b_name=eb.name,
                        entity_a_type=ea.entity_type,
                        entity_b_type=eb.entity_type,
                    )
                )

        self._sample_cache = rows
        logger.info("Loaded %d relationship rows for filtering", len(rows))
        return rows

    def filter_rows(self, rows: list[RelationshipRow]) -> list[RelationshipRow]:
        """Apply all configured filters to provided relationships.

        Args:
            rows: Relationships to filter

        Returns:
            List of relationships that pass all filter criteria.
        """
        kept: list[RelationshipRow] = []
        for row in rows:
            if row.docs_count < self.config.docs_count_min:
                continue
            if row.sent_count < self.config.sent_count_min:
                continue
            if not self._passes_stopword_filter(row):
                continue
            if not self._passes_entity_type_pairs(row):
                continue
            kept.append(row)
        return kept

    def apply_filters(self) -> list[RelationshipRow]:
        """Apply all configured filters to loaded relationships.

        Convenience method that loads data and applies filters.
        For more control, use load_rows() and filter_rows() separately.

        Returns:
            List of relationships that pass all filter criteria.
        """
        return self.filter_rows(self.load_rows())

    def apply_and_split(
        self,
    ) -> tuple[list[RelationshipRow], list[RelationshipRow]]:
        """Apply filters and return (kept, removed) for analysis.

        Useful for notebook quality review and understanding filter impact.
        """
        kept = self.apply_filters()
        kept_set = {(r.entity_a_id, r.entity_b_id) for r in kept}
        removed = [
            r
            for r in self.load_rows()
            if (r.entity_a_id, r.entity_b_id) not in kept_set
        ]
        return kept, removed

    def removal_reasons(self, row: RelationshipRow) -> list[str]:
        """Explain which criteria caused removal for a row.

        Returns list of reason strings: 'docs_count', 'sent_count', 'stopword', 'entity_type_focus'
        """
        reasons: list[str] = []
        if row.docs_count < self.config.docs_count_min:
            reasons.append("docs_count")
        if row.sent_count < self.config.sent_count_min:
            reasons.append("sent_count")
        if not self._passes_stopword_filter(row):
            reasons.append("stopword")
        if not self._passes_entity_type_pairs(row):
            reasons.append("entity_type_focus")
        return reasons

    def _passes_stopword_filter(self, row: RelationshipRow) -> bool:
        """Check if relationship passes stopword filter."""
        if not self.config.stopwords_enabled:
            return True
        a = row.entity_a_name.lower()
        b = row.entity_b_name.lower()
        return not (a in self._stopwords or b in self._stopwords)

    def _passes_entity_type_pairs(self, row: RelationshipRow) -> bool:
        """Check if relationship passes entity type pair filter."""
        if not self._allowed_type_pairs:
            return True
        pair = (row.entity_a_type, row.entity_b_type)
        pair_rev = (row.entity_b_type, row.entity_a_type)
        return pair in self._allowed_type_pairs or pair_rev in self._allowed_type_pairs

    @property
    def stopwords(self) -> set[str]:
        """Expose stopwords for notebook inspection."""
        return self._stopwords
