"""
Database utilities for consistent SQLite connection handling.

This module provides a simple utility function to standardize database
connections across the MVP. Primarily for SQLite consistency and testing.
"""

from sqlalchemy import Engine
from sqlmodel import create_engine


def create_db_engine(
    db_path: str | None = None, database_url: str | None = None, echo: bool = False
) -> Engine:
    """
    Create a database engine with consistent connection handling.

    MVP-focused utility for SQLite connections with optional flexibility.

    Args:
        db_path: SQLite database file path (used if database_url is None)
        database_url: Full database URL (mainly for testing with :memory:)
        echo: Whether to log SQL statements (useful for debugging)

    Returns:
        SQLAlchemy Engine instance

    Examples:
        # SQLite file (primary MVP usage)
        engine = create_db_engine(db_path="data/kg.db")

        # In-memory SQLite (testing)
        engine = create_db_engine(database_url="sqlite:///:memory:")
    """
    if database_url is not None:
        return create_engine(database_url, echo=echo)

    if db_path is not None:
        return create_engine(f"sqlite:///{db_path}", echo=echo)

    raise ValueError("Either db_path or database_url must be provided")
