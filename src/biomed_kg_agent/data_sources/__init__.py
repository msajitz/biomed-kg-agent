"""Data source integration modules for pipeline processing."""

# Import NCBI models for external use
from .ncbi.models import PubmedDocument

__all__ = [
    "PubmedDocument",
]
