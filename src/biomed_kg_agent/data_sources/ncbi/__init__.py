"""
NCBI Data Sources Module.

This module provides utilities and connectors for accessing NCBI databases
including PubMed and PMC (PubMed Central).

No convenience imports: import explicitly from submodules, e.g.:
  - from biomed_kg_agent.data_sources.ncbi.models import PubmedDocument
  - from biomed_kg_agent.data_sources.ncbi.ncbi_utils import make_ncbi_request, fetch_ids_for_term
  - from biomed_kg_agent.data_sources.ncbi.pubmed import fetch_abstracts, get_pubmed_pmids
  - from biomed_kg_agent.data_sources.ncbi.pmc import fetch_pmc_articles, get_pmc_pmids

Modules:
    models: Data models for NCBI sources (PubmedDocument, etc.)
    ncbi_utils: Shared utilities for NCBI API interactions
    pubmed: PubMed-specific data connector and processing
    pmc: PMC-specific data connector and processing
"""
