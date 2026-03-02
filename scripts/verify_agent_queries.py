#!/usr/bin/env python3
"""
Manual test script for agent query methods with real Neo4j.

Run this to verify agent queries work before writing integration tests.

NOTE: Test entities (breast cancer, BRCA1, HER2, etc.) are examples from a
breast cancer-focused KG. Modify the entity parameters below for your KG's entities.

Usage:
    export NEO4J_URI=<your-uri>
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=<your-password>
    python scripts/verify_agent_queries.py
"""

import os
import sys

import dotenv
from neo4j import GraphDatabase

from biomed_kg_agent.agent.queries import (
    explain_relationship,
    query_disease_genes,
    query_entity_neighbors,
    query_gene_diseases,
    query_shared_neighbors,
)

dotenv.load_dotenv()


def main() -> None:
    """Run manual tests against Neo4j."""
    # Get Neo4j credentials from environment
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    if not password:
        print("Error: NEO4J_PASSWORD environment variable not set")
        sys.exit(1)

    print(f"Connecting to Neo4j at {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))

    try:
        # Test 1: Query disease genes
        print("\n" + "=" * 80)
        print("Test 1: query_disease_genes('breast cancer', min_evidence=10)")
        print("=" * 80)
        results = query_disease_genes(
            driver, disease="breast cancer", min_evidence=10, limit=5
        )
        print(f"Found {len(results)} results\n")
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. {result['neighbor_name']}")
            print(f"   Disease: {result['disease_name']}")
            print(
                f"   Evidence: {result['docs_count']} docs, {result['sent_count']} sentences"
            )
            if result["evidence_sentences"]:
                print(f"   Sample: {result['evidence_sentences'][0][:100]}...")
            print(f"   PMIDs: {result['sample_doc_ids']}")
            print()

        # Test 2: Query gene diseases
        print("\n" + "=" * 80)
        print("Test 2: query_gene_diseases('BRCA1', min_evidence=5)")
        print("=" * 80)
        results = query_gene_diseases(driver, gene="BRCA1", min_evidence=5, limit=5)
        print(f"Found {len(results)} results\n")
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. {result['neighbor_name']}")
            print(f"   Gene: {result['gene_name']}")
            print(f"   Evidence: {result['docs_count']} docs")
            print()

        # Test 3: Query entity neighbors
        print("\n" + "=" * 80)
        print("Test 3: query_entity_neighbors('trastuzumab', entity_type='gene')")
        print("=" * 80)
        results = query_entity_neighbors(
            driver, entity="trastuzumab", entity_type="gene", min_evidence=5, limit=5
        )
        print(f"Found {len(results)} results\n")
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. {result['neighbor_name']} ({result['neighbor_type']})")
            print(f"   Evidence: {result['docs_count']} docs")
            print()

        # Test 4: Explain relationship
        print("\n" + "=" * 80)
        print("Test 4: explain_relationship('HER2', 'breast cancer')")
        print("=" * 80)
        result = explain_relationship(driver, "HER2", "breast cancer")
        if result["found"]:
            print("Relationship found!")
            print(f"   {result['entity_a_name']} ({result['entity_a_type']})")
            print("   <-->")
            print(f"   {result['entity_b_name']} ({result['entity_b_type']})")
            print(
                f"\n   Evidence: {result['docs_count']} docs, {result['sent_count']} sentences"
            )
            print("\n   Sample sentences:")
            for i, sent in enumerate(result["evidence_sentences"][:3], 1):
                print(f"   {i}. {sent[:100]}...")
            print(f"\n   PMIDs: {result['sample_doc_ids'][:5]}")
        else:
            print("Relationship not found")

        # Test 5: Fuzzy matching for p53 variants
        print("\n" + "=" * 80)
        print("Test 5: Fuzzy matching - query_gene_diseases('p53', min_evidence=3)")
        print("=" * 80)
        results = query_gene_diseases(driver, gene="p53", min_evidence=3, limit=5)
        print(f"Found {len(results)} results (should catch p53 variants)\n")
        # Show which gene variants were matched
        gene_variants = set(r["gene_name"] for r in results)
        print(f"Gene variants matched: {', '.join(sorted(gene_variants)[:10])}")

        # Test 6: Find shared neighbors
        print("\n" + "=" * 80)
        print(
            "Test 6: query_shared_neighbors('breast cancer', 'ovarian cancer', "
            "neighbor_type='gene', min_evidence=5)"
        )
        print("=" * 80)
        results = query_shared_neighbors(
            driver,
            entity_a="breast cancer",
            entity_b="ovarian cancer",
            neighbor_type="gene",
            min_evidence=5,
            limit=10,
        )
        print(f"Found {len(results)} shared genes\n")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result['neighbor_name']} (type: {result['neighbor_type']})")
            print(
                f"   Evidence: {result['entity_a_docs']} docs (breast cancer), "
                f"{result['entity_b_docs']} docs (ovarian cancer)"
            )
            print(f"   Total: {result['total_evidence']} combined")
            print(f"   Sample PMIDs: {result['sample_doc_ids'][:3]}")
            print()

        print("\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
