#!/usr/bin/env python3
"""
Manual test script for natural language agent with real Neo4j.

Tests the BiomedKGAgent class (LangChain + Anthropic) with various question types.

NOTE: Test entities (breast cancer, BRCA1, HER2, etc.) are examples from a
breast cancer-focused KG. Modify `test_questions` below for your KG's entities.

Usage:
    export NEO4J_URI=<your-uri>
    export NEO4J_USER=neo4j
    export NEO4J_PASSWORD=<your-password>
    export ANTHROPIC_API_KEY=<your-key>

    # Optional: specify model (defaults to claude-haiku-4-5-20251001)
    python scripts/verify_agent.py
    python scripts/verify_agent.py --model "claude-haiku-4-5-20251001"
"""

import argparse
import logging
import os
import sys
import traceback

import dotenv
from neo4j import GraphDatabase

from biomed_kg_agent.agent.core import BiomedKGAgent

logging.basicConfig(level=logging.WARNING)
dotenv.load_dotenv()


def main() -> None:
    """Run manual tests against Neo4j + LLM."""
    # Get Neo4j credentials
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD")

    if not neo4j_password:
        print("Error: NEO4J_PASSWORD environment variable not set")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Manual verification of BiomedKGAgent")
    parser.add_argument(
        "--model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model name (default: claude-haiku-4-5-20251001)",
    )
    args = parser.parse_args()
    model = args.model

    # Check if Anthropic API key is set
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("   Anthropic models require this API key")
        sys.exit(1)

    print(f"Connecting to Neo4j at {neo4j_uri}...")
    print(f"Using model: {model}")

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

    try:
        # Initialize agent with LangChain + Anthropic
        agent = BiomedKGAgent(driver, model=model, verbose=True)

        # Test questions covering different query types
        test_questions = [
            # Should route to query_disease_genes
            "What genes are associated with breast cancer?",
            # Should route to query_gene_diseases
            "What diseases involve BRCA1?",
            # Should route to query_entity_neighbors
            "What is trastuzumab related to in the knowledge graph?",
            # Should route to explain_relationship
            "Explain the connection between HER2 and breast cancer",
            # Fuzzy matching test
            "Tell me about p53 and cancer",
        ]

        for i, question in enumerate(test_questions, 1):
            print("\n" + "=" * 80)
            print(f"Test {i}: {question}")
            print("=" * 80)

            try:
                result = agent.ask(question)

                print("\nAnswer:")
                print(f"{result['answer']}")
                print(
                    f"\nDocument IDs for verification: {', '.join(result['doc_ids'][:5])}"
                )

                if "tool_calls" in result:
                    print("\nTools used:")
                    for tc in result["tool_calls"]:
                        print(f"   - {tc['tool']}: {tc['input']}")

            except Exception as e:
                print(f"\nError: {e}")
                traceback.print_exc()

        print("\n" + "=" * 80)
        print("All tests completed!")
        print("=" * 80)

    except Exception as e:
        print(f"\nFatal error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
