#!/usr/bin/env python3
"""
Benchmark script for comparing different scispaCy models.

This script evaluates various scispaCy models on:
1. Processing speed
2. Entity extraction coverage
3. Entity type diversity
4. Model size
5. Quality of entity extraction on sample biomedical texts

Usage:
    python scripts/benchmark_scispacy_models.py

Note:
    The model download URLs below are pinned to scispaCy v0.5.4.
    Check https://github.com/allenai/scispacy for the latest release
    and update the version string accordingly before installing.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:
    import spacy
except ImportError:
    print("Please install spacy: pip install spacy")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Available scispaCy models (as of v0.5.4)
SCISPACY_MODELS = {
    # Core scientific models
    "en_core_sci_sm": {
        "description": "Small general biomedical model (~15MB)",
        "entities": ["General scientific entities"],
        "size": "small",
        "specialty": "general",
    },
    "en_core_sci_md": {
        "description": "Medium general biomedical model (~50MB)",
        "entities": ["General scientific entities"],
        "size": "medium",
        "specialty": "general",
    },
    "en_core_sci_lg": {
        "description": "Large general biomedical model (~600MB)",
        "entities": ["General scientific entities"],
        "size": "large",
        "specialty": "general",
    },
    # Specialized NER models
    "en_ner_bc5cdr_md": {
        "description": "BC5CDR chemical and disease NER (~50MB)",
        "entities": ["CHEMICAL", "DISEASE"],
        "size": "medium",
        "specialty": "chemical_disease",
    },
    "en_ner_jnlpba_md": {
        "description": "JNLPBA biomedical NER (~50MB)",
        "entities": ["DNA", "RNA", "PROTEIN", "CELL_LINE", "CELL_TYPE"],
        "size": "medium",
        "specialty": "gene_protein",
    },
    "en_ner_craft_md": {
        "description": "CRAFT corpus biomedical NER (~50MB)",
        "entities": [
            "GGP",
            "SO",
            "TAXON",
            "CHEBI",
            "GO_BP",
            "GO_CC",
            "GO_MF",
            "CL",
            "UBERON",
        ],
        "size": "medium",
        "specialty": "ontology_based",
    },
    "en_ner_bionlp13cg_md": {
        "description": "BioNLP 2013 CG NER (~50MB)",
        "entities": [
            "AMINO_ACID",
            "ANATOMICAL_SYSTEM",
            "CANCER",
            "CELL",
            "CELLULAR_COMPONENT",
            "DEVELOPING_ANATOMICAL_STRUCTURE",
            "GENE_OR_GENE_PRODUCT",
            "IMMATERIAL_ANATOMICAL_ENTITY",
            "MULTI-TISSUE_STRUCTURE",
            "ORGAN",
            "ORGANISM",
            "ORGANISM_SUBDIVISION",
            "ORGANISM_SUBSTANCE",
            "PATHOLOGICAL_FORMATION",
            "SIMPLE_CHEMICAL",
            "TISSUE",
        ],
        "size": "medium",
        "specialty": "comprehensive",
    },
    # SciBERT-based models
    "en_core_sci_scibert": {
        "description": "SciBERT-based biomedical model (~500MB)",
        "entities": ["General scientific entities with SciBERT embeddings"],
        "size": "large",
        "specialty": "scibert_general",
    },
}

# Test texts for benchmarking
SAMPLE_TEXTS = [
    # Cancer metabolism text
    """Cancer cells exhibit altered glucose metabolism characterized by increased
    glycolysis and lactate production, even in the presence of oxygen (Warburg
    effect). This metabolic reprogramming involves key enzymes such as hexokinase,
    phosphofructokinase, and pyruvate kinase. Tumor suppressor p53 regulates
    metabolism by modulating TIGAR expression and glycolysis. Metformin, a diabetes
    drug, shows anti-cancer effects by targeting mitochondrial complex I and
    activating AMPK pathway.""",
    # Drug discovery text
    """The development of tyrosine kinase inhibitors like imatinib for chronic
myeloid leukemia treatment represents a major breakthrough in targeted therapy.
Imatinib specifically inhibits BCR-ABL fusion protein by binding to the
ATP-binding site. However, resistance mutations such as T315I in ABL kinase
domain can reduce drug efficacy. Second-generation inhibitors including
dasatinib and nilotinib were developed to overcome resistance.""",
    # Neuroscience text
    """Alzheimer's disease is characterized by accumulation of amyloid-beta
plaques and neurofibrillary tangles containing hyperphosphorylated tau protein
in brain tissue. The amyloid precursor protein (APP) is cleaved by beta-secretase
and gamma-secretase to produce amyloid-beta peptides. Mutations in presenilin-1
and presenilin-2 genes affect gamma-secretase activity and are associated with
familial Alzheimer's disease.""",
    # Immunology text
    """T helper cells differentiate into distinct subsets including Th1, Th2,
Th17, and regulatory T cells (Tregs) based on cytokine environment and
transcription factor expression. Th1 cells express T-bet and produce
interferon-gamma and IL-2. Th2 cells express GATA3 and secrete IL-4, IL-5,
and IL-13. Th17 cells are characterized by RORγt expression and IL-17
production. Tregs express Foxp3 and suppress immune responses.""",
]


class ModelBenchmark:
    """Benchmark class for evaluating scispaCy models."""

    def __init__(self) -> None:
        self.results: dict[str, Any] = {}

    def benchmark_model(self, model_name: str) -> dict[str, Any] | None:
        """Benchmark a single model."""
        logger.info(f"Benchmarking model: {model_name}")

        try:
            # Load model and measure load time
            start_time = time.time()
            nlp = spacy.load(model_name)
            load_time = time.time() - start_time

            results: dict[str, Any] = {
                "model_name": model_name,
                "description": SCISPACY_MODELS.get(model_name, {}).get(
                    "description", "Unknown"
                ),
                "load_time": load_time,
                "entity_types": set(),
                "total_entities": 0,
                "processing_times": [],
                "entities_per_text": [],
                "sample_entities": [],
            }

            # Test on each sample text
            for i, text in enumerate(SAMPLE_TEXTS):
                start_time = time.time()
                doc = nlp(text)
                processing_time = time.time() - start_time

                # Extract entities
                entities = []
                for ent in doc.ents:
                    entity_info = {
                        "text": ent.text,
                        "label": ent.label_,
                        "start": ent.start_char,
                        "end": ent.end_char,
                    }
                    entities.append(entity_info)
                    results["entity_types"].add(ent.label_)

                results["processing_times"].append(processing_time)
                results["entities_per_text"].append(len(entities))
                results["total_entities"] += len(entities)

                # Store sample entities from first text
                if i == 0:
                    results["sample_entities"] = entities[:10]  # First 10 entities

            # Convert set to list for JSON serialization
            results["entity_types"] = list(results["entity_types"])
            results["avg_processing_time"] = sum(results["processing_times"]) / len(
                results["processing_times"]
            )
            results["avg_entities_per_text"] = sum(results["entities_per_text"]) / len(
                results["entities_per_text"]
            )

            return results

        except OSError:
            logger.warning(f"Model {model_name} not available - skipping")
            return None
        except Exception as e:
            logger.error(f"Error benchmarking {model_name}: {e}")
            return None

    def run_benchmarks(self) -> dict[str, Any]:
        """Run benchmarks on all available models."""
        logger.info("Starting scispaCy model benchmarks...")

        for model_name in SCISPACY_MODELS.keys():
            result = self.benchmark_model(model_name)
            if result:
                self.results[model_name] = result

        return self.results

    def print_summary(self) -> None:
        """Print a summary of benchmark results."""
        if not self.results:
            logger.error("No benchmark results available")
            return

        print("\n" + "=" * 80)
        print("SCISPACY MODEL BENCHMARK RESULTS")
        print("=" * 80)

        # Summary table
        print(
            f"\n{'Model':<25} {'Load Time':<12} {'Proc Time':<12} "
            f"{'Entities':<10} {'Types':<8}"
        )
        print("-" * 70)

        for model_name, result in self.results.items():
            load_time = f"{result['load_time']:.2f}s"
            proc_time = f"{result['avg_processing_time']*1000:.1f}ms"
            entities = f"{result['avg_entities_per_text']:.1f}"
            types_count = len(result["entity_types"])

            print(
                f"{model_name:<25} {load_time:<12} {proc_time:<12} "
                f"{entities:<10} {types_count:<8}"
            )

        # Detailed results for each model
        for model_name, result in self.results.items():
            print(f"\n{'-'*60}")
            print(f"MODEL: {model_name}")
            print(f"{'-'*60}")
            print(f"Description: {result['description']}")
            print(f"Load time: {result['load_time']:.2f} seconds")
            print(
                f"Average processing time: "
                f"{result['avg_processing_time']*1000:.1f} ms per text"
            )
            print(f"Total entities extracted: {result['total_entities']}")
            print(f"Average entities per text: {result['avg_entities_per_text']:.1f}")
            print(f"Entity types found: {len(result['entity_types'])}")
            print(f"Entity types: {', '.join(sorted(result['entity_types']))}")

            if result["sample_entities"]:
                print("\nSample entities from first text:")
                for i, ent in enumerate(result["sample_entities"][:5], 1):
                    print(f"  {i}. '{ent['text']}' ({ent['label']})")

    def save_results(self, filename: str = "scispacy_benchmark_results.json") -> None:
        """Save results to JSON file."""
        output_file = Path(__file__).parent / filename

        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Results saved to {output_file}")

    def recommend_models(self) -> None:
        """Provide model recommendations based on benchmark results."""
        if not self.results:
            return

        print(f"\n{'='*60}")
        print("MODEL RECOMMENDATIONS")
        print(f"{'='*60}")

        # Find fastest model
        fastest_model = min(
            self.results.items(), key=lambda x: x[1]["avg_processing_time"]
        )

        # Find model with most entities
        most_entities = max(
            self.results.items(), key=lambda x: x[1]["avg_entities_per_text"]
        )

        # Find model with most entity types
        most_types = max(self.results.items(), key=lambda x: len(x[1]["entity_types"]))

        print(f"\nFASTEST MODEL: {fastest_model[0]}")
        print(
            f"   Processing time: {fastest_model[1]['avg_processing_time']*1000:.1f} ms"
        )
        print(f"   Description: {fastest_model[1]['description']}")

        print(f"\nMOST ENTITIES: {most_entities[0]}")
        print(
            f"   Avg entities per text: {most_entities[1]['avg_entities_per_text']:.1f}"
        )
        print(f"   Description: {most_entities[1]['description']}")

        print(f"\nMOST ENTITY TYPES: {most_types[0]}")
        print(f"   Entity types: {len(most_types[1]['entity_types'])}")
        print(f"   Types: {', '.join(sorted(most_types[1]['entity_types'])[:5])}...")
        print(f"   Description: {most_types[1]['description']}")

        print("\nRECOMMENDATIONS:")
        print(f"   - For speed: Use {fastest_model[0]}")
        print(f"   - For comprehensive entity extraction: Use {most_entities[0]}")
        print(f"   - For entity type diversity: Use {most_types[0]}")
        print("   - For general biomedical text: en_core_sci_sm or en_core_sci_md")
        print("   - For chemical/disease focus: en_ner_bc5cdr_md")
        print("   - For gene/protein focus: en_ner_jnlpba_md")


def main() -> None:
    """Main function to run the benchmarks."""
    benchmark = ModelBenchmark()

    # Run benchmarks
    results = benchmark.run_benchmarks()

    if not results:
        logger.error("No models could be benchmarked. Please install scispaCy models.")
        print("\nTo install models, run:")
        for model in SCISPACY_MODELS.keys():
            print(
                f"  pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
                f"releases/v0.5.4/{model.replace('_', '-')}-0.5.4.tar.gz"
            )
        return

    # Print results
    benchmark.print_summary()
    benchmark.recommend_models()

    # Save results
    benchmark.save_results()

    print(f"\nBenchmark complete! Tested {len(results)} models.")


if __name__ == "__main__":
    main()
