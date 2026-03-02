#!/usr/bin/env python3
"""
Quick verification script to check what entity types each scispaCy model produces.

This helps validate the descriptions in docs/validation.md.

Usage:
    poetry run python scripts/verify_model_entity_types.py
"""

import spacy

# Test text with diverse biomedical entities
TEST_TEXT = """
p53 mutations are associated with breast cancer and lung cancer.
Glucose metabolism involves the liver and pancreas.
COX-2 inhibitors reduce inflammation in cells.
BRCA1 gene encodes a protein involved in DNA repair.
The patient received aspirin and metformin treatment.
T cells and B cells are important for immune response.
Mitochondria generate ATP through cellular respiration.
"""


def test_model(model_name: str) -> dict[str, set[str]]:
    """Load a model and extract entity labels it produces.

    Returns:
        Dictionary mapping entity labels to example entity texts
    """
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print("=" * 80)

    try:
        nlp = spacy.load(model_name)
    except OSError:
        print(f"Model {model_name} not found. Install with:")
        print(
            f"   pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/"
            f"releases/v0.5.4/{model_name.replace('_', '-')}-0.5.4.tar.gz"
        )
        return {}

    doc = nlp(TEST_TEXT)

    # Collect labels and examples
    label_examples: dict[str, set[str]] = {}

    for ent in doc.ents:
        if ent.label_ not in label_examples:
            label_examples[ent.label_] = set()
        label_examples[ent.label_].add(ent.text)

    # Print results
    print(f"\nEntity labels produced by {model_name}:")
    print(f"Total unique labels: {len(label_examples)}")
    print()

    for label in sorted(label_examples.keys()):
        examples = sorted(label_examples[label])[:3]  # Show max 3 examples
        examples_str = ", ".join(f'"{ex}"' for ex in examples)
        print(f"  {label:30s} -> {examples_str}")

    return label_examples


def main() -> None:
    """Test all three models and compare."""

    models = {
        "BC5CDR": "en_ner_bc5cdr_md",
        "BioNLP13CG": "en_ner_bionlp13cg_md",
        "CRAFT": "en_ner_craft_md",
    }

    results = {}

    for display_name, model_name in models.items():
        results[display_name] = test_model(model_name)

    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print("=" * 80)

    for display_name, labels in results.items():
        if labels:
            label_list = ", ".join(sorted(labels.keys()))
            print(f"\n{display_name}:")
            print(f"  Labels: {label_list}")

    # Check for overlaps
    print(f"\n{'='*80}")
    print("LABEL OVERLAPS")
    print("=" * 80)

    if len(results) >= 2:
        all_labels: set[str] = set()
        for labels in results.values():
            all_labels.update(labels.keys())

        print(f"\nAll unique labels across models: {len(all_labels)}")

        # Find labels that appear in multiple models
        for label in sorted(all_labels):
            models_with_label = [
                name for name, labels in results.items() if label in labels
            ]
            if len(models_with_label) > 1:
                print(f"  {label:30s} -> {', '.join(models_with_label)}")


if __name__ == "__main__":
    main()
