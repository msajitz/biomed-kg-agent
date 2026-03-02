from biomed_kg_agent.core.models import DocumentInternal
from biomed_kg_agent.kg.transforms import extract_entities_and_mentions
from biomed_kg_agent.nlp.models import ExtractedEntity


def make_doc(doc_id: str, entities: list[ExtractedEntity]) -> DocumentInternal:
    return DocumentInternal(
        id=doc_id,
        title="t",
        text="x",
        source="test",
        pub_year=None,
        entities=entities,
    )


def test_extract_entities_and_mentions_first_mention_name_and_ids() -> None:
    e1 = ExtractedEntity(
        text="TP53",
        start_pos=0,
        end_pos=4,
        source_model="m",
        entity_type="gene",
        source_label="GENE",
        doc_id="d1",
        sentence_id=0,
        sentence_text="TP53 is a tumor suppressor gene.",
        umls_cui="C0043227",
        umls_preferred_name=None,
        ner_confidence=None,
        linking_confidence=None,
        chebi_id=None,
        go_id=None,
        mesh_id=None,
    )
    # Same entity with different surface form later
    e2 = ExtractedEntity(
        text="p53",
        start_pos=10,
        end_pos=13,
        source_model="m",
        entity_type="gene",
        source_label="GENE",
        doc_id="d1",
        sentence_id=1,
        sentence_text="The role of p53 in cancer.",
        umls_cui="C0043227",
        umls_preferred_name=None,
        ner_confidence=None,
        linking_confidence=None,
        chebi_id=None,
        go_id=None,
        mesh_id=None,
    )
    # Different entity without UMLS, should fallback to normalized text
    e3 = ExtractedEntity(
        text="Glucose ",
        start_pos=0,
        end_pos=8,
        source_model="m",
        entity_type="chemical",
        source_label="CHEMICAL",
        doc_id="d2",
        sentence_id=0,
        sentence_text="Glucose  metabolism in cells.",
        ner_confidence=None,
        linking_confidence=None,
        umls_cui=None,
        umls_preferred_name=None,
        chebi_id=None,
        go_id=None,
        mesh_id=None,
    )

    d1 = make_doc("d1", [e1, e2])
    d2 = make_doc("d2", [e3])

    entities, mentions = extract_entities_and_mentions([d1, d2])

    # We expect two entities: one for C0043227 and one for CUSTOM:glucose
    entity_map = {e.id: e for e in entities}
    assert set(entity_map.keys()) == {"C0043227", "CUSTOM:glucose"}

    # Without UMLS preferred name, should fall back to first-occurrence naming:
    # TP53 (normalized to tp53)
    assert entity_map["C0043227"].name == "tp53"
    # Glucose normalized
    assert entity_map["CUSTOM:glucose"].name == "glucose"

    # Mentions preserve original text and all occurrences are captured
    assert len(mentions) == 3
    texts = [m.text for m in mentions]
    assert "TP53" in texts and "p53" in texts and "Glucose " in texts


def test_extract_entities_and_mentions_types_and_links() -> None:
    e = ExtractedEntity(
        text="Aspirin",
        start_pos=0,
        end_pos=7,
        source_model="m",
        entity_type="chemical",
        source_label="CHEMICAL",
        doc_id="d1",
        sentence_id=0,
        sentence_text="Aspirin reduces inflammation.",
        chebi_id="CHEBI:15365",
        ner_confidence=None,
        linking_confidence=None,
        umls_cui=None,
        umls_preferred_name=None,
        go_id=None,
        mesh_id=None,
    )
    d = make_doc("d1", [e])

    entities, mentions = extract_entities_and_mentions([d])

    assert len(entities) == 1
    ent = entities[0]
    assert ent.entity_type == "chemical"
    assert ent.chebi_id == "CHEBI:15365"
    assert mentions[0].entity_id == ent.id


def test_extract_entities_with_umls_preferred_name() -> None:
    """Test that UMLS preferred names are used when available."""
    # Entity with UMLS preferred name
    e1 = ExtractedEntity(
        text="cancer",
        start_pos=0,
        end_pos=6,
        source_model="m",
        entity_type="disease",
        source_label="DISEASE",
        doc_id="d1",
        sentence_id=0,
        sentence_text="Cancer is a complex disease.",
        umls_cui="C0006826",
        umls_preferred_name="Malignant Neoplasms",  # UMLS preferred term
        ner_confidence=None,
        linking_confidence=0.95,
        chebi_id=None,
        go_id=None,
        mesh_id=None,
    )
    # Same entity, different text, same preferred name
    e2 = ExtractedEntity(
        text="malignancy",
        start_pos=0,
        end_pos=10,
        source_model="m",
        entity_type="disease",
        source_label="DISEASE",
        doc_id="d2",
        sentence_id=0,
        sentence_text="Malignancy requires treatment.",
        umls_cui="C0006826",
        umls_preferred_name="Malignant Neoplasms",  # Same UMLS preferred term
        ner_confidence=None,
        linking_confidence=0.92,
        chebi_id=None,
        go_id=None,
        mesh_id=None,
    )

    d1 = make_doc("d1", [e1])
    d2 = make_doc("d2", [e2])

    entities, mentions = extract_entities_and_mentions([d1, d2])

    # Should create one entity with UMLS CUI
    assert len(entities) == 1
    entity = entities[0]

    # Should use colloquial name (first occurrence), not UMLS preferred name
    assert entity.id == "C0006826"
    assert entity.name == "cancer"  # Colloquial name (first occurrence, normalized)
    assert entity.umls_preferred_name == "Malignant Neoplasms"  # UMLS stored separately

    # Both mentions should be preserved
    assert len(mentions) == 2
    assert {m.text for m in mentions} == {"cancer", "malignancy"}
