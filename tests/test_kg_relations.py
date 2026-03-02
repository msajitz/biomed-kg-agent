import json

from biomed_kg_agent.kg.models import Mention
from biomed_kg_agent.kg.relations import extract_cooccurrences


def make_mention(
    doc: str, sent: int, ent_id: str, text: str = "x", sent_text: str = "Test sentence."
) -> Mention:
    return Mention(
        doc_id=doc,
        entity_id=ent_id,
        text=text,
        sentence_id=sent,
        sentence_text=sent_text,
        start_pos=0,
        end_pos=1,
        source_label="TEST",
    )


def test_extract_cooccurrences_counts_and_ordering() -> None:
    # d1 s0: A,B,B (duplicate B) => one pair (A,B) counted once
    m1 = make_mention("d1", 0, "A")
    m2 = make_mention("d1", 0, "B")
    m3 = make_mention("d1", 0, "B")
    # d1 s1: B,C => pair (B,C)
    m4 = make_mention("d1", 1, "B")
    m5 = make_mention("d1", 1, "C")
    # d2 s0: A,C => pair (A,C)
    m6 = make_mention("d2", 0, "A")
    m7 = make_mention("d2", 0, "C")

    co = extract_cooccurrences([m1, m2, m3, m4, m5, m6, m7])
    pairs = {(c.entity_a_id, c.entity_b_id): c for c in co}

    # Expect three pairs: (A,B), (B,C), (A,C)
    assert set(pairs.keys()) == {("A", "B"), ("B", "C"), ("A", "C")}

    # Sent counts: (A,B)=1, (B,C)=1, (A,C)=1
    assert pairs[("A", "B")].sent_count == 1
    assert pairs[("B", "C")].sent_count == 1
    assert pairs[("A", "C")].sent_count == 1

    # Docs counts: (A,B) only d1; (B,C) only d1; (A,C) only d2
    assert pairs[("A", "B")].docs_count == 1
    assert pairs[("B", "C")].docs_count == 1
    assert pairs[("A", "C")].docs_count == 1


def test_extract_cooccurrences_doc_ids_sample_present() -> None:
    # Multiple sentences across two docs for same pair
    # doc_ids_sample is now aligned with evidence_sentences (parallel arrays)
    mentions = [
        make_mention("d1", 0, "X"),
        make_mention("d1", 0, "Y"),
        make_mention("d1", 1, "X"),
        make_mention("d1", 1, "Y"),
        make_mention("d2", 0, "X"),
        make_mention("d2", 0, "Y"),
    ]
    co = extract_cooccurrences(mentions)
    pairs = {(c.entity_a_id, c.entity_b_id): c for c in co}
    c = pairs[("X", "Y")]
    assert c.docs_count == 2
    assert c.doc_ids_sample is not None
    doc_ids = json.loads(c.doc_ids_sample)
    # Round-robin selection: d1, d2, d1 (parallel to evidence sentences)
    assert doc_ids == ["d1", "d2", "d1"]
    # Verify both docs are represented
    assert set(doc_ids) == {"d1", "d2"}


def test_extract_cooccurrences_evidence_sentences() -> None:
    # Test that diverse evidence sentences are extracted
    mentions = [
        make_mention("d1", 0, "A", sent_text="First sentence in doc1."),
        make_mention("d1", 0, "B", sent_text="First sentence in doc1."),
        make_mention("d1", 1, "A", sent_text="Second sentence in doc1."),
        make_mention("d1", 1, "B", sent_text="Second sentence in doc1."),
        make_mention("d2", 0, "A", sent_text="First sentence in doc2."),
        make_mention("d2", 0, "B", sent_text="First sentence in doc2."),
    ]
    co = extract_cooccurrences(mentions)
    pairs = {(c.entity_a_id, c.entity_b_id): c for c in co}
    c = pairs[("A", "B")]

    assert c.evidence_sentences is not None
    evidence = json.loads(c.evidence_sentences)

    # Should have 3 diverse sentences (round-robin across docs)
    assert len(evidence) == 3
    assert "First sentence in doc1." in evidence
    assert "First sentence in doc2." in evidence
    assert "Second sentence in doc1." in evidence


def test_extract_cooccurrences_evidence_sentences_max_limit() -> None:
    # Test that evidence sentences are capped at 5
    mentions = []
    for doc_idx in range(10):  # 10 documents
        for sent_idx in range(2):  # 2 sentences per doc = 20 total
            doc_id = f"d{doc_idx}"
            sent_text = f"Sentence {sent_idx} in doc {doc_idx}."
            mentions.append(make_mention(doc_id, sent_idx, "X", sent_text=sent_text))
            mentions.append(make_mention(doc_id, sent_idx, "Y", sent_text=sent_text))

    co = extract_cooccurrences(mentions)
    pairs = {(c.entity_a_id, c.entity_b_id): c for c in co}
    c = pairs[("X", "Y")]

    assert c.evidence_sentences is not None
    evidence = json.loads(c.evidence_sentences)

    # Should be capped at 5 sentences
    assert len(evidence) == 5
    # Should prioritize diversity across documents (round-robin)
    assert "Sentence 0 in doc 0." in evidence
    assert "Sentence 0 in doc 1." in evidence
