"""
Core pipeline for biomedical document processing.

This module provides batch document processing using memory-efficient
two-pass NER and entity linking for large document collections.

Functions:
- process_documents(): Memory-efficient batch processing (source-agnostic)
- process_single_document_demo(): Demo helper for single document processing
"""

import logging
from typing import Optional

from biomed_kg_agent.nlp.two_pass_processor import process_documents_two_pass

from .models import DocumentInternal

logger = logging.getLogger(__name__)


def process_documents(
    docs: list[DocumentInternal],
    config_path: Optional[str] = None,
    output_db: Optional[str] = None,
) -> list[DocumentInternal]:
    """
    Process documents with memory-efficient NER and entity linking.

    Uses two-pass processing to efficiently extract and link biomedical entities
    from large document collections while minimizing memory usage.

    Args:
        docs: List of DocumentInternal objects to process
        config_path: Path to NER config file (optional, defaults to ner_config.yaml)
        output_db: Database path for intermediate results

    Returns:
        List of processed DocumentInternal objects with extracted and linked entities

    Example:
        >>> from biomed_kg_agent.core.pipeline import process_documents
        >>> from biomed_kg_agent.core.connectors import pubmed_to_internal
        >>>
        >>> # Convert documents to canonical format
        >>> internal_docs = [pubmed_to_internal(doc) for doc in pubmed_docs]
        >>>
        >>> # Process with NER and linking
        >>> processed_docs = process_documents(internal_docs)
    """
    if not docs:
        logger.info("No documents to process")
        return []

    logger.info(f"Processing {len(docs)} documents with two-pass NER + linking")

    # Extract texts, IDs, and metadata for two-pass processing
    texts = [doc.text for doc in docs]
    doc_ids = [doc.id for doc in docs]
    doc_metadata = {
        doc.id: {
            "title": doc.title,
            "source": doc.source,
            "pub_year": doc.pub_year,
            "extras": doc.extras,
        }
        for doc in docs
    }

    # Run two-pass NER and entity linking with database auto-resume
    entities_by_document, umls_mappings = process_documents_two_pass(
        texts, doc_ids, config_path, output_db, doc_metadata
    )

    # Attach entities to documents (UMLS mappings already applied in two-pass processor)
    for i, doc in enumerate(docs):
        if i < len(entities_by_document):
            # Flatten entity categories into single list
            all_entities = []
            for entity_list in entities_by_document[i].values():
                all_entities.extend(entity_list)
            doc.entities = all_entities
            logger.debug(f"Document {doc.id}: {len(all_entities)} entities")

    logger.info("Document processing complete")
    total_entities = sum(len(doc.entities) for doc in docs)
    logger.info(f"Summary: {len(docs)} documents, {total_entities} total entities")

    return docs


def process_single_document_demo(
    text: str,
    doc_id: str = "demo",
    title: str = "",
    config_path: Optional[str] = None,
) -> DocumentInternal:
    """
    Demo helper for processing a single document text.

    This is a convenience function for demos and testing. For production
    use, prefer process_documents() with proper DocumentInternal objects.

    Args:
        text: Document text to process
        doc_id: Document identifier
        title: Document title (optional)
        config_path: Path to NER config file (optional, defaults to ner_config.yaml)

    Returns:
        Processed DocumentInternal with extracted entities

    Example:
        >>> result = process_single_document_demo("Glucose metabolism in cancer")
        >>> print(f"Found {len(result.entities)} entities")
    """
    doc = DocumentInternal(id=doc_id, title=title, text=text, source="demo")

    processed_docs = process_documents([doc], config_path)
    return processed_docs[0]
