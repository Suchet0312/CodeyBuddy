def reciprocal_rank_fusion(
    dense_documents,
    bm25_documents,
    k=60
):
    scores = {}
    documents = {}

    # Process Dense Retrieval results
    for rank, document in enumerate(dense_documents, start=1):

        document_id = document.metadata.get("file")

        documents[document_id] = document

        score = 1 / (k + rank)

        scores[document_id] = (
            scores.get(document_id, 0)
            + score
        )

    # Process BM25 Retrieval results
    for rank, document in enumerate(bm25_documents, start=1):

        document_id = document.metadata.get("file")

        documents[document_id] = document

        score = 1 / (k + rank)

        scores[document_id] = (
            scores.get(document_id, 0)
            + score
        )

    # Sort documents by final RRF score
    ranked_document_ids = sorted(
        scores,
        key=scores.get,
        reverse=True
    )

    # Return documents in hybrid ranking order
    return [
        documents[document_id]
        for document_id in ranked_document_ids
    ]