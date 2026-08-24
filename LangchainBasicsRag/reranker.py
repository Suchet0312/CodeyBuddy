from sentence_transformers import CrossEncoder


def get_reranker():

    reranker = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )

    return reranker


def rerank_documents(
    question,
    documents,
    reranker,
    top_k=3
):

    # Create query-document pairs
    pairs = [
        (question, document.page_content)
        for document in documents
    ]

    # Score every pair
    scores = reranker.predict(pairs)

    # Combine documents with scores
    scored_documents = list(
        zip(documents, scores)
    )

    # Sort by score
    scored_documents.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Return top documents
    return scored_documents[:top_k]