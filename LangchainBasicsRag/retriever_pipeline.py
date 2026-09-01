try:
    from hybrid import reciprocal_rank_fusion
    from reranker import rerank_documents
except ImportError:
    from LangchainBasicsRag.hybrid import reciprocal_rank_fusion
    from LangchainBasicsRag.reranker import rerank_documents


def retrieve_hybrid(
    question,
    dense_retriever,
    bm25_retriever,
    reranker,
    final_k=2
):

    # =========================================
    # 1. DENSE RETRIEVAL
    # =========================================

    dense_documents = dense_retriever.invoke(
        question
    )


    # =========================================
    # 2. BM25 RETRIEVAL
    # =========================================

    bm25_documents = bm25_retriever.invoke(
        question
    )


    # =========================================
    # 3. HYBRID RETRIEVAL
    # =========================================

    hybrid_documents = reciprocal_rank_fusion(
        dense_documents,
        bm25_documents
    )


    # =========================================
    # 4. RERANKING
    # =========================================

    reranked_results = rerank_documents(
        question,
        hybrid_documents,
        reranker,
        top_k=final_k
    )


    # =========================================
    # 5. EXTRACT FINAL DOCUMENTS
    # =========================================

    final_documents = [
        document
        for document, score in reranked_results
    ]


    # =========================================
    # 6. RETURN ALL RETRIEVAL STAGES
    # =========================================

    return {
        "dense": dense_documents,
        "bm25": bm25_documents,
        "hybrid": hybrid_documents,
        "reranked": reranked_results,
        "final_documents": final_documents
    }