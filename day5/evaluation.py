from dataset import evaluation_dataset
from metrics import recall_at_k


def evaluate_recall(
    retrieval_results,
    relevant_files,
    k=1
):
    # Dense
    dense_recall = recall_at_k(
        retrieval_results["dense"],
        relevant_files,
        k
    )

    # BM25
    bm25_recall = recall_at_k(
        retrieval_results["bm25"],
        relevant_files,
        k
    )

    # Hybrid
    hybrid_recall = recall_at_k(
        retrieval_results["hybrid"],
        relevant_files,
        k
    )

    # Reranked results contain:
    # (document, score)
    reranked_documents = [
        document
        for document, score
        in retrieval_results["reranked"]
    ]

    reranked_recall = recall_at_k(
        reranked_documents,
        relevant_files,
        k
    )

    return {
        "dense": dense_recall,
        "bm25": bm25_recall,
        "hybrid": hybrid_recall,
        "reranked": reranked_recall
    }