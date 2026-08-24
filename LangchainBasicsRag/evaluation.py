from metrics import (
    recall_at_k,
    precision_at_k,
    hit_rate_at_k,
    reciprocal_rank
)


def extract_reranked_documents(
    reranked_results
):
    """
    Convert:

    [(document, score), ...]

    into:

    [document, ...]
    """

    return [
        document
        for document, score in reranked_results
    ]


def evaluate_strategy(
    retrieved_documents,
    relevant_files,
    k
):
    """
    Evaluate one retrieval strategy.
    """

    return {
        "recall": recall_at_k(
            retrieved_documents,
            relevant_files,
            k
        ),

        "precision": precision_at_k(
            retrieved_documents,
            relevant_files,
            k
        ),

        "hit_rate": hit_rate_at_k(
            retrieved_documents,
            relevant_files,
            k
        ),

        "reciprocal_rank": reciprocal_rank(
            retrieved_documents,
            relevant_files
        )
    }


def evaluate_all_strategies(
    retrieval_results,
    relevant_files,
    k
):
    """
    Evaluate:

    Dense
    BM25
    Hybrid
    Reranked
    """

    reranked_documents = (
        extract_reranked_documents(
            retrieval_results["reranked"]
        )
    )

    return {
        "dense": evaluate_strategy(
            retrieval_results["dense"],
            relevant_files,
            k
        ),

        "bm25": evaluate_strategy(
            retrieval_results["bm25"],
            relevant_files,
            k
        ),

        "hybrid": evaluate_strategy(
            retrieval_results["hybrid"],
            relevant_files,
            k
        ),

        "reranked": evaluate_strategy(
            reranked_documents,
            relevant_files,
            k
        )
    }