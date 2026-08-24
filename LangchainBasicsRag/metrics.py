def get_file_name(document):
    """
    Extract normalized filename from a document.
    """

    file_path = document.metadata.get(
        "file"
    )

    file_name = (
        file_path
        .replace("\\", "/")
        .split("/")[-1]
    )

    return file_name


# =========================================
# RECALL@K
# =========================================

def recall_at_k(
    retrieved_documents,
    relevant_files,
    k
):
    """
    Recall@K

    Relevant documents retrieved in top K
    ------------------------------------
    Total relevant documents
    """

    top_k_documents = retrieved_documents[:k]

    retrieved_files = {
        get_file_name(document)
        for document in top_k_documents
    }

    relevant_files = set(
        relevant_files
    )

    relevant_retrieved = (
        retrieved_files
        &
        relevant_files
    )

    if len(relevant_files) == 0:
        return 0.0

    return (
        len(relevant_retrieved)
        /
        len(relevant_files)
    )


# =========================================
# PRECISION@K
# =========================================

def precision_at_k(
    retrieved_documents,
    relevant_files,
    k
):
    """
    Precision@K

    Relevant documents retrieved in top K
    ------------------------------------
    Total documents retrieved in top K
    """

    top_k_documents = retrieved_documents[:k]

    retrieved_files = {
        get_file_name(document)
        for document in top_k_documents
    }

    relevant_files = set(
        relevant_files
    )

    relevant_retrieved = (
        retrieved_files
        &
        relevant_files
    )

    if len(top_k_documents) == 0:
        return 0.0

    return (
        len(relevant_retrieved)
        /
        len(top_k_documents)
    )


# =========================================
# HIT RATE@K
# =========================================

def hit_rate_at_k(
    retrieved_documents,
    relevant_files,
    k
):
    """
    Hit Rate@K

    Returns:
    1.0 -> At least one relevant document
           was found in top K

    0.0 -> No relevant document was found
           in top K
    """

    top_k_documents = retrieved_documents[:k]

    retrieved_files = {
        get_file_name(document)
        for document in top_k_documents
    }

    relevant_files = set(
        relevant_files
    )

    relevant_retrieved = (
        retrieved_files
        &
        relevant_files
    )

    if len(relevant_retrieved) > 0:
        return 1.0

    return 0.0


# =========================================
# RECIPROCAL RANK
# =========================================

def reciprocal_rank(
    retrieved_documents,
    relevant_files
):
    """
    Reciprocal Rank

    Finds the first relevant document.

    Rank 1 -> 1 / 1 = 1.0
    Rank 2 -> 1 / 2 = 0.5
    Rank 3 -> 1 / 3 = 0.333

    No relevant document -> 0.0
    """

    relevant_files = set(
        relevant_files
    )

    for rank, document in enumerate(
        retrieved_documents,
        start=1
    ):

        file_name = get_file_name(
            document
        )

        if file_name in relevant_files:

            return 1 / rank

    return 0.0