def recall_at_k(
    retrieved_documents,
    relevant_files,
    k
):
    """
    Calculate Recall@K.

    retrieved_documents:
        Ranked documents returned by retrieval.

    relevant_files:
        Ground-truth relevant file names.

    k:
        Number of top retrieved documents to consider.
    """

    # Take only top K results
    top_k_documents = retrieved_documents[:k]

    # Extract retrieved file names
    retrieved_files = set()

    for document in top_k_documents:

        file_path = document.metadata.get(
            "file"
        )

        # Convert path to just filename
        file_name = file_path.replace(
            "\\",
            "/"
        ).split("/")[-1]

        retrieved_files.add(
            file_name
        )

    # Ground-truth relevant files
    relevant_files = set(
        relevant_files
    )

    # Find correctly retrieved relevant files
    relevant_retrieved = (
        retrieved_files
        &
        relevant_files
    )

    # Recall formula:
    #
    # Relevant retrieved
    # ------------------
    # Total relevant

    recall = (
        len(relevant_retrieved)
        /
        len(relevant_files)
    )

    return recall

if __name__ == "__main__":

    from langchain_core.documents import Document


    retrieved_documents = [

        Document(
            page_content="User code",
            metadata={
                "file": "../target_repo/user.py"
            }
        ),

        Document(
            page_content="Authentication code",
            metadata={
                "file": "../target_repo/auth.py"
            }
        ),

        Document(
            page_content="Payment code",
            metadata={
                "file": "../target_repo/payment.py"
            }
        )
    ]


    relevant_files = [
        "auth.py"
    ]


    print(
        "Recall@1:",
        recall_at_k(
            retrieved_documents,
            relevant_files,
            k=1
        )
    )


    print(
        "Recall@2:",
        recall_at_k(
            retrieved_documents,
            relevant_files,
            k=2
        )
    )