from loader import load_repo
from splitter import split_documents
from embedder import get_embeddings
from vector_store import create_vector_store
from generator import format_documents, create_generator

from bm25 import create_bm25_retriever
from hybrid import reciprocal_rank_fusion
from reranker import get_reranker, rerank_documents
from retriever_pipeline import retrieve_hybrid

from dataset import evaluation_dataset
from evaluation import evaluate_all_strategies


# =========================================
# 1. LOAD TARGET REPOSITORY
# =========================================

documents = load_repo(
    "../target_repo"
)

print("Total documents:", len(documents))


# =========================================
# 2. SPLIT DOCUMENTS
# =========================================

chunks = split_documents(
    documents
)

print("Total chunks:", len(chunks))


# =========================================
# 3. LOAD EMBEDDING MODEL
# =========================================

embeddings = get_embeddings()


# =========================================
# 4. CREATE FAISS VECTOR STORE
# =========================================

vector_store = create_vector_store(
    chunks,
    embeddings
)


# =========================================
# 5. CREATE DENSE RETRIEVER
# =========================================

dense_retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# =========================================
# 6. CREATE BM25 RETRIEVER
# =========================================

bm25_retriever = create_bm25_retriever(
    chunks,
    k=3
)


# =========================================
# 7. CREATE RERANKER
# =========================================

reranker = get_reranker()


# =========================================
# 8. TEST QUESTIONS
# =========================================

questions = [
    "Where is authentication implemented?",
    "Explain what happens when a user logs in.",
    "Where is payment processing implemented?"
]


# =========================================
# 9. RETRIEVE AND INSPECT RESULTS
# =========================================

for question in questions:

    print("\n" + "=" * 60)
    print("QUESTION:", question)
    print("=" * 60)


    # -----------------------------------------
    # DENSE RETRIEVAL
    # -----------------------------------------

    dense_documents = dense_retriever.invoke(
        question
    )

    print("\nDENSE RETRIEVAL RESULTS:")

    for rank, document in enumerate(
        dense_documents,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            "File:",
            document.metadata.get("file")
        )

        print("Content:")
        print(document.page_content)


    # -----------------------------------------
    # BM25 RETRIEVAL
    # -----------------------------------------

    bm25_documents = bm25_retriever.invoke(
        question
    )

    print("\n" + "-" * 40)
    print("BM25 RETRIEVAL RESULTS:")
    print("-" * 40)

    for rank, document in enumerate(
        bm25_documents,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            "File:",
            document.metadata.get("file")
        )

        print("Content:")
        print(document.page_content)


    # -----------------------------------------
    # HYBRID RETRIEVAL
    # -----------------------------------------

    hybrid_documents = reciprocal_rank_fusion(
        dense_documents,
        bm25_documents
    )

    print("\n" + "-" * 40)
    print("HYBRID RETRIEVAL RESULTS:")
    print("-" * 40)

    for rank, document in enumerate(
        hybrid_documents,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            "File:",
            document.metadata.get("file")
        )

        print("Content:")
        print(document.page_content)


    # -----------------------------------------
    # RERANKING
    # -----------------------------------------

    reranked_results = rerank_documents(
        question,
        hybrid_documents,
        reranker,
        top_k=3
    )

    print("\n" + "-" * 40)
    print("RERANKED RESULTS:")
    print("-" * 40)

    for rank, (
        document,
        score
    ) in enumerate(
        reranked_results,
        start=1
    ):

        print(f"\nRank {rank}")

        print(
            "Score:",
            float(score)
        )

        print(
            "File:",
            document.metadata.get("file")
        )

        print("Content:")
        print(document.page_content)


# =========================================
# GENERATION USING HYBRID + RERANKING
# =========================================


# =========================================
# 10. SELECT QUESTION FOR GENERATION
# =========================================

generation_question = (
    "Explain what happens when a user logs in."
)


# =========================================
# 11. RUN COMPLETE RETRIEVAL PIPELINE
# =========================================

retrieval_results = retrieve_hybrid(
    question=generation_question,
    dense_retriever=dense_retriever,
    bm25_retriever=bm25_retriever,
    reranker=reranker,
    final_k=2
)


# =========================================
# 12. EXTRACT FINAL DOCUMENTS
# =========================================

final_documents = retrieval_results[
    "final_documents"
]


# =========================================
# 13. PRINT FINAL RETRIEVED DOCUMENTS
# =========================================

print("\n" + "=" * 60)
print("FINAL RETRIEVAL FOR GENERATION")
print("=" * 60)

for rank, document in enumerate(
    final_documents,
    start=1
):

    print(f"\nRank {rank}")

    print(
        "File:",
        document.metadata.get("file")
    )

    print("Content:")
    print(document.page_content)


# =========================================
# 14. BUILD CONTEXT
# =========================================

context = format_documents(
    final_documents
)


# =========================================
# 15. CREATE PROMPT AND LLM
# =========================================

prompt, llm = create_generator()


# =========================================
# 16. CREATE FINAL PROMPT
# =========================================

messages = prompt.invoke({
    "context": context,
    "question": generation_question
})


# =========================================
# 17. GENERATE ANSWER
# =========================================

response = llm.invoke(
    messages
)


# =========================================
# 18. PRINT FINAL ANSWER
# =========================================

print("\n" + "=" * 60)
print("FINAL GENERATION - HYBRID + RERANKING")
print("=" * 60)

print("\nQUESTION:")
print(generation_question)

print("\nANSWER:")
print(response.content)


# =========================================
# DAY 5 - COMPLETE RETRIEVAL EVALUATION
# =========================================

print("\n" + "=" * 60)
print("DAY 5 - COMPLETE RETRIEVAL EVALUATION")
print("=" * 60)


# =========================================
# EVALUATION K
# =========================================

k = 3


# =========================================
# STORE TOTAL METRICS
# =========================================

total_scores = {

    "dense": {
        "recall": 0,
        "precision": 0,
        "hit_rate": 0,
        "reciprocal_rank": 0
    },

    "bm25": {
        "recall": 0,
        "precision": 0,
        "hit_rate": 0,
        "reciprocal_rank": 0
    },

    "hybrid": {
        "recall": 0,
        "precision": 0,
        "hit_rate": 0,
        "reciprocal_rank": 0
    },

    "reranked": {
        "recall": 0,
        "precision": 0,
        "hit_rate": 0,
        "reciprocal_rank": 0
    }
}


# =========================================
# EVALUATE EVERY BENCHMARK QUESTION
# =========================================

for item in evaluation_dataset:

    question = item["question"]

    relevant_files = item[
        "relevant_files"
    ]


    print("\n" + "-" * 60)
    print("QUESTION:", question)
    print("RELEVANT FILES:", relevant_files)
    print("-" * 60)


    # -----------------------------------------
    # RUN COMPLETE RETRIEVAL PIPELINE
    # -----------------------------------------

    retrieval_results = retrieve_hybrid(
        question=question,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        final_k=3
    )


    # -----------------------------------------
    # EVALUATE ALL STRATEGIES
    # -----------------------------------------

    scores = evaluate_all_strategies(
        retrieval_results,
        relevant_files,
        k
    )


    # =========================================
    # PRINT RESULTS FOR THIS QUESTION
    # =========================================

    for strategy, metrics in scores.items():

        print("\n" + strategy.upper())

        print(
            f"Recall@{k}:",
            round(
                metrics["recall"],
                3
            )
        )

        print(
            f"Precision@{k}:",
            round(
                metrics["precision"],
                3
            )
        )

        print(
            f"Hit Rate@{k}:",
            round(
                metrics["hit_rate"],
                3
            )
        )

        print(
            "Reciprocal Rank:",
            round(
                metrics["reciprocal_rank"],
                3
            )
        )


    # =========================================
    # ADD RESULTS TO TOTALS
    # =========================================

    for strategy in total_scores:

        total_scores[strategy][
            "recall"
        ] += scores[strategy][
            "recall"
        ]

        total_scores[strategy][
            "precision"
        ] += scores[strategy][
            "precision"
        ]

        total_scores[strategy][
            "hit_rate"
        ] += scores[strategy][
            "hit_rate"
        ]

        total_scores[strategy][
            "reciprocal_rank"
        ] += scores[strategy][
            "reciprocal_rank"
        ]


# =========================================
# CALCULATE FINAL AVERAGES
# =========================================

num_questions = len(
    evaluation_dataset
)


# =========================================
# FINAL DAY 5 EVALUATION REPORT
# =========================================

print("\n" + "=" * 60)
print("FINAL DAY 5 EVALUATION REPORT")
print("=" * 60)


for strategy, metrics in total_scores.items():

    # Average Recall
    average_recall = (
        metrics["recall"]
        / num_questions
    )


    # Average Precision
    average_precision = (
        metrics["precision"]
        / num_questions
    )


    # Average Hit Rate
    average_hit_rate = (
        metrics["hit_rate"]
        / num_questions
    )


    # Mean Reciprocal Rank
    mrr = (
        metrics["reciprocal_rank"]
        / num_questions
    )


    print("\n" + "-" * 40)

    print(
        "STRATEGY:",
        strategy.upper()
    )

    print("-" * 40)


    print(
        f"Average Recall@{k}:",
        round(
            average_recall,
            3
        )
    )


    print(
        f"Average Precision@{k}:",
        round(
            average_precision,
            3
        )
    )


    print(
        f"Average Hit Rate@{k}:",
        round(
            average_hit_rate,
            3
        )
    )


    print(
        "MRR:",
        round(
            mrr,
            3
        )
    )