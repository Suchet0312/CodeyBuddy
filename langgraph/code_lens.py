import sys
from pathlib import Path
from typing import TypedDict, Annotated
from operator import add

from langgraph.graph import (
    StateGraph,
    START,
    END
)

from langgraph.checkpoint.memory import (
    MemorySaver
)


# =========================================
# PROJECT PATH SETUP
# =========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)


# =========================================
# IMPORT CODELENS RAG COMPONENTS
# =========================================

from LangchainBasicsRag.loader import load_repo
from LangchainBasicsRag.splitter import split_documents
from LangchainBasicsRag.embedder import get_embeddings
from LangchainBasicsRag.vector_store import create_vector_store
from LangchainBasicsRag.generator import (
    format_documents,
    create_generator
)


# =========================================
# IMPORT RETRIEVAL COMPONENTS
# =========================================

from LangchainBasicsRag.bm25 import (
    create_bm25_retriever
)

from LangchainBasicsRag.hybrid import (
    reciprocal_rank_fusion
)

from LangchainBasicsRag.reranker import (
    get_reranker,
    rerank_documents
)


# =========================================
# TARGET REPOSITORY
# =========================================

TARGET_REPO = (
    PROJECT_ROOT
    / "target_repo"
)


# =========================================
# DEFINE CODELENS STATE
# =========================================

class CodeLensState(TypedDict):

    # Original user question.
    # Never modified.
    original_question: str

    # Query used for retrieval.
    # Can be refined by the agent.
    search_query: str

    # Selected graph route.
    route: str

    # Retrieval results.
    dense_documents: list
    hybrid_documents: list
    reranked_documents: list

    # Final evidence documents.
    documents: list

    # Final generated answer.
    answer: str

    # Evidence evaluation result.
    evidence_sufficient: bool

    # Number of retrieval retries.
    retry_count: int

    # Execution trace.
    #
    # add means:
    #
    # existing_history + new_history
    #
    # instead of replacing the list.
    investigation_history: Annotated[
        list[str],
        add
    ]


# =========================================
# LOAD REPOSITORY
# =========================================

print("Loading repository...")

documents = load_repo(
    str(TARGET_REPO)
)

chunks = split_documents(
    documents
)

print(
    "Total chunks:",
    len(chunks)
)


# =========================================
# LOAD EMBEDDINGS
# =========================================

print("Loading embeddings...")

embeddings = get_embeddings()


# =========================================
# CREATE VECTOR STORE
# =========================================

print("Creating vector store...")

vector_store = create_vector_store(
    chunks,
    embeddings
)


# =========================================
# CREATE RETRIEVERS
# =========================================

dense_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)

bm25_retriever = create_bm25_retriever(
    chunks,
    k=3
)


# =========================================
# CREATE RERANKER
# =========================================

print("Loading reranker...")

reranker = get_reranker()


# =========================================
# CREATE GENERATOR
# =========================================

print("Loading generator...")

prompt, llm = create_generator()


# =========================================
# LLM ROUTER NODE
# =========================================

def route_question_node(state):

    print("\n" + "=" * 60)
    print("LLM ROUTER NODE")
    print("=" * 60)

    question = (
        state["original_question"]
    )

    router_prompt = f"""
You are a routing agent for CodeLens,
an AI system that investigates code repositories.

Classify the user's question into exactly one category.

DIRECT:
Use when the question can likely be answered by
finding one specific implementation, file, function,
class, or module.

INVESTIGATE:
Use when the question requires understanding
relationships, behavior, execution flow, interactions,
or multiple parts of the codebase.

USER QUESTION:
{question}

Return only one word:

DIRECT

or

INVESTIGATE
"""

    response = llm.invoke(
        router_prompt
    )

    decision = (
        response.content
        .strip()
        .upper()
    )

    # Defensive handling.
    #
    # Even if the LLM returns extra text,
    # try to safely map it.

    if "INVESTIGATE" in decision:

        route = "investigate"

    else:

        route = "direct"

    print(
        "LLM router decision:",
        decision
    )

    print(
        "Selected route:",
        route
    )

    return {
        "route": route,

        "investigation_history": [
            f"LLM router selected route: {route}"
        ]
    }


# =========================================
# ROUTING FUNCTION
# =========================================

def decide_route(state):

    return state["route"]


# =========================================
# DENSE RETRIEVAL NODE
# =========================================

def dense_retrieval_node(state):

    print("\n" + "=" * 60)
    print("DENSE RETRIEVAL NODE")
    print("=" * 60)

    search_query = (
        state["search_query"]
    )

    print(
        "Search query:",
        search_query
    )

    dense_documents = (
        dense_retriever.invoke(
            search_query
        )
    )

    print(
        "Dense documents:",
        len(dense_documents)
    )

    return {
        "dense_documents": dense_documents,

        "documents": dense_documents,

        "investigation_history": [
            "Dense retrieval completed. "
            f"Retrieved {len(dense_documents)} documents."
        ]
    }


# =========================================
# HYBRID RETRIEVAL NODE
# =========================================

def hybrid_retrieval_node(state):

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVAL NODE")
    print("=" * 60)

    search_query = (
        state["search_query"]
    )

    print(
        "Search query:",
        search_query
    )

    # Dense retrieval

    dense_documents = (
        dense_retriever.invoke(
            search_query
        )
    )

    # BM25 retrieval

    bm25_documents = (
        bm25_retriever.invoke(
            search_query
        )
    )

    # Reciprocal Rank Fusion

    hybrid_documents = (
        reciprocal_rank_fusion(
            dense_documents,
            bm25_documents
        )
    )

    print(
        "Dense results:",
        len(dense_documents)
    )

    print(
        "BM25 results:",
        len(bm25_documents)
    )

    print(
        "Hybrid results:",
        len(hybrid_documents)
    )

    return {
        "dense_documents": dense_documents,

        "hybrid_documents": hybrid_documents,

        "investigation_history": [
            "Hybrid retrieval completed. "
            f"Dense: {len(dense_documents)}, "
            f"BM25: {len(bm25_documents)}, "
            f"Hybrid: {len(hybrid_documents)}."
        ]
    }


# =========================================
# RERANK NODE
# =========================================

def rerank_node(state):

    print("\n" + "=" * 60)
    print("RERANK NODE")
    print("=" * 60)

    search_query = (
        state["search_query"]
    )

    hybrid_documents = (
        state["hybrid_documents"]
    )

    reranked_results = (
        rerank_documents(
            search_query,
            hybrid_documents,
            reranker,
            top_k=2
        )
    )

    reranked_documents = [
        document
        for document, score
        in reranked_results
    ]

    print(
        "Reranked documents:",
        len(reranked_documents)
    )

    return {
        "reranked_documents": reranked_documents,

        "documents": reranked_documents,

        "investigation_history": [
            "Reranking completed. "
            f"Selected top "
            f"{len(reranked_documents)} documents."
        ]
    }


# =========================================
# CHECK EVIDENCE NODE
# =========================================

def check_evidence_node(state):

    print("\n" + "=" * 60)
    print("CHECK EVIDENCE NODE")
    print("=" * 60)

    original_question = (
        state["original_question"]
    )

    documents = (
        state["documents"]
    )

    context = format_documents(
        documents
    )

    evaluation_prompt = f"""
You are an evidence evaluator for CodeLens.

Determine whether the retrieved code contains enough
information to answer the user's original question.

ORIGINAL QUESTION:
{original_question}

RETRIEVED CODE:
{context}

Return only:

YES

if sufficient evidence exists.

Return only:

NO

if sufficient evidence does not exist.
"""

    response = llm.invoke(
        evaluation_prompt
    )

    decision = (
        response.content
        .strip()
        .upper()
    )

    evidence_sufficient = (
        "YES" in decision
    )

    print(
        "LLM decision:",
        decision
    )

    print(
        "Evidence sufficient:",
        evidence_sufficient
    )

    if evidence_sufficient:

        history_message = (
            "Evidence check passed. "
            "Retrieved code is sufficient."
        )

    else:

        history_message = (
            "Evidence check failed. "
            "Retrieved code is insufficient."
        )

    return {
        "evidence_sufficient":
            evidence_sufficient,

        "investigation_history": [
            history_message
        ]
    }


# =========================================
# EVIDENCE ROUTING FUNCTION
# =========================================

def decide_after_evidence_check(state):

    if state["evidence_sufficient"]:

        return "generate"

    if state["retry_count"] < 1:

        return "refine"

    return "insufficient_evidence"


# =========================================
# REFINE QUERY NODE
# =========================================

def refine_query_node(state):

    print("\n" + "=" * 60)
    print("REFINE QUERY NODE")
    print("=" * 60)

    original_question = (
        state["original_question"]
    )

    current_search_query = (
        state["search_query"]
    )

    retry_count = (
        state["retry_count"]
    )

    refinement_prompt = f"""
You are a query refinement agent for CodeLens.

The current retrieval query did not retrieve enough
evidence to answer the user's original question.

Create a better query for searching source code.

ORIGINAL QUESTION:
{original_question}

CURRENT QUERY:
{current_search_query}

The improved query should:

- preserve the original intent
- use technical terms
- include likely code concepts
- include likely function or module terms
- be optimized for code retrieval

Return only the improved search query.
"""

    response = llm.invoke(
        refinement_prompt
    )

    refined_query = (
        response.content
        .strip()
    )

    new_retry_count = (
        retry_count + 1
    )

    print(
        "Current query:",
        current_search_query
    )

    print(
        "Refined query:",
        refined_query
    )

    print(
        "Retry count:",
        new_retry_count
    )

    return {
        "search_query": refined_query,

        "retry_count":
            new_retry_count,

        "investigation_history": [
            "Query refined by LLM. "
            f"New query: {refined_query}"
        ]
    }


# =========================================
# INSUFFICIENT EVIDENCE NODE
# =========================================

def insufficient_evidence_node(state):

    print("\n" + "=" * 60)
    print("INSUFFICIENT EVIDENCE NODE")
    print("=" * 60)

    original_question = (
        state["original_question"]
    )

    answer = (
        "I could not find sufficient evidence in the "
        "available code to answer the question: "
        f"'{original_question}'."
    )

    return {
        "answer": answer,

        "investigation_history": [
            "Investigation ended. "
            "Insufficient evidence after retry."
        ]
    }


# =========================================
# GENERATE NODE
# =========================================

def generate_node(state):

    print("\n" + "=" * 60)
    print("GENERATE NODE")
    print("=" * 60)

    original_question = (
        state["original_question"]
    )

    documents = (
        state["documents"]
    )

    context = format_documents(
        documents
    )

    messages = prompt.invoke({
        "context": context,
        "question": original_question
    })

    response = llm.invoke(
        messages
    )

    return {
        "answer": response.content,

        "investigation_history": [
            "Final answer generated successfully."
        ]
    }


# =========================================
# CREATE GRAPH
# =========================================

graph_builder = StateGraph(
    CodeLensState
)


# =========================================
# ADD NODES
# =========================================

graph_builder.add_node(
    "route",
    route_question_node
)

graph_builder.add_node(
    "dense_retrieve",
    dense_retrieval_node
)

graph_builder.add_node(
    "hybrid_retrieve",
    hybrid_retrieval_node
)

graph_builder.add_node(
    "rerank",
    rerank_node
)

graph_builder.add_node(
    "check_evidence",
    check_evidence_node
)

graph_builder.add_node(
    "refine_query",
    refine_query_node
)

graph_builder.add_node(
    "insufficient_evidence",
    insufficient_evidence_node
)

graph_builder.add_node(
    "generate",
    generate_node
)


# =========================================
# ADD EDGES
# =========================================

graph_builder.add_edge(
    START,
    "route"
)


# -----------------------------------------
# ROUTE SELECTION
# -----------------------------------------

graph_builder.add_conditional_edges(
    "route",
    decide_route,
    {
        "direct":
            "dense_retrieve",

        "investigate":
            "hybrid_retrieve"
    }
)


# -----------------------------------------
# DIRECT PATH
# -----------------------------------------

graph_builder.add_edge(
    "dense_retrieve",
    "check_evidence"
)


# -----------------------------------------
# INVESTIGATION PATH
# -----------------------------------------

graph_builder.add_edge(
    "hybrid_retrieve",
    "rerank"
)

graph_builder.add_edge(
    "rerank",
    "check_evidence"
)


# -----------------------------------------
# EVIDENCE DECISION
# -----------------------------------------

graph_builder.add_conditional_edges(
    "check_evidence",
    decide_after_evidence_check,
    {
        "generate":
            "generate",

        "refine":
            "refine_query",

        "insufficient_evidence":
            "insufficient_evidence"
    }
)


# -----------------------------------------
# RETRY LOOP
# -----------------------------------------

graph_builder.add_edge(
    "refine_query",
    "dense_retrieve"
)


# -----------------------------------------
# END
# -----------------------------------------

graph_builder.add_edge(
    "generate",
    END
)

graph_builder.add_edge(
    "insufficient_evidence",
    END
)


# =========================================
# CHECKPOINTING
# =========================================

memory = MemorySaver()


# =========================================
# COMPILE GRAPH
# =========================================

graph = graph_builder.compile(
    checkpointer=memory
)


# =========================================
# VISUALIZE GRAPH
# =========================================

graph_image = (
    graph.get_graph()
    .draw_mermaid_png()
)

with open(
    "codelens_graph.png",
    "wb"
) as file:

    file.write(
        graph_image
    )

print(
    "\nGraph visualization saved as:"
)

print(
    "codelens_graph.png"
)


# =========================================
# RUN GRAPH
# =========================================

question = (
    "Explain what happens when a user logs in."
)


initial_state = {
    "original_question": question,

    "search_query": question,

    "route": "",

    "dense_documents": [],

    "hybrid_documents": [],

    "reranked_documents": [],

    "documents": [],

    "answer": "",

    "evidence_sufficient": False,

    "retry_count": 0,

    "investigation_history": []
}


# =========================================
# CONFIGURATION
# =========================================

config = {
    "configurable": {
        "thread_id":
            "codelens-session-1"
    }
}


# =========================================
# EXECUTE GRAPH
# =========================================

result = graph.invoke(
    initial_state,
    config=config
)


# =========================================
# FINAL RESULT
# =========================================

print("\n" + "=" * 60)
print("FINAL CODELENS RESULT")
print("=" * 60)

print(
    "\nORIGINAL QUESTION:"
)

print(
    result["original_question"]
)

print(
    "\nFINAL SEARCH QUERY:"
)

print(
    result["search_query"]
)

print(
    "\nROUTE:"
)

print(
    result["route"]
)

print(
    "\nEVIDENCE SUFFICIENT:"
)

print(
    result["evidence_sufficient"]
)

print(
    "\nRETRY COUNT:"
)

print(
    result["retry_count"]
)

print(
    "\nANSWER:"
)

print(
    result["answer"]
)


# =========================================
# INVESTIGATION HISTORY
# =========================================

print("\n" + "=" * 60)
print("INVESTIGATION HISTORY")
print("=" * 60)

for step, history_item in enumerate(
    result["investigation_history"],
    start=1
):

    print(
        f"{step}. {history_item}"
    )


# =========================================
# CHECKPOINT STATE
# =========================================

print("\n" + "=" * 60)
print("CHECKPOINTED STATE")
print("=" * 60)

checkpoint_state = graph.get_state(
    config
)

print(
    "Checkpoint values available:",
    checkpoint_state.values is not None
)

print(
    "Checkpoint next node:",
    checkpoint_state.next
)