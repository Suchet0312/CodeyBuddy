import sys
from pathlib import Path
from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END
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
# IMPORT CODELENS COMPONENTS
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
# TARGET REPOSITORY
# =========================================

TARGET_REPO = (
    PROJECT_ROOT
    / "target_repo"
)


# =========================================
# DEFINE STATE
# =========================================

class CodeLensState(TypedDict):

    question: str
    documents: list
    route: str
    answer: str


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
# CREATE EMBEDDINGS
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
# CREATE RETRIEVER
# =========================================

dense_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 2
    }
)


# =========================================
# CREATE GENERATOR
# =========================================

print("Loading generator...")

prompt, llm = create_generator()


# =========================================
# RETRIEVE NODE
# =========================================

def retrieve_node(state):

    print("\n" + "=" * 60)
    print("RETRIEVE NODE")
    print("=" * 60)

    question = state["question"]

    retrieved_documents = (
        dense_retriever.invoke(
            question
        )
    )

    print(
        "Retrieved:",
        len(retrieved_documents),
        "documents"
    )

    return {
        "documents": retrieved_documents
    }


# =========================================
# ROUTER NODE
# =========================================

def route_node(state):

    print("\n" + "=" * 60)
    print("ROUTER NODE")
    print("=" * 60)

    question = state["question"].lower()

    investigation_keywords = [
        "explain",
        "how",
        "flow",
        "trace",
        "relationship",
        "interact"
    ]

    if any(
        keyword in question
        for keyword in investigation_keywords
    ):

        route = "investigate"

    else:

        route = "direct"

    print(
        "Selected route:",
        route
    )

    return {
        "route": route
    }


# =========================================
# ROUTING FUNCTION
# =========================================

def decide_route(state):

    return state["route"]


# =========================================
# INVESTIGATE NODE
# =========================================

def investigate_node(state):

    print("\n" + "=" * 60)
    print("INVESTIGATION NODE")
    print("=" * 60)

    documents = state["documents"]

    print(
        "Investigating",
        len(documents),
        "retrieved documents"
    )

    for document in documents:

        print(
            "File:",
            document.metadata.get(
                "file"
            )
        )

    return {}


# =========================================
# GENERATE NODE
# =========================================

def generate_node(state):

    print("\n" + "=" * 60)
    print("GENERATE NODE")
    print("=" * 60)

    question = state["question"]

    documents = state["documents"]

    context = format_documents(
        documents
    )

    messages = prompt.invoke({
        "context": context,
        "question": question
    })

    response = llm.invoke(
        messages
    )

    return {
        "answer": response.content
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
    "retrieve",
    retrieve_node
)

graph_builder.add_node(
    "route",
    route_node
)

graph_builder.add_node(
    "investigate",
    investigate_node
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
    "retrieve"
)

graph_builder.add_edge(
    "retrieve",
    "route"
)


# =========================================
# CONDITIONAL ROUTING
# =========================================

graph_builder.add_conditional_edges(
    "route",
    decide_route,
    {
        "direct": "generate",
        "investigate": "investigate"
    }
)


# =========================================
# INVESTIGATION PATH
# =========================================

graph_builder.add_edge(
    "investigate",
    "generate"
)


# =========================================
# END
# =========================================

graph_builder.add_edge(
    "generate",
    END
)


# =========================================
# COMPILE GRAPH
# =========================================

graph = graph_builder.compile()


# =========================================
# RUN GRAPH
# =========================================

question = (
    "Explain what happens when a user logs in."
)


initial_state = {
    "question": question,
    "documents": [],
    "route": "",
    "answer": ""
}


result = graph.invoke(
    initial_state
)


# =========================================
# FINAL RESULT
# =========================================

print("\n" + "=" * 60)
print("FINAL CODELENS RESULT")
print("=" * 60)

print(
    "\nQUESTION:"
)

print(
    result["question"]
)

print(
    "\nROUTE:"
)

print(
    result["route"]
)

print(
    "\nANSWER:"
)

print(
    result["answer"]
)