import sys
from pathlib import Path
from typing import TypedDict, Annotated

from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
)

from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# ============================================================
# PROJECT PATH SETUP
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


# ============================================================
# IMPORT CODELENS TOOLS
# ============================================================

from tools import CODELENS_TOOLS


# ============================================================
# IMPORT RETRIEVAL COMPONENTS
# ============================================================

from LangchainBasicsRag.loader import load_repo
from LangchainBasicsRag.splitter import split_documents
from LangchainBasicsRag.embedder import get_embeddings
from LangchainBasicsRag.vector_store import create_vector_store
from LangchainBasicsRag.generator import create_generator


# ============================================================
# IMPORT DAY 4 COMPONENTS
# ============================================================

from day4.bm25 import create_bm25_retriever

from day4.hybrid import reciprocal_rank_fusion

from day4.reranker import (
    get_reranker,
    rerank_documents,
)


# ============================================================
# REPOSITORY
# ============================================================

TARGET_REPO = PROJECT_ROOT / "target_repo"


# ============================================================
# INITIALIZE RETRIEVAL PIPELINE
# ============================================================

print("=" * 70)
print("INITIALIZING CODELENS DAY 7")
print("=" * 70)


# ------------------------------------------------------------
# Load repository
# ------------------------------------------------------------

documents = load_repo(
    str(TARGET_REPO)
)


# ------------------------------------------------------------
# Split documents
# ------------------------------------------------------------

chunks = split_documents(
    documents
)


print(
    f"Documents loaded : {len(documents)}"
)

print(
    f"Chunks created   : {len(chunks)}"
)


# ------------------------------------------------------------
# Embeddings
# ------------------------------------------------------------

embeddings = get_embeddings()


# ------------------------------------------------------------
# Vector store
# ------------------------------------------------------------

vector_store = create_vector_store(
    chunks,
    embeddings
)


# ------------------------------------------------------------
# Dense retriever
# ------------------------------------------------------------

dense_retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# ------------------------------------------------------------
# BM25 retriever
# ------------------------------------------------------------

bm25_retriever = create_bm25_retriever(
    chunks,
    k=3
)


# ------------------------------------------------------------
# CrossEncoder reranker
# ------------------------------------------------------------

reranker = get_reranker()


# ============================================================
# LLM
# ============================================================

prompt, llm = create_generator()


# ============================================================
# BIND TOOLS
# ============================================================

llm_with_tools = llm.bind_tools(
    CODELENS_TOOLS
)


# ============================================================
# STATE
# ============================================================

class CodeLensState(TypedDict, total=False):

    # LangGraph conversation history
    messages: Annotated[
        list,
        add_messages
    ]

    # Original user question
    original_question: str

    # Current retrieval query
    search_query: str

    # Retrieval results
    dense_documents: list
    hybrid_documents: list
    reranked_documents: list

    # Final evidence collection
    documents: list

    # Tool evidence
    tool_results: list
    tool_evidence: list

    # Final answer
    answer: str

    # Evidence decision
    evidence_sufficient: bool

    # Retry counter
    retry_count: int

    # Investigation trace
    investigation_history: list


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(question: str):

    print("\n" + "=" * 70)
    print("RETRIEVAL PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------

    dense_documents = dense_retriever.invoke(
        question
    )

    print(
        f"Dense results: {len(dense_documents)}"
    )


    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------

    bm25_documents = bm25_retriever.invoke(
        question
    )

    print(
        f"BM25 results: {len(bm25_documents)}"
    )


    # --------------------------------------------------------
    # Reciprocal Rank Fusion
    # --------------------------------------------------------

    hybrid_documents = reciprocal_rank_fusion(
        dense_documents,
        bm25_documents
    )

    print(
        f"Hybrid results: {len(hybrid_documents)}"
    )


    # --------------------------------------------------------
    # CrossEncoder reranking
    # --------------------------------------------------------

    reranked_results = rerank_documents(
        question,
        hybrid_documents,
        reranker
    )


    # --------------------------------------------------------
    # Extract documents from:
    #
    # (document, score)
    #
    # pairs returned by reranker
    # --------------------------------------------------------

    reranked_documents = [
        document
        for document, score in reranked_results
    ]


    print(
        f"Reranked results: {len(reranked_documents)}"
    )


    return (
        dense_documents,
        hybrid_documents,
        reranked_documents
    )


# ============================================================
# AGENT NODE
# ============================================================

def agent_node(
    state: CodeLensState
):

    print("\n" + "=" * 70)
    print("AGENT NODE")
    print("=" * 70)


    # --------------------------------------------------------
    # First invocation
    # --------------------------------------------------------

    if not state.get("messages"):

        system_message = SystemMessage(
            content="""
You are CodeLens, an agent for investigating
software repositories.

Your job is to determine how to investigate
the user's question.

Available repository tools:

1. list_repo_files
   Use this when you need to understand
   repository structure.

2. read_repository_file
   Use this when you need the exact contents
   of a known file.

3. search_repository
   Use this when you need to find a function,
   class, variable, symbol, or text.

Use tools when direct repository inspection
is appropriate.

Do not invent repository facts.

If the question requires repository-wide
semantic understanding, retrieval can be used.

If a specific file or symbol needs inspection,
use the appropriate repository tool.

After receiving tool results, reason about
the evidence and continue the investigation.
"""
        )

        messages = [
            system_message,
            HumanMessage(
                content=state["original_question"]
            )
        ]

    else:

        # ----------------------------------------------------
        # Continue existing agent conversation
        # ----------------------------------------------------

        messages = state["messages"]


    # --------------------------------------------------------
    # Invoke LLM
    # --------------------------------------------------------

    response = llm_with_tools.invoke(
        messages
    )


    # --------------------------------------------------------
    # Debug output
    # --------------------------------------------------------

    print(
        "Tool calls:"
    )

    print(
        getattr(
            response,
            "tool_calls",
            []
        )
    )


    # --------------------------------------------------------
    # Return response
    # --------------------------------------------------------

    return {
        "messages": [
            response
        ],

        "investigation_history": [
            "Agent evaluated the repository investigation requirements."
        ]
    }


# ============================================================
# TOOL NODE
# ============================================================

tool_node = ToolNode(
    CODELENS_TOOLS
)


# ============================================================
# COLLECT TOOL EVIDENCE
# ============================================================

def collect_tool_evidence(
    state: CodeLensState
):

    print("\n" + "=" * 70)
    print("COLLECTING TOOL EVIDENCE")
    print("=" * 70)


    messages = state.get(
        "messages",
        []
    )


    tool_evidence = list(
        state.get(
            "tool_evidence",
            []
        )
    )


    tool_results = list(
        state.get(
            "tool_results",
            []
        )
    )


    # --------------------------------------------------------
    # Inspect messages for ToolMessage
    # --------------------------------------------------------

    for message in messages:

        if message.__class__.__name__ != "ToolMessage":
            continue


        content = str(
            message.content
        )


        tool_name = getattr(
            message,
            "name",
            "unknown_tool"
        )


        tool_results.append(
            content
        )


        tool_evidence.append(
            {
                "tool": tool_name,
                "content": content,
            }
        )


        print(
            f"Tool executed: {tool_name}"
        )


    return {

        "tool_results": tool_results,

        "tool_evidence": tool_evidence,

        "investigation_history": [
            "Repository tool executed and result stored as evidence."
        ]
    }


# ============================================================
# AGENT ROUTER
# ============================================================

def decide_after_agent(
    state: CodeLensState
):

    last_message = (
        state["messages"][-1]
    )


    # --------------------------------------------------------
    # LLM requested tool
    # --------------------------------------------------------

    if getattr(
        last_message,
        "tool_calls",
        None
    ):

        print(
            "ROUTING → TOOLS"
        )

        return "tools"


    # --------------------------------------------------------
    # No tool call
    # --------------------------------------------------------

    print(
        "ROUTING → RETRIEVAL"
    )

    return "retrieve"


# ============================================================
# RETRIEVAL NODE
# ============================================================

def retrieval_node(
    state: CodeLensState
):

    question = state.get(
        "search_query",
        state["original_question"]
    )


    (
        dense_documents,
        hybrid_documents,
        reranked_documents
    ) = retrieve_documents(
        question
    )


    return {

        "dense_documents": dense_documents,

        "hybrid_documents": hybrid_documents,

        "reranked_documents": reranked_documents,

        "documents": reranked_documents,

        "investigation_history": [
            f"Retrieved and reranked repository evidence for: {question}"
        ]
    }


# ============================================================
# EVIDENCE CHECK
# ============================================================

def evidence_check(
    state: CodeLensState
):

    documents = state.get(
        "documents",
        []
    )


    tool_evidence = state.get(
        "tool_evidence",
        []
    )


    total_evidence = (
        len(documents)
        +
        len(tool_evidence)
    )


    sufficient = (
        total_evidence > 0
    )


    print("\n" + "=" * 70)
    print("EVIDENCE CHECK")
    print("=" * 70)


    print(
        f"Retrieved documents : {len(documents)}"
    )


    print(
        f"Tool evidence       : {len(tool_evidence)}"
    )


    print(
        f"Total evidence      : {total_evidence}"
    )


    print(
        f"Sufficient          : {sufficient}"
    )


    return {
        "evidence_sufficient": sufficient
    }


# ============================================================
# EVIDENCE ROUTER
# ============================================================

def decide_after_evidence(
    state: CodeLensState
):

    if state.get(
        "evidence_sufficient",
        False
    ):

        return "generate"


    retry_count = state.get(
        "retry_count",
        0
    )


    if retry_count >= 1:

        return "generate"


    return "refine"


# ============================================================
# QUERY REFINEMENT
# ============================================================

def refine_query(
    state: CodeLensState
):

    original_question = state[
        "original_question"
    ]


    old_query = state.get(
        "search_query",
        original_question
    )


    refined_query = (
        old_query
        + " implementation function class"
    )


    retry_count = (
        state.get(
            "retry_count",
            0
        )
        + 1
    )


    print("\n" + "=" * 70)
    print("QUERY REFINEMENT")
    print("=" * 70)


    print(
        f"New query: {refined_query}"
    )


    return {

        "search_query": refined_query,

        "retry_count": retry_count,

        "investigation_history": [
            f"Refined retrieval query: {refined_query}"
        ]
    }


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    state: CodeLensState
):

    print("\n" + "=" * 70)
    print("FINAL GENERATION")
    print("=" * 70)


    question = state[
        "original_question"
    ]


    documents = state.get(
        "documents",
        []
    )


    tool_evidence = state.get(
        "tool_evidence",
        []
    )


    context_parts = []


    # ========================================================
    # RETRIEVAL EVIDENCE
    # ========================================================

    for document in documents:

        metadata = getattr(
            document,
            "metadata",
            {}
        )


        source = (
            metadata.get("file")
            or metadata.get("source")
            or "unknown"
        )


        content = getattr(
            document,
            "page_content",
            str(document)
        )


        context_parts.append(
            f"""
[RETRIEVED EVIDENCE]
File: {source}

{content}
"""
        )


    # ========================================================
    # TOOL EVIDENCE
    # ========================================================

    for evidence in tool_evidence:

        context_parts.append(
            f"""
[TOOL EVIDENCE]
Tool: {evidence["tool"]}

{evidence["content"]}
"""
        )


    # ========================================================
    # COMBINE EVIDENCE
    # ========================================================

    context = "\n".join(
        context_parts
    )


    # ========================================================
    # GENERATION PROMPT
    # ========================================================

    generation_prompt = f"""
You are CodeLens, a repository investigation assistant.

Answer the user's question using ONLY the repository
evidence provided below.

USER QUESTION:
{question}

REPOSITORY EVIDENCE:
{context}

Rules:

1. Do not invent repository facts.
2. Use the provided evidence.
3. Mention relevant file names.
4. Explain the reasoning clearly.
5. If tool evidence was used, rely on it explicitly.
6. If the evidence is insufficient, say so.
"""


    response = llm.invoke(
        generation_prompt
    )


    answer = (
        response.content
        if hasattr(response, "content")
        else str(response)
    )


    return {

        "answer": answer,

        "investigation_history": [
            "Generated final answer using retrieval and tool evidence."
        ]
    }


# ============================================================
# BUILD GRAPH
# ============================================================

graph_builder = StateGraph(
    CodeLensState
)


# ============================================================
# ADD NODES
# ============================================================

graph_builder.add_node(
    "agent",
    agent_node
)


graph_builder.add_node(
    "tools",
    tool_node
)


graph_builder.add_node(
    "collect_tool_evidence",
    collect_tool_evidence
)


graph_builder.add_node(
    "retrieve",
    retrieval_node
)


graph_builder.add_node(
    "evidence_check",
    evidence_check
)


graph_builder.add_node(
    "refine",
    refine_query
)


graph_builder.add_node(
    "generate",
    generate_answer
)


# ============================================================
# START → AGENT
# ============================================================

graph_builder.add_edge(
    START,
    "agent"
)


# ============================================================
# AGENT → TOOL OR RETRIEVAL
# ============================================================

graph_builder.add_conditional_edges(
    "agent",
    decide_after_agent,
    {
        "tools": "tools",
        "retrieve": "retrieve",
    }
)


# ============================================================
# TOOL → COLLECT EVIDENCE
# ============================================================

graph_builder.add_edge(
    "tools",
    "collect_tool_evidence"
)


# ============================================================
# TOOL EVIDENCE → RETRIEVAL
# ============================================================

graph_builder.add_edge(
    "collect_tool_evidence",
    "retrieve"
)


# ============================================================
# RETRIEVAL → EVIDENCE CHECK
# ============================================================

graph_builder.add_edge(
    "retrieve",
    "evidence_check"
)


# ============================================================
# EVIDENCE CHECK → GENERATE / REFINE
# ============================================================

graph_builder.add_conditional_edges(
    "evidence_check",
    decide_after_evidence,
    {
        "generate": "generate",
        "refine": "refine",
    }
)


# ============================================================
# REFINE → RETRIEVAL
# ============================================================

graph_builder.add_edge(
    "refine",
    "retrieve"
)


# ============================================================
# GENERATE → END
# ============================================================

graph_builder.add_edge(
    "generate",
    END
)


# ============================================================
# COMPILE
# ============================================================

graph = graph_builder.compile()


# ============================================================
# RUN CODELENS
# ============================================================

def run_codelens(
    question: str
):

    initial_state = {

        "messages": [],

        "original_question": question,

        "search_query": question,

        "dense_documents": [],

        "hybrid_documents": [],

        "reranked_documents": [],

        "documents": [],

        "tool_results": [],

        "tool_evidence": [],

        "answer": "",

        "evidence_sufficient": False,

        "retry_count": 0,

        "investigation_history": [],
    }


    result = graph.invoke(
        initial_state
    )


    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print()

    question = input(
        "Ask CodeLens a question: "
    )


    result = run_codelens(
        question
    )


    print("\n" + "=" * 70)
    print("CODELENS ANSWER")
    print("=" * 70)


    print(
        result.get(
            "answer",
            "No answer generated."
        )
    )


    print("\n" + "=" * 70)
    print("TOOL EVIDENCE")
    print("=" * 70)


    for evidence in result.get(
        "tool_evidence",
        []
    ):

        print(
            f"\nTool: {evidence['tool']}"
        )

        print(
            evidence["content"]
        )


    print("\n" + "=" * 70)
    print("INVESTIGATION HISTORY")
    print("=" * 70)


    for event in result.get(
        "investigation_history",
        []
    ):

        print(
            "-",
            event
        )