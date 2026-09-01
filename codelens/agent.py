"""
codelens/agent.py

The CodeLens LangGraph investigation agent.

Graph topology:

    START
      ↓
    agent_node          ← LLM decides: tool call or go to retrieval
      ↓
    ┌─────────────────────────────┐
    │  tool_calls?                │
    │  YES → tools (ToolNode)     │
    │       → collect_tool_evidence│
    │       → back to agent_node  │
    │  NO  → retrieval_node       │
    └─────────────────────────────┘
      ↓
    evidence_check
      ↓
    ┌────────────────────────────────┐
    │  sufficient?                   │
    │  YES → generation_node → END   │
    │  NO  → refine_query            │
    │       → retrieval_node         │
    │       → evidence_check (loop)  │
    │  too many retries → generation │
    └────────────────────────────────┘

The agent is stateless between calls — a fresh graph run is
created for every question against the currently loaded
IndexedRepository.
"""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing import Annotated

from .config import LLM_MODEL, RERANK_TOP_K
from .retriever import retrieve, RetrievalResult
from .tools import build_tools

if TYPE_CHECKING:
    from .indexer import IndexedRepository


# ============================================================
# CONSTANTS
# ============================================================

MAX_RETRIES = 2


# ============================================================
# STATE
# ============================================================

class CodeLensState(dict):
    """
    LangGraph state for one investigation run.

    Using a plain TypedDict-style dict so LangGraph can handle
    the `add_messages` reducer on the messages key.
    """


from typing import TypedDict

class CodeLensState(TypedDict, total=False):
    # The user's original question (never modified)
    original_question: str

    # Possibly refined query used for retrieval
    search_query: str

    # LangChain message history (manages tool call ↔ tool result pairing)
    messages: Annotated[list, add_messages]

    # Retrieval stages
    dense_documents: list
    bm25_documents: list
    hybrid_documents: list
    reranked_documents: list
    documents: list              # final reranked docs used for generation

    # Tool evidence
    tool_results: list[str]
    tool_evidence: list[dict]

    # Evidence gate
    evidence_sufficient: bool
    retry_count: int

    # Final answer
    answer: str

    # Human-readable investigation trace (appended to, never replaced)
    investigation_history: Annotated[list, lambda a, b: a + b]


# ============================================================
# AGENT BUILDER
# ============================================================

def build_agent(indexed_repo: "IndexedRepository"):
    """
    Compile and return a LangGraph graph for *indexed_repo*.

    Returns a callable: graph.invoke(initial_state) → final_state
    """

    # --------------------------------------------------------
    # Tools and LLM
    # --------------------------------------------------------
    tools = build_tools(indexed_repo)
    tool_node = ToolNode(tools)

    llm = ChatOllama(model=LLM_MODEL)
    llm_with_tools = llm.bind_tools(tools)

    repo_name = indexed_repo.repo_info.name

    # --------------------------------------------------------
    # NODE: agent
    # --------------------------------------------------------
    def agent_node(state: CodeLensState) -> dict:
        print("\n[Agent] Agent node")

        if not state.get("messages"):
            system = SystemMessage(content=f"""You are CodeLens, an expert software investigation agent.

You are analysing the repository: {repo_name}

Your job is to investigate the user's question thoroughly using the
available tools and retrieved evidence.

Available tools:
  • list_repo_files        — list all files (supports optional glob pattern)
  • read_repository_file   — read a specific file by relative path
  • search_repository      — search for a function, class, variable, or text

When to use tools vs retrieval:
  • "What files are in this repo?" → list_repo_files
  • "Show me auth.py" → read_repository_file
  • "Where is login() used?" → search_repository
  • "How does authentication work?" → start with retrieval; use tools if you need specific files

Rules:
  • Do not invent repository facts.
  • If you call a tool, wait for the result before drawing conclusions.
  • If the question requires understanding multiple files, use search_repository first.
  • After receiving tool results, reflect on the evidence before deciding the next step.
""")
            messages = [system, HumanMessage(content=state["original_question"])]
        else:
            messages = state["messages"]

        response = llm_with_tools.invoke(messages)

        tool_calls = getattr(response, "tool_calls", [])
        if tool_calls:
            tool_names = [tc["name"] for tc in tool_calls]
            print(f"[Agent] Tool calls requested: {tool_names}")
            history = [f"Agent requested tool(s): {', '.join(tool_names)}"]
        else:
            print("[Agent] No tool calls — routing to retrieval")
            history = ["Agent completed tool loop, routing to retrieval."]

        return {
            "messages": [response],
            "investigation_history": history,
        }

    # --------------------------------------------------------
    # NODE: collect_tool_evidence
    # --------------------------------------------------------
    def collect_tool_evidence(state: CodeLensState) -> dict:
        print("[Agent] Collecting tool evidence")

        messages = state.get("messages", [])
        tool_results = list(state.get("tool_results", []))
        tool_evidence = list(state.get("tool_evidence", []))

        for msg in messages:
            if not isinstance(msg, ToolMessage):
                continue
            content = str(msg.content)
            tool_name = getattr(msg, "name", "unknown_tool")
            tool_results.append(content)
            tool_evidence.append({"tool": tool_name, "content": content})
            print(f"[Agent] Tool evidence captured: {tool_name}")

        return {
            "tool_results": tool_results,
            "tool_evidence": tool_evidence,
            "investigation_history": [
                f"Tool results captured ({len(tool_evidence)} total)."
            ],
        }

    # --------------------------------------------------------
    # NODE: retrieval
    # --------------------------------------------------------
    def retrieval_node(state: CodeLensState) -> dict:
        question = state.get("search_query") or state["original_question"]
        print(f"[Agent] Retrieval node — query: {question!r}")

        result: RetrievalResult = retrieve(question, indexed_repo)

        return {
            "dense_documents": result.dense_documents,
            "bm25_documents": result.bm25_documents,
            "hybrid_documents": result.hybrid_documents,
            "reranked_documents": result.final_documents,
            "documents": result.final_documents,
            "investigation_history": [
                f"Retrieval: dense={len(result.dense_documents)}, "
                f"bm25={len(result.bm25_documents)}, "
                f"hybrid={len(result.hybrid_documents)}, "
                f"reranked={len(result.final_documents)} docs."
            ],
        }

    # --------------------------------------------------------
    # NODE: evidence_check
    # --------------------------------------------------------
    def evidence_check(state: CodeLensState) -> dict:
        docs = state.get("documents", [])
        tool_evidence = state.get("tool_evidence", [])
        total = len(docs) + len(tool_evidence)
        sufficient = total > 0

        print(f"[Agent] Evidence check — docs={len(docs)}, "
              f"tool_evidence={len(tool_evidence)}, sufficient={sufficient}")

        return {
            "evidence_sufficient": sufficient,
            "investigation_history": [
                f"Evidence check: {total} piece(s) — "
                f"{'sufficient' if sufficient else 'insufficient'}."
            ],
        }

    # --------------------------------------------------------
    # NODE: refine_query
    # --------------------------------------------------------
    def refine_query(state: CodeLensState) -> dict:
        original = state["original_question"]
        retry = state.get("retry_count", 0) + 1
        refined = original + " implementation details function class"
        print(f"[Agent] Query refinement (retry {retry}): {refined!r}")
        return {
            "search_query": refined,
            "retry_count": retry,
            "investigation_history": [
                f"Query refined (attempt {retry}): {refined!r}"
            ],
        }

    # --------------------------------------------------------
    # NODE: generation
    # --------------------------------------------------------
    def generation_node(state: CodeLensState) -> dict:
        print("[Agent] Generation node")

        question = state["original_question"]
        docs: list[Document] = state.get("documents", [])
        tool_evidence: list[dict] = state.get("tool_evidence", [])

        # Build context from retrieved docs
        retrieval_context = ""
        for doc in docs:
            file_label = doc.metadata.get("file", "unknown")
            retrieval_context += f"\n--- FILE: {file_label} ---\n{doc.page_content}\n"

        # Build context from tool results
        tool_context = ""
        for ev in tool_evidence:
            tool_context += f"\n--- TOOL: {ev['tool']} ---\n{ev['content']}\n"

        full_context = retrieval_context + tool_context

        if not full_context.strip():
            answer = (
                "I was unable to find sufficient evidence in the repository "
                f"to answer your question: {question!r}"
            )
            return {
                "answer": answer,
                "investigation_history": ["Generation: no evidence available."],
            }

        # Collect evidence file names for citation
        evidence_files = list(
            dict.fromkeys(
                doc.metadata.get("file", "unknown") for doc in docs
            )
        )

        generation_prompt = f"""You are CodeLens, an expert software investigation assistant.

Answer the user's question based ONLY on the repository evidence provided below.

Rules:
- Cite specific files when referencing code.
- Distinguish between a function's DEFINITION and its USAGE/CALLS.
- If a function is defined but you cannot find explicit calls, say so explicitly.
- Never invent repository behaviour.
- Be concise but complete.

REPOSITORY EVIDENCE:
{full_context}

QUESTION:
{question}

Provide a clear, evidence-grounded answer:"""

        llm_plain = ChatOllama(model=LLM_MODEL)
        response = llm_plain.invoke([HumanMessage(content=generation_prompt)])
        answer = response.content

        print(f"[Agent] Answer generated ({len(answer)} chars)")

        return {
            "answer": answer,
            "investigation_history": [
                f"Answer generated using {len(docs)} retrieved doc(s) "
                f"and {len(tool_evidence)} tool result(s)."
            ],
        }

    # --------------------------------------------------------
    # ROUTING: after agent node
    # --------------------------------------------------------
    def decide_after_agent(state: CodeLensState) -> str:
        last = state["messages"][-1] if state.get("messages") else None
        if last and getattr(last, "tool_calls", None):
            return "tools"
        return "retrieve"

    # --------------------------------------------------------
    # ROUTING: after evidence check
    # --------------------------------------------------------
    def decide_after_evidence(state: CodeLensState) -> str:
        if state.get("evidence_sufficient", False):
            return "generate"
        retry = state.get("retry_count", 0)
        if retry >= MAX_RETRIES:
            print("[Agent] Max retries reached — generating with available evidence")
            return "generate"
        return "refine"

    # --------------------------------------------------------
    # BUILD GRAPH
    # --------------------------------------------------------
    builder = StateGraph(CodeLensState)

    builder.add_node("agent",                agent_node)
    builder.add_node("tools",                tool_node)
    builder.add_node("collect_tool_evidence", collect_tool_evidence)
    builder.add_node("retrieve",             retrieval_node)
    builder.add_node("evidence_check",       evidence_check)
    builder.add_node("refine_query",         refine_query)
    builder.add_node("generate",             generation_node)

    builder.add_edge(START,                   "agent")
    builder.add_conditional_edges(
        "agent",
        decide_after_agent,
        {"tools": "tools", "retrieve": "retrieve"},
    )
    builder.add_edge("tools",                "collect_tool_evidence")
    builder.add_edge("collect_tool_evidence", "agent")
    builder.add_edge("retrieve",             "evidence_check")
    builder.add_conditional_edges(
        "evidence_check",
        decide_after_evidence,
        {"generate": "generate", "refine": "refine_query"},
    )
    builder.add_edge("refine_query",         "retrieve")
    builder.add_edge("generate",             END)

    return builder.compile()


# ============================================================
# INVESTIGATION RESULT
# ============================================================

@dataclass
class InvestigationResult:
    """Structured result returned to the API layer."""
    question: str
    answer: str
    evidence_files: list[str] = field(default_factory=list)
    tool_evidence: list[dict] = field(default_factory=list)
    investigation_history: list[str] = field(default_factory=list)
    retrieved_documents: list[dict] = field(default_factory=list)


# ============================================================
# PUBLIC ENTRY POINT
# ============================================================

def run_investigation(
    question: str,
    indexed_repo: "IndexedRepository",
) -> InvestigationResult:
    """
    Run a full CodeLens investigation for *question* against *indexed_repo*.

    This is the single function the API layer calls.
    """
    print(f"\n{'='*60}")
    print(f"CODELENS INVESTIGATION")
    print(f"Repository : {indexed_repo.repo_info.name}")
    print(f"Question   : {question}")
    print(f"{'='*60}")

    graph = build_agent(indexed_repo)

    initial_state: CodeLensState = {
        "original_question": question,
        "search_query": question,
        "messages": [],
        "dense_documents": [],
        "bm25_documents": [],
        "hybrid_documents": [],
        "reranked_documents": [],
        "documents": [],
        "tool_results": [],
        "tool_evidence": [],
        "evidence_sufficient": False,
        "retry_count": 0,
        "answer": "",
        "investigation_history": [f"Investigation started for: {question!r}"],
    }

    final_state = graph.invoke(initial_state)

    # Collect evidence file names (deduplicated, order-preserving)
    docs: list[Document] = final_state.get("documents", [])
    evidence_files = list(
        dict.fromkeys(d.metadata.get("file", "unknown") for d in docs)
    )

    # Serialise retrieved documents for the frontend
    retrieved_docs_serialised = [
        {
            "file": d.metadata.get("file", "unknown"),
            "content": d.page_content[:500],   # truncate for API response
        }
        for d in docs
    ]

    return InvestigationResult(
        question=question,
        answer=final_state.get("answer", "No answer generated."),
        evidence_files=evidence_files,
        tool_evidence=final_state.get("tool_evidence", []),
        investigation_history=final_state.get("investigation_history", []),
        retrieved_documents=retrieved_docs_serialised,
    )
