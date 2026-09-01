"""
codelens/retriever.py

Dynamic retrieval pipeline backed by an IndexedRepository.

Architecture (unchanged from Day 4):

    Dense Retrieval (FAISS)
           +
        BM25
           ↓
    Reciprocal Rank Fusion
           ↓
      CrossEncoder rerank
           ↓
        Top-K docs

The reranker is loaded once and reused across calls.
All retrieval components operate against the currently
indexed repository — nothing is hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .config import RERANK_TOP_K
from .indexer import IndexedRepository

if TYPE_CHECKING:
    pass


# ============================================================
# RERANKER — loaded once, shared across requests
# ============================================================

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_reranker_instance: CrossEncoder | None = None


def _get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        print(f"[Retriever] Loading reranker: {_RERANKER_MODEL}")
        _reranker_instance = CrossEncoder(_RERANKER_MODEL)
    return _reranker_instance


# ============================================================
# RRF — reused from Day 4 (LangchainBasicsRag/hybrid.py logic)
# ============================================================

def _reciprocal_rank_fusion(
    dense_docs: list[Document],
    bm25_docs: list[Document],
    k: int = 60,
) -> list[Document]:
    """
    Merge dense and BM25 result lists using Reciprocal Rank Fusion.
    Uses metadata["file"] as the document identity key.
    """
    scores: dict[str, float] = {}
    doc_map: dict[str, Document] = {}

    for rank, doc in enumerate(dense_docs, start=1):
        doc_id = doc.metadata.get("file", doc.page_content[:64])
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    for rank, doc in enumerate(bm25_docs, start=1):
        doc_id = doc.metadata.get("file", doc.page_content[:64])
        doc_map[doc_id] = doc
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    ranked_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [doc_map[did] for did in ranked_ids]


# ============================================================
# RERANKING — reused from Day 4 (LangchainBasicsRag/reranker.py logic)
# ============================================================

def _rerank(
    question: str,
    documents: list[Document],
    top_k: int = RERANK_TOP_K,
) -> list[tuple[Document, float]]:
    """
    Score query-document pairs with CrossEncoder and return top-k
    as (document, score) tuples, sorted descending by score.
    """
    if not documents:
        return []

    reranker = _get_reranker()
    pairs = [(question, doc.page_content) for doc in documents]
    scores = reranker.predict(pairs)

    scored = sorted(zip(documents, scores), key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# ============================================================
# RETRIEVAL RESULT
# ============================================================

@dataclass
class RetrievalResult:
    """All stages of one retrieval pass, returned to the agent."""

    question: str
    dense_documents: list[Document] = field(default_factory=list)
    bm25_documents: list[Document] = field(default_factory=list)
    hybrid_documents: list[Document] = field(default_factory=list)
    reranked_pairs: list[tuple[Document, float]] = field(default_factory=list)
    final_documents: list[Document] = field(default_factory=list)


# ============================================================
# PUBLIC RETRIEVAL FUNCTION
# ============================================================

def retrieve(
    question: str,
    indexed_repo: IndexedRepository,
    top_k: int = RERANK_TOP_K,
) -> RetrievalResult:
    """
    Run the full Dense + BM25 → RRF → CrossEncoder pipeline.

    Args:
        question:     The user's question or refined query.
        indexed_repo: The currently active IndexedRepository.
        top_k:        Number of documents to return after reranking.

    Returns:
        RetrievalResult with all intermediate stages populated.
    """
    print(f"\n[Retriever] Query: {question!r}")

    # --------------------------------------------------------
    # Dense retrieval
    # --------------------------------------------------------
    dense_docs = indexed_repo.dense_retriever.invoke(question)
    print(f"[Retriever] Dense   : {len(dense_docs)} docs")

    # --------------------------------------------------------
    # BM25 retrieval
    # --------------------------------------------------------
    bm25_docs = indexed_repo.bm25_retriever.invoke(question)
    print(f"[Retriever] BM25    : {len(bm25_docs)} docs")

    # --------------------------------------------------------
    # Reciprocal Rank Fusion
    # --------------------------------------------------------
    hybrid_docs = _reciprocal_rank_fusion(dense_docs, bm25_docs)
    print(f"[Retriever] Hybrid  : {len(hybrid_docs)} docs")

    # --------------------------------------------------------
    # CrossEncoder reranking
    # --------------------------------------------------------
    reranked_pairs = _rerank(question, hybrid_docs, top_k=top_k)
    final_docs = [doc for doc, _ in reranked_pairs]
    print(f"[Retriever] Reranked: {len(final_docs)} docs")

    return RetrievalResult(
        question=question,
        dense_documents=dense_docs,
        bm25_documents=bm25_docs,
        hybrid_documents=hybrid_docs,
        reranked_pairs=reranked_pairs,
        final_documents=final_docs,
    )
