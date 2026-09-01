"""
codelens/indexer.py

Reusable indexing pipeline.

  Repository path
       ↓
  File discovery (config-filtered)
       ↓
  Document creation (LangChain Documents)
       ↓
  Code-aware chunking (RecursiveCharacterTextSplitter)
       ↓
  Embeddings  (HuggingFace BGE)
       ↓
  FAISS vector store
       ↓
  BM25 index
       ↓
  IndexedRepository  ← returned to caller

The result is held in memory for the lifetime of the session.
Re-indexing is only triggered when a new repository is loaded.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever

from .config import (
    IGNORE_DIRS,
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL,
    DENSE_K,
    BM25_K,
)
from .repository_manager import RepositoryInfo


# ============================================================
# LANGUAGE DETECTION
# ============================================================

# Map file extensions to LangChain Language enum for smart splitting.
# Falls back to plain text splitting when extension is not in the map.
_EXT_TO_LANGUAGE: dict[str, Language] = {
    ".py":   Language.PYTHON,
    ".js":   Language.JS,
    ".jsx":  Language.JS,
    ".mjs":  Language.JS,
    ".cjs":  Language.JS,
    ".ts":   Language.TS,
    ".tsx":  Language.TS,
    ".java": Language.JAVA,
    ".go":   Language.GO,
    ".rs":   Language.RUST,
    ".rb":   Language.RUBY,
    ".cpp":  Language.CPP,
    ".cxx":  Language.CPP,
    ".cc":   Language.CPP,
    ".c":    Language.C,
    ".h":    Language.C,
    ".hpp":  Language.CPP,
    ".cs":   Language.CSHARP,
    ".md":   Language.MARKDOWN,
    ".html": Language.HTML,
    ".htm":  Language.HTML,
    ".scala": Language.SCALA,
    ".swift": Language.SWIFT,
    ".kt":   Language.KOTLIN,
    ".php":  Language.PHP,
    ".rb":   Language.RUBY,
    ".ex":   Language.ELIXIR,
    ".exs":  Language.ELIXIR,
}

# ============================================================
# INDEXED REPOSITORY
# ============================================================

@dataclass
class IndexedRepository:
    """
    Holds all retrieval artefacts for one indexed repository.
    Passed to the retriever and tools layer.
    """
    repo_info: RepositoryInfo
    documents: list[Document]          # raw documents (one per file)
    chunks: list[Document]             # after splitting
    vector_store: FAISS                # dense index
    dense_retriever: object            # FAISS retriever
    bm25_retriever: BM25Retriever      # sparse index


# ============================================================
# DOCUMENT LOADING
# ============================================================

def _load_documents(repo_path: Path, files: list[str]) -> list[Document]:
    """
    Read each discovered file and wrap it in a LangChain Document.
    Metadata carries: file (relative path), abs_file (absolute path),
    extension, and language (if detected).
    """
    documents: list[Document] = []

    for rel_path in files:
        abs_path = repo_path / rel_path

        # Guard: extension check (discover_files already filters, but be safe)
        suffix = abs_path.suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            continue

        # Size guard
        try:
            size = abs_path.stat().st_size
            if size > MAX_FILE_SIZE_BYTES:
                continue
        except OSError:
            continue

        # Read
        try:
            content = abs_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        if not content.strip():
            continue

        language = _EXT_TO_LANGUAGE.get(suffix, "text")

        doc = Document(
            page_content=content,
            metadata={
                "file": rel_path,                   # relative — used as doc ID in RRF
                "abs_file": str(abs_path),          # absolute — used by tools
                "extension": suffix,
                "language": language,
            },
        )
        documents.append(doc)

    return documents


# ============================================================
# SPLITTING
# ============================================================

def _split_documents(documents: list[Document]) -> list[Document]:
    """
    Split documents using language-aware RecursiveCharacterTextSplitter.
    Documents with unrecognised extensions fall back to generic splitting.
    """
    # Group by language so we use the right splitter per group
    groups: dict[str | Language, list[Document]] = {}
    for doc in documents:
        lang = doc.metadata.get("language", "text")
        groups.setdefault(lang, []).append(doc)

    all_chunks: list[Document] = []

    for lang, docs in groups.items():
        if isinstance(lang, Language):
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=lang,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
            )

        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)

    return all_chunks


# ============================================================
# INDEX PIPELINE
# ============================================================

def index_repository(repo_info: RepositoryInfo) -> IndexedRepository:
    """
    Run the full indexing pipeline for *repo_info*.

    Steps:
      1. Load documents from discovered files
      2. Split into chunks
      3. Build FAISS vector store
      4. Build BM25 index
      5. Return IndexedRepository

    This is the single entry point called by the API layer.
    """
    print(f"[Indexer] Loading documents from: {repo_info.path}")

    # --------------------------------------------------------
    # 1. Load
    # --------------------------------------------------------
    documents = _load_documents(repo_info.path, repo_info.files)
    print(f"[Indexer] Documents loaded : {len(documents)}")

    if not documents:
        raise ValueError(
            f"No indexable documents found in repository: {repo_info.name}"
        )

    # --------------------------------------------------------
    # 2. Split
    # --------------------------------------------------------
    chunks = _split_documents(documents)
    print(f"[Indexer] Chunks created   : {len(chunks)}")

    # --------------------------------------------------------
    # 3. Embeddings + FAISS
    # --------------------------------------------------------
    print(f"[Indexer] Loading embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("[Indexer] Building FAISS vector store...")
    vector_store = FAISS.from_documents(documents=chunks, embedding=embeddings)

    dense_retriever = vector_store.as_retriever(
        search_kwargs={"k": DENSE_K}
    )

    # --------------------------------------------------------
    # 4. BM25
    # --------------------------------------------------------
    print("[Indexer] Building BM25 index...")
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = BM25_K

    print(f"[Indexer] Indexing complete for: {repo_info.name}")

    return IndexedRepository(
        repo_info=repo_info,
        documents=documents,
        chunks=chunks,
        vector_store=vector_store,
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
    )
