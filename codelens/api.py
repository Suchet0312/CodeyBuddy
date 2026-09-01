"""
codelens/api.py

FastAPI backend for CodeLens.

KEY DESIGN: indexing runs in a background thread.
Upload / GitHub / Local endpoints return IMMEDIATELY with
{"status": "indexing"} and the frontend polls GET /api/repository/status
until it becomes {"loaded": true}.

This prevents browser fetch timeouts for large repos.

Endpoints:
  POST /api/repository/upload      — upload a .zip archive
  POST /api/repository/local       — load a local folder by path
  POST /api/repository/github      — clone a GitHub URL
  GET  /api/repository/status      — current repo info + index status
  DELETE /api/repository           — clear active repository
  POST /api/ask                    — ask a question about the loaded repo
  GET  /api/health                 — liveness check
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .repository_manager import (
    RepositoryInfo,
    load_local_folder,
    load_zip_archive,
    load_github_url,
    cleanup_repository,
    is_valid_git_url,
)
from .indexer import IndexedRepository, index_repository
from .agent import run_investigation, InvestigationResult


# ============================================================
# THREAD POOL — indexing runs here, not in the event loop
# ============================================================

_executor = ThreadPoolExecutor(max_workers=2)


# ============================================================
# APPLICATION STATE
# ============================================================

class AppState:
    """Holds the single active repository and its index."""

    def __init__(self) -> None:
        self.repo_info: Optional[RepositoryInfo] = None
        self.indexed_repo: Optional[IndexedRepository] = None
        self.indexing: bool = False
        self.error: Optional[str] = None

    def is_ready(self) -> bool:
        return self.indexed_repo is not None and not self.indexing

    def clear(self) -> None:
        if self.repo_info is not None:
            cleanup_repository(self.repo_info)
        self.repo_info = None
        self.indexed_repo = None
        self.indexing = False
        self.error = None


_state = AppState()


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="CodeLens API",
    description="Agentic codebase intelligence platform",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# PYDANTIC MODELS
# ============================================================

class GitHubRequest(BaseModel):
    url: str

class LocalFolderRequest(BaseModel):
    path: str

class AskRequest(BaseModel):
    question: str

class StatusResponse(BaseModel):
    loaded: bool
    indexing: bool
    name: Optional[str] = None
    source: Optional[str] = None
    url: Optional[str] = None
    file_count: int = 0
    files: list[str] = []
    error: Optional[str] = None

class AskResponse(BaseModel):
    question: str
    answer: str
    evidence_files: list[str] = []
    tool_evidence: list[dict] = []
    investigation_history: list[str] = []
    retrieved_documents: list[dict] = []
    duration_seconds: float = 0.0


# ============================================================
# BACKGROUND INDEXING
# ============================================================

def _index_in_background(repo_info: RepositoryInfo) -> None:
    """
    Runs in a thread-pool thread.
    Updates _state on completion or failure.
    """
    try:
        indexed = index_repository(repo_info)
        _state.indexed_repo = indexed
        _state.repo_info = repo_info   # ensure always set after indexing
        _state.error = None
    except Exception as exc:
        _state.error = str(exc)
        traceback.print_exc()
    finally:
        _state.indexing = False


async def _start_indexing(repo_info: RepositoryInfo) -> None:
    """
    Store repo_info, mark indexing=True, then fire off background thread.
    Returns immediately — caller should return a 202 to the client.
    """
    _state.repo_info = repo_info
    _state.indexed_repo = None
    _state.indexing = True
    _state.error = None
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_executor, _index_in_background, repo_info)


def _require_ready() -> IndexedRepository:
    if _state.indexing:
        raise HTTPException(503, "Repository indexing is still in progress. Poll /api/repository/status.")
    if not _state.is_ready():
        raise HTTPException(400, "No repository loaded. Upload or provide a repository first.")
    return _state.indexed_repo  # type: ignore[return-value]


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():
    return {"status": "ok", "service": "CodeLens API"}


# ============================================================
# STATUS
# ============================================================

@app.get("/api/repository/status", response_model=StatusResponse)
def repository_status():
    if _state.repo_info is None and not _state.indexing:
        return StatusResponse(loaded=False, indexing=False, error=_state.error)

    # While cloning, repo_info may not be set yet
    name    = _state.repo_info.name        if _state.repo_info else "cloning…"
    source  = _state.repo_info.source      if _state.repo_info else "github"
    url     = _state.repo_info.url         if _state.repo_info else None
    fcount  = _state.repo_info.file_count  if _state.repo_info else 0
    files   = _state.repo_info.files       if _state.repo_info else []

    return StatusResponse(
        loaded=_state.is_ready(),
        indexing=_state.indexing,
        name=name,
        source=source,
        url=url,
        file_count=fcount,
        files=files,
        error=_state.error,
    )


# ============================================================
# UPLOAD ZIP
# ============================================================

@app.post("/api/repository/upload", status_code=202)
async def upload_repository(file: UploadFile = File(...)):
    """
    Accept a .zip archive.
    Extracts it, then kicks off background indexing.
    Returns 202 immediately — poll /api/repository/status.
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(400, "Only .zip archives are supported.")

    _state.clear()

    tmp_path = Path(tempfile.mktemp(suffix=".zip"))
    try:
        content = await file.read()
        tmp_path.write_bytes(content)
        repo_info = load_zip_archive(tmp_path)
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise HTTPException(500, f"Failed to read zip: {exc}")
    finally:
        tmp_path.unlink(missing_ok=True)

    # Override the temp-name with the original zip filename (strip .zip)
    original_name = Path(file.filename).stem
    repo_info.name = original_name

    await _start_indexing(repo_info)

    return {
        "status": "indexing",
        "message": f"Indexing '{repo_info.name}' in the background.",
        "name": repo_info.name,
        "file_count": repo_info.file_count,
    }


# ============================================================
# GITHUB
# ============================================================

@app.post("/api/repository/github", status_code=202)
async def load_github(request: GitHubRequest):
    """
    Clone a public GitHub repo, then kick off background indexing.
    Both clone and indexing run in the background thread.
    Returns 202 immediately — poll /api/repository/status.
    """
    url = request.url.strip()
    if not is_valid_git_url(url):
        raise HTTPException(400, f"Invalid GitHub URL: {url!r}")

    _state.clear()
    _state.indexing = True
    _state.error = None

    # Derive a display name from the URL immediately for the status response
    from .repository_manager import _slug_from_url
    display_name = _slug_from_url(url)

    async def _clone_then_index():
        loop = asyncio.get_event_loop()
        try:
            repo_info = await loop.run_in_executor(_executor, lambda: load_github_url(url))
            _state.repo_info = repo_info
            # Now index in the same background pool
            await loop.run_in_executor(_executor, _index_in_background, repo_info)
        except Exception as exc:
            _state.error = str(exc)
            _state.indexing = False
            traceback.print_exc()

    # Fire off without waiting
    asyncio.create_task(_clone_then_index())

    return {
        "status": "indexing",
        "message": f"Cloning and indexing '{display_name}' in the background.",
        "name": display_name,
        "file_count": 0,
        "url": url,
    }


# ============================================================
# LOCAL FOLDER
# ============================================================

@app.post("/api/repository/local", status_code=202)
async def load_local(request: LocalFolderRequest):
    """Load a local folder by absolute path, then index in background."""
    path = request.path.strip()
    if not path:
        raise HTTPException(400, "Folder path cannot be empty.")

    _state.clear()

    try:
        repo_info = load_local_folder(path)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, str(exc))

    await _start_indexing(repo_info)

    return {
        "status": "indexing",
        "message": f"Loading '{repo_info.name}', indexing in the background.",
        "name": repo_info.name,
        "file_count": repo_info.file_count,
    }


# ============================================================
# CLEAR
# ============================================================

@app.delete("/api/repository")
def clear_repository():
    _state.clear()
    return {"message": "Repository cleared."}


# ============================================================
# ASK
# ============================================================

@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Run a full CodeLens investigation. Requires a loaded, indexed repo."""
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Question cannot be empty.")

    indexed_repo = _require_ready()

    t0 = time.perf_counter()

    try:
        loop = asyncio.get_event_loop()
        result: InvestigationResult = await loop.run_in_executor(
            _executor,
            lambda: run_investigation(question=question, indexed_repo=indexed_repo),
        )
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(500, f"Investigation failed: {exc}")

    duration = round(time.perf_counter() - t0, 2)

    return AskResponse(
        question=result.question,
        answer=result.answer,
        evidence_files=result.evidence_files,
        tool_evidence=result.tool_evidence,
        investigation_history=result.investigation_history,
        retrieved_documents=result.retrieved_documents,
        duration_seconds=duration,
    )


# ============================================================
# SERVE FRONTEND
# ============================================================

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

if _FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend():
        index_path = _FRONTEND_DIR / "index.html"
        if index_path.exists():
            return FileResponse(str(index_path))
        return JSONResponse({"message": "Frontend not found."}, status_code=404)
