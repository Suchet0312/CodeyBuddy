"""
codelens/repository_manager.py

Handles two repository input modes:
  1. Local folder / uploaded zip archive
  2. GitHub (or any public Git) URL

Provides a unified RepositoryInfo dataclass that the rest
of CodeLens uses to refer to the active repository.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .config import IGNORE_DIRS, ALLOWED_EXTENSIONS, MAX_FILE_SIZE_BYTES, REPOS_WORKSPACE


# ============================================================
# DATA MODEL
# ============================================================

@dataclass
class RepositoryInfo:
    """Describes a repository that has been loaded into the workspace."""

    # Unique session identifier for this repo
    repo_id: str

    # Human-readable name (repo folder name or URL slug)
    name: str

    # Absolute path to the root of the repository on disk
    path: Path

    # How the repo was supplied
    source: str          # "local" | "github" | "upload"

    # Original URL (only set for github source)
    url: Optional[str] = None

    # Whether the directory was cloned/extracted into a temp workspace
    # (needs cleanup when done)
    is_temporary: bool = False

    # Populated after file discovery
    file_count: int = 0
    files: list[str] = field(default_factory=list)


# ============================================================
# GITHUB URL VALIDATION
# ============================================================

_GITHUB_URL_RE = re.compile(
    r"^https?://(www\.)?github\.com/[\w.\-]+/[\w.\-]+(\.git)?(/.*)?$",
    re.IGNORECASE,
)

_GIT_URL_RE = re.compile(
    r"^https?://[\w.\-]+(:\d+)?/.*\.git$",
    re.IGNORECASE,
)


def is_valid_git_url(url: str) -> bool:
    """Return True if *url* looks like a clonable public Git URL."""
    url = url.strip()
    return bool(_GITHUB_URL_RE.match(url) or _GIT_URL_RE.match(url))


def _normalise_github_url(url: str) -> str:
    """Strip trailing slashes and ensure the URL ends without .git for display."""
    url = url.strip().rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]
    return url


def _slug_from_url(url: str) -> str:
    """Extract 'owner-repo' style slug from a GitHub URL."""
    url = _normalise_github_url(url)
    parts = url.rstrip("/").split("/")
    if len(parts) >= 2:
        return f"{parts[-2]}-{parts[-1]}"
    return parts[-1]


# ============================================================
# WORKSPACE SETUP
# ============================================================

def _ensure_workspace() -> Path:
    REPOS_WORKSPACE.mkdir(parents=True, exist_ok=True)
    return REPOS_WORKSPACE


# ============================================================
# FILE DISCOVERY
# ============================================================

def discover_files(repo_path: Path) -> list[str]:
    """
    Walk *repo_path* and return relative paths of all indexable files.

    Skips:
      - IGNORE_DIRS directories
      - Files whose extension is not in ALLOWED_EXTENSIONS
      - Files larger than MAX_FILE_SIZE_BYTES
      - Binary files (detected by trying UTF-8 decode of first 512 bytes)
    """
    indexable: list[str] = []

    for item in repo_path.rglob("*"):
        # Skip directories themselves
        if item.is_dir():
            continue

        # Skip files inside ignored directories
        if any(part in IGNORE_DIRS for part in item.parts):
            continue

        # Extension filter
        if item.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        # Size filter
        try:
            if item.stat().st_size > MAX_FILE_SIZE_BYTES:
                continue
        except OSError:
            continue

        # Binary check — try reading first 512 bytes as UTF-8
        try:
            with item.open("rb") as fh:
                sample = fh.read(512)
            sample.decode("utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        rel = str(item.relative_to(repo_path))
        indexable.append(rel)

    return sorted(indexable)


# ============================================================
# LOCAL FOLDER INPUT
# ============================================================

def load_local_folder(folder_path: str | Path) -> RepositoryInfo:
    """
    Accept a local directory path and return a RepositoryInfo.

    The directory is used in-place (not copied).
    """
    path = Path(folder_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Folder not found: {path}")
    if not path.is_dir():
        raise ValueError(f"Path is not a directory: {path}")

    repo_id = str(uuid.uuid4())
    name = path.name
    files = discover_files(path)

    return RepositoryInfo(
        repo_id=repo_id,
        name=name,
        path=path,
        source="local",
        is_temporary=False,
        file_count=len(files),
        files=files,
    )


# ============================================================
# ZIP ARCHIVE UPLOAD INPUT
# ============================================================

def load_zip_archive(zip_path: str | Path) -> RepositoryInfo:
    """
    Extract a .zip archive into the repos workspace and return a RepositoryInfo.

    The extracted directory IS temporary and will be cleaned up by
    cleanup_repository().
    """
    zip_path = Path(zip_path).resolve()

    if not zip_path.exists():
        raise FileNotFoundError(f"Zip archive not found: {zip_path}")
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File is not a valid zip archive: {zip_path}")

    workspace = _ensure_workspace()
    repo_id = str(uuid.uuid4())
    extract_dir = workspace / repo_id
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    # If the zip contained a single top-level folder, descend into it
    children = [c for c in extract_dir.iterdir()]
    if len(children) == 1 and children[0].is_dir():
        repo_root = children[0]
        name = repo_root.name
    else:
        repo_root = extract_dir
        name = zip_path.stem

    files = discover_files(repo_root)

    return RepositoryInfo(
        repo_id=repo_id,
        name=name,
        path=repo_root,
        source="upload",
        is_temporary=True,
        file_count=len(files),
        files=files,
    )


# ============================================================
# GITHUB URL INPUT
# ============================================================

def load_github_url(url: str) -> RepositoryInfo:
    """
    Clone a public GitHub repository into the repos workspace.

    Raises:
        ValueError: If the URL is not a valid Git URL.
        RuntimeError: If git clone fails.
    """
    url = url.strip()

    if not is_valid_git_url(url):
        raise ValueError(
            f"Invalid or unsupported Git URL: {url!r}. "
            "Only public HTTPS GitHub URLs are supported."
        )

    workspace = _ensure_workspace()
    repo_id = str(uuid.uuid4())
    clone_dir = workspace / repo_id
    clone_dir.mkdir(parents=True, exist_ok=True)

    name = _slug_from_url(url)

    # Ensure the URL ends with .git for clone
    clone_url = url if url.endswith(".git") else url + ".git"

    print(f"[RepositoryManager] Cloning {clone_url} → {clone_dir}")

    result = subprocess.run(
        ["git", "clone", "--depth", "1", clone_url, str(clone_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        # Clean up the empty directory on failure
        shutil.rmtree(clone_dir, ignore_errors=True)
        raise RuntimeError(
            f"git clone failed for {url!r}.\n"
            f"stderr: {result.stderr.strip()}"
        )

    print(f"[RepositoryManager] Clone complete.")

    files = discover_files(clone_dir)

    return RepositoryInfo(
        repo_id=repo_id,
        name=name,
        path=clone_dir,
        source="github",
        url=url,
        is_temporary=True,
        file_count=len(files),
        files=files,
    )


# ============================================================
# CLEANUP
# ============================================================

def cleanup_repository(repo_info: RepositoryInfo) -> None:
    """
    Remove a temporary repository from disk.

    Safe to call on non-temporary repos — it will do nothing.
    """
    if not repo_info.is_temporary:
        return

    if repo_info.path.exists():
        shutil.rmtree(repo_info.path, ignore_errors=True)
        print(f"[RepositoryManager] Cleaned up {repo_info.path}")
