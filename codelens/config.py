"""
codelens/config.py

Central configuration for file filtering, chunking,
retrieval, and workspace paths.
"""

from pathlib import Path

# ============================================================
# WORKSPACE
# ============================================================

# Root of the CodeLens project itself
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where cloned GitHub repos and uploaded repos are stored
REPOS_WORKSPACE = PROJECT_ROOT / "repos_workspace"

# ============================================================
# FILE FILTERING
# ============================================================

# Directories that are always skipped during file discovery
IGNORE_DIRS: set[str] = {
    ".git",
    ".github",
    ".svn",
    "venv",
    ".venv",
    "env",
    ".env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "bower_components",
    "dist",
    "build",
    "out",
    "target",        # Rust / Maven build output
    ".idea",
    ".vscode",
    ".DS_Store",
    "site-packages",
    "eggs",
    ".eggs",
}

# File extensions that are considered source code / text
ALLOWED_EXTENSIONS: set[str] = {
    # Python
    ".py",
    # JavaScript / TypeScript
    ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
    # Web
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    # JVM
    ".java", ".kt", ".scala", ".groovy",
    # C family
    ".c", ".h", ".cpp", ".cxx", ".cc", ".hpp",
    # Go / Rust / Swift
    ".go", ".rs", ".swift",
    # Ruby / PHP / Perl
    ".rb", ".php", ".pl",
    # Shell
    ".sh", ".bash", ".zsh", ".fish",
    # Config / data
    ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".env.example",
    # Docs
    ".md", ".rst", ".txt",
    # SQL
    ".sql",
    # XML
    ".xml",
}

# Hard upper limit on individual file size to index (bytes)
MAX_FILE_SIZE_BYTES: int = 500_000  # 500 KB

# ============================================================
# CHUNKING
# ============================================================

CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ============================================================
# RETRIEVAL
# ============================================================

DENSE_K: int = 5
BM25_K: int = 5
RERANK_TOP_K: int = 3

# ============================================================
# EMBEDDING MODEL
# ============================================================

EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"

# ============================================================
# LLM
# ============================================================

LLM_MODEL: str = "qwen2:7b"
