"""
codelens/tools.py

Repository inspection tools for the CodeLens agent.

All three tools are dynamically bound to whichever repository
is currently loaded — no path is hardcoded.

Usage:
    from codelens.tools import build_tools

    tools = build_tools(indexed_repo)   # returns list of @tool functions
    llm_with_tools = llm.bind_tools(tools)

Design decisions:
  - Tools are constructed fresh per repository (closures capture repo_path).
  - Security: every file-read goes through a path-traversal check.
  - No import-time side effects (no auto-execution on import).
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from .indexer import IndexedRepository


# ============================================================
# TOOL FACTORY
# ============================================================

def build_tools(indexed_repo: "IndexedRepository") -> list:
    """
    Create and return the three CodeLens repository tools,
    all bound to *indexed_repo*.

    Returns a list suitable for llm.bind_tools().
    """

    repo_root: Path = indexed_repo.repo_info.path.resolve()
    repo_name: str  = indexed_repo.repo_info.name

    # --------------------------------------------------------
    # TOOL 1 — list_repo_files
    # --------------------------------------------------------

    @tool
    def list_repo_files(pattern: str = "") -> str:
        """
        List all source files in the repository.

        Args:
            pattern: Optional glob pattern to filter results,
                     e.g. "*.py", "src/**/*.ts".
                     Leave empty to list all files.

        Returns:
            Newline-separated list of relative file paths,
            or an error message.
        """
        files = indexed_repo.repo_info.files

        if pattern.strip():
            files = [
                f for f in files
                if fnmatch.fnmatch(f, pattern.strip())
            ]

        if not files:
            return (
                f"No files found"
                + (f" matching pattern '{pattern}'" if pattern.strip() else "")
                + f" in repository '{repo_name}'."
            )

        header = f"Repository: {repo_name}  ({len(files)} files)"
        return header + "\n" + "\n".join(files)

    # --------------------------------------------------------
    # TOOL 2 — read_repository_file
    # --------------------------------------------------------

    @tool
    def read_repository_file(path: str) -> str:
        """
        Read the contents of a specific file in the repository.

        Args:
            path: Relative path of the file within the repository,
                  e.g. "src/auth.py" or "lib/utils.ts".

        Returns:
            Full file contents, or an error message.
        """
        if not path or not path.strip():
            return "ERROR: File path cannot be empty."

        # Resolve and enforce the repository boundary
        try:
            requested = (repo_root / path.strip()).resolve()
        except Exception:
            return f"ERROR: Invalid path: {path!r}"

        # Path-traversal security check
        try:
            requested.relative_to(repo_root)
        except ValueError:
            return (
                "ERROR: Access denied. "
                "The requested path is outside the repository boundary."
            )

        if not requested.exists():
            return f"ERROR: File does not exist: {path}"

        if not requested.is_file():
            return f"ERROR: Path is not a file: {path}"

        # Size guard — refuse to dump enormous files into the context
        try:
            size = requested.stat().st_size
            if size > 200_000:
                return (
                    f"ERROR: File is too large to display ({size:,} bytes). "
                    "Use search_repository to inspect specific sections."
                )
        except OSError:
            pass

        try:
            content = requested.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"ERROR: Could not read file: {exc}"

        return f"FILE: {path}\n\n{content}"

    # --------------------------------------------------------
    # TOOL 3 — search_repository
    # --------------------------------------------------------

    @tool
    def search_repository(query: str) -> str:
        """
        Search the repository for a function, class, variable,
        import, or any text pattern.

        Performs a case-insensitive line-by-line text search across
        all indexed files in the repository.

        Args:
            query: Text, function name, class name, variable name,
                   or code symbol to search for.

        Returns:
            Matching file paths with line numbers and matching lines,
            or a message indicating no matches were found.
        """
        if not query or not query.strip():
            return "ERROR: Search query cannot be empty."

        query_lower = query.strip().lower()
        matches: list[str] = []

        for rel_path in indexed_repo.repo_info.files:
            abs_path = repo_root / rel_path

            try:
                content = abs_path.read_text(
                    encoding="utf-8", errors="replace"
                )
            except OSError:
                continue

            for line_no, line in enumerate(content.splitlines(), start=1):
                if query_lower in line.lower():
                    matches.append(
                        f"{rel_path}:{line_no}: {line.rstrip()}"
                    )

        if not matches:
            return (
                f"No matches found for: {query!r}\n"
                f"Repository: {repo_name}"
            )

        header = (
            f"Search results for: {query!r}  "
            f"({len(matches)} match{'es' if len(matches) != 1 else ''})"
        )
        return header + "\n" + "\n".join(matches)

    # --------------------------------------------------------
    # Return all three tools
    # --------------------------------------------------------

    return [list_repo_files, read_repository_file, search_repository]
