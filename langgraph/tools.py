from pathlib import Path
from langchain_core.tools import tool
PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

TARGET_REPO = (
    PROJECT_ROOT/"target_repo"
)

@tool
def list_repo_files():
    """
    List all files inside the CodeLens target repository.

    Returns:
        list[str]: Relative paths of repository files.
    """
    files = []
    for path in TARGET_REPO.rglob("*"):
        if path.is_file():
            relative_path = path.relative_to(TARGET_REPO)
            
            files.append(str(relative_path))

    return files

# =========================================
# READ REPOSITORY FILE
# =========================================

@tool
def read_repository_file(path: str):
    """
    Read the contents of a file inside the CodeLens
    target repository.

    Args:
        path: Relative path of the file inside target_repo.

    Returns:
        str: File contents or an error message.
    """

    requested_path = (
        TARGET_REPO / path
    ).resolve()

    # =========================================
    # SECURITY CHECK
    # =========================================

    try:

        requested_path.relative_to(
            TARGET_REPO.resolve()
        )

    except ValueError:

        return (
            "ERROR: Access denied. "
            "The requested path is outside "
            "the target repository."
        )

    # =========================================
    # FILE EXISTENCE CHECK
    # =========================================

    if not requested_path.exists():

        return (
            f"ERROR: File does not exist: {path}"
        )

    # =========================================
    # FILE TYPE CHECK
    # =========================================

    if not requested_path.is_file():

        return (
            f"ERROR: Path is not a file: {path}"
        )

    # =========================================
    # READ FILE
    # =========================================

    try:

        content = requested_path.read_text(
            encoding="utf-8"
        )

        return content

    except Exception as error:

        return (
            "ERROR: Could not read file. "
            f"Reason: {error}"
        )

print("\n" + "=" * 60)
print("READ FILE TOOL TEST")
print("=" * 60)

files = list_repo_files.invoke({})

if files:

    test_file = files[0]

    print("\nReading:")
    print(test_file)

    content = read_repository_file.invoke({
        "path": test_file
    })

    print("\nFile content:")
    print(content)

# =========================================
# SEARCH REPOSITORY
# =========================================

@tool
def search_repository(query: str):
    """
    Search for a text or code symbol across files
    inside the CodeLens target repository.

    Args:
        query: Text, function name, class name, variable,
               or code symbol to search for.

    Returns:
        str: Matching file paths and line numbers.
    """

    if not query.strip():

        return (
            "ERROR: Search query cannot be empty."
        )

    matches = []

    for path in TARGET_REPO.rglob("*"):

        if not path.is_file():
            continue

        try:

            content = path.read_text(
                encoding="utf-8"
            )

        except (UnicodeDecodeError, OSError):

            # Skip files that cannot be read as text.
            continue

        for line_number, line in enumerate(
            content.splitlines(),
            start=1
        ):

            if query.lower() in line.lower():

                relative_path = (
                    path.relative_to(TARGET_REPO)
                )

                matches.append(
                    f"{relative_path}:{line_number}: "
                    f"{line.strip()}"
                )

    if not matches:

        return (
            f"No matches found for: {query}"
        )

    return "\n".join(matches)

# =========================================
# CODELENS TOOL COLLECTION
# =========================================

CODELENS_TOOLS = [
    list_repo_files,
    read_repository_file,
    search_repository
]