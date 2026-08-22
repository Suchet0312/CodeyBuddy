import os

SUPPORTED_EXTENSIONS = {
    ".py"
}

IGNORED_DIRS = {
    ".git",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode"
}


def scan_repository(root_path):
    files_found = []

    for root, dirs, files in os.walk(root_path):

        # Prevent os.walk() from entering ignored directories
        dirs[:] = [
            d for d in dirs
            if d not in IGNORED_DIRS
        ]

        for file in files:

            _, extension = os.path.splitext(file)

            if extension in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, file)
                files_found.append(full_path)

    return files_found