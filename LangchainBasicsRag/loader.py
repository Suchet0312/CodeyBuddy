from langchain_core.documents import Document
import os
IGNORE_DIRS = {
    ".git",
    "venv",
    "__pycache__",
    ".idea",
    ".vscode"
}

def load_repo(root_path:str)->list[Document]:
    documents = []
    for root,dirs,files in os.walk(root_path):
        dirs[:] = [
            d for d in dirs
            if d not in IGNORE_DIRS
        ]
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root,file)

            try:
                with open(
                    file_path,"r",encoding="utf-8"
                ) as f:
                    content = f.read()

                document = Document(
                    page_content=content,
                    metadata = {
                        "file":file_path,
                        "language":"python",
                    }
                )
                documents.append(document)
            except Exception as e:
                print(f"error reading {file_path}: {e}")

    return documents