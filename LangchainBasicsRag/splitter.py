from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)


def split_documents(documents):

    splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    return chunks