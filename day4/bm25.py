from langchain_community.retrievers import BM25Retriever


def create_bm25_retriever(chunks, k=3):

    retriever = BM25Retriever.from_documents(
        chunks
    )

    retriever.k = k

    return retriever