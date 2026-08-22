from langchain_huggingface import HuggingFaceEmbeddings


MODEL_NAME = "BAAI/bge-base-en-v1.5"


def get_embeddings():

    embeddings = HuggingFaceEmbeddings(
        model_name=MODEL_NAME
    )

    return embeddings