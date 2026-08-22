from sentence_transformers import SentenceTransformer

import numpy as np
MODEL_NAME = "BAAI/bge-base-en-v1.5"

model = SentenceTransformer(MODEL_NAME)


def create_embedding(text):
    embedding = model.encode(text)

    return embedding

if __name__ == "__main__":

    text = "def login(username, password):"

    embedding = create_embedding(text)
    print(np.linalg.norm(embedding))
    print("Type:", type(embedding))
    print("Shape:", embedding.shape)
    print("First 5 values:", embedding[:5])