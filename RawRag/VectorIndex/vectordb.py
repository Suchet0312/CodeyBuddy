import faiss
import numpy as np

class VectorStore:
    def __init__(self,dimension):
        self.index = faiss.IndexFlatIP(dimension)

    def add(self,embeddings):
        embeddings = np.array(embeddings).astype("float32")
        self.index.add(embeddings)

    def search(self,query_embedding,k = 3):
        query_embedding = np.array([query_embedding]).astype("float32")
        scores,indices = self.index.search(query_embedding,k)
        return scores,indices
