from ingestion.scanner import scan_repository
from ingestion.parser import parse_file
from ingestion.extraction import extractor
from embeddings.embedder import create_embedding
from VectorIndex.vectordb import VectorStore
from generation.context_builder import build_context
from generation.generator import generate_answer

def build_index(repo_path):

    # 1. Scan repository
    files = scan_repository(repo_path)

    all_chunks = []
    all_embeddings = []

    # 2. Process every file
    for file_path in files:

        # Parse
        content = parse_file(file_path)

        # Chunk
        chunks = extractor(content, file_path)

        # Embed every chunk
        for chunk in chunks:

            embedding = create_embedding(
                chunk["content"]
            )

            all_chunks.append(chunk)
            all_embeddings.append(embedding)

    # 3. Create FAISS index
    dimension = len(all_embeddings[0])

    vector_store = VectorStore(dimension)

    # 4. Add vectors
    vector_store.add(all_embeddings)

    return vector_store, all_chunks

if __name__ == "__main__":
    from ingestion.scanner import scan_repository
    from ingestion.parser import parse_file
    from ingestion.extraction import extractor
    from embeddings.embedder import create_embedding
    from VectorIndex.vectordb import VectorStore


    files = scan_repository(".")

    all_chunks = []
    all_embeddings = []


    for file_path in files:

        content = parse_file(file_path)

        chunks = extractor(content, file_path)

        for chunk in chunks:

            embedding = create_embedding(chunk["content"])

            all_chunks.append(chunk)
            all_embeddings.append(embedding)


    print("Total chunks:", len(all_chunks))
    print("Total embeddings:", len(all_embeddings))
    print("Embedding dimension:", len(all_embeddings[0]))


    vector_store = VectorStore(len(all_embeddings[0]))

    vector_store.add(all_embeddings)

    print("Vectors stored in FAISS:", vector_store.index.ntotal)

    query = "Explain what happens when a user logs in?"

    query_embedding = create_embedding(query)

    scores, indices = vector_store.search(
        query_embedding,
        k=3
    )

    print("\nQUERY:", query)

    retrieved_chunks = []

    for score, index in zip(scores[0], indices[0]):

        chunk = all_chunks[index]

        retrieved_chunks.append(chunk)

        print("\n--------------------")
        print("Score:", score)
        print("File:", chunk["metadata"]["file"])
        print("Function:", chunk["metadata"]["function"])
        print("Lines:",
            chunk["metadata"]["start_line"],
            "-",
            chunk["metadata"]["end_line"])
        print("\nCode:")
        print(chunk["content"])

    context = build_context(retrieved_chunks)

    print("\nCONTEXT:")
    print(context)
    answer = generate_answer(query, context)

    print("\nANSWER:")  
    print(answer)