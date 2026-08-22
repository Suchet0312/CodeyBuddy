from loader import load_repo
from splitter import split_documents
from embedder import get_embeddings
from vector_store import create_vector_store
from generator import format_documents, create_generator


# 1. Load target repository
documents = load_repo("../target_repo")

print("Total documents:", len(documents))


# 2. Split documents
chunks = split_documents(documents)

print("Total chunks:", len(chunks))


# 3. Load embedding model
embeddings = get_embeddings()


# 4. Create FAISS vector store
vector_store = create_vector_store(
    chunks,
    embeddings
)


# 5. Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# 6. Ask question
question = "Where is authentication implemented?"


# 7. Retrieve relevant documents
retrieved_documents = retriever.invoke(question)


# 8. Build context
context = format_documents(
    retrieved_documents
)


# 9. Create prompt and LLM
prompt, llm = create_generator()


# 10. Create final prompt
messages = prompt.invoke({
    "context": context,
    "question": question
})


# 11. Generate answer
response = llm.invoke(messages)


print("\nQUESTION:")
print(question)

print("\nANSWER:")
print(response.content)