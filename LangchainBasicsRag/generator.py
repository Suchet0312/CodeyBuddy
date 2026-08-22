from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


def format_documents(documents):

    context = ""

    for doc in documents:

        file_name = doc.metadata["file"]

        context += f"""
SOURCE: {file_name}

CODE:
{doc.page_content}

"""

    return context


def create_generator():

    prompt = ChatPromptTemplate.from_template("""
You are CodeLens, an AI assistant that analyzes code repositories.

Answer the user's question using only the provided code context.

If the answer is not present in the context, say you cannot determine it.

CODE CONTEXT:
{context}

QUESTION:
{question}

Provide a concise, evidence-based answer.
""")

    llm = ChatOllama(
        model="qwen2:7b"
    )

    return prompt, llm