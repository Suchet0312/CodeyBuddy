import ollama


def generate_answer(question, context):

    prompt = f"""
You are CodeLens, an AI assistant that analyzes code repositories.

Answer the user's question using only the code context that is directly relevant to the question.

Ignore retrieved code that is not relevant.

Do not mention unrelated files, functions, or behavior.

When giving the answer, mention the relevant file, function, and line numbers.

If the answer is not present in the context, say that you do not have enough information.

QUESTION:
{question}

CODE CONTEXT:
{context}
"""

    response = ollama.generate(
        model="qwen2:7b",
        prompt=prompt
    )

    return response["response"]