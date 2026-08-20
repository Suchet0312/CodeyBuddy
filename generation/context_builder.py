def build_context(chunks):

    context = ""

    for chunk in chunks:

        metadata = chunk["metadata"]

        context += f"""
SOURCE: {metadata["file"]}
FUNCTION: {metadata["function"]}
LINES: {metadata["start_line"]}-{metadata["end_line"]}

{chunk["content"]}

--------------------
"""

    return context