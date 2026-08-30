import sys
from pathlib import Path


# =========================================
# PROJECT PATH SETUP
# =========================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

sys.path.append(
    str(PROJECT_ROOT)
)

from tools import (
    CODELENS_TOOLS
)

from LangchainBasicsRag.generator import (
    create_generator
)


# =========================================
# LOAD LLM
# =========================================

prompt, llm = create_generator()


# =========================================
# BIND TOOLS TO LLM
# =========================================

llm_with_tools = llm.bind_tools(
    CODELENS_TOOLS
)


# =========================================
# TEST TOOL CALLING
# =========================================

question = """
I need to understand the structure of the
repository before investigating it.
List all files in the repository.
"""

response = llm_with_tools.invoke(
    question
)


# =========================================
# INSPECT RESPONSE
# =========================================

print("=" * 60)
print("LLM TOOL CALL TEST")
print("=" * 60)

print("\nResponse:")
print(response)

print("\nTool calls:")

print(
    response.tool_calls
)