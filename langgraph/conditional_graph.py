from typing import TypedDict

from langgraph.graph import (
    StateGraph,
    START,
    END
)


# =========================================
# DEFINE STATE
# =========================================

class State(TypedDict):

    question: str
    route: str


# =========================================
# ROUTER NODE
# =========================================

def router_node(state):

    question = state["question"]

    print("\nROUTER NODE RUNNING")

    # Temporary simple routing logic
    #
    # If the question contains "explain",
    # treat it as complex.
    #
    # Otherwise treat it as simple.

    if "explain" in question.lower():

        route = "complex"

    else:

        route = "simple"

    print(
        "Selected route:",
        route
    )

    return {
        "route": route
    }


# =========================================
# SIMPLE NODE
# =========================================

def simple_node(state):

    print("\nSIMPLE NODE RUNNING")

    return {
        "route": state["route"]
    }


# =========================================
# COMPLEX NODE
# =========================================

def complex_node(state):

    print("\nCOMPLEX NODE RUNNING")

    return {
        "route": state["route"]
    }


# =========================================
# ROUTING FUNCTION
# =========================================

def decide_next_node(state):

    return state["route"]


# =========================================
# CREATE GRAPH
# =========================================

graph_builder = StateGraph(
    State
)


# =========================================
# ADD NODES
# =========================================

graph_builder.add_node(
    "router",
    router_node
)

graph_builder.add_node(
    "simple",
    simple_node
)

graph_builder.add_node(
    "complex",
    complex_node
)


# =========================================
# ADD START EDGE
# =========================================

graph_builder.add_edge(
    START,
    "router"
)


# =========================================
# ADD CONDITIONAL EDGE
# =========================================

graph_builder.add_conditional_edges(
    "router",
    decide_next_node,
    {
        "simple": "simple",
        "complex": "complex"
    }
)


# =========================================
# ADD END EDGES
# =========================================

graph_builder.add_edge(
    "simple",
    END
)

graph_builder.add_edge(
    "complex",
    END
)


# =========================================
# COMPILE GRAPH
# =========================================

graph = graph_builder.compile()


# =========================================
# TEST QUESTION
# =========================================

question = (
    "Explain what happens when a user logs in."
)


# =========================================
# RUN GRAPH
# =========================================

result = graph.invoke({
    "question": question,
    "route": ""
})


# =========================================
# FINAL RESULT
# =========================================

print("\n" + "=" * 50)
print("FINAL STATE")
print("=" * 50)

print(result)