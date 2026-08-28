from typing import TypedDict
from langgraph.graph import (
    START,
    StateGraph,
    END
)

class State(TypedDict):
    message:str

def first_node(state):
    print("first node is running")

    return {
        "message":(
            state["message"] + "-> processed by first node"
        )
    }

def second_node(state):
    print("second node is running")

    return {
        "message":(
            state["message"] + "-> processed by second node"
        )
    }

graph_builder = StateGraph(
    State
)

graph_builder.add_node("first",first_node)

graph_builder.add_node("second",second_node)

graph_builder.add_edge(START,"first")

graph_builder.add_edge("first","second")

graph_builder.add_edge("second",END)

graph = graph_builder.compile()


result = graph.invoke({
    "message":"hello"
})

print("\nFINAL RESULT:")

print(result)