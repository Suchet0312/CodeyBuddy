from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    message: str

def greeting_node(state: State):
    print("Greeting node is running")
    return {
        "message": state["message"] + " from langgraph"
    }

# 1. Pass the State class to StateGraph
graph_builder = StateGraph(State)

# 2. Add the node explicitly with (name, function)
graph_builder.add_node("greeting", greeting_node)

# 3. Define the edges using the registered node name
graph_builder.add_edge(START, "greeting")
graph_builder.add_edge("greeting", END)

# 4. Compile and invoke
graph = graph_builder.compile()

result = graph.invoke({
    "message": "Hello"
})

print("\nFinal Result:")
print(result)