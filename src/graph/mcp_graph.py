from langgraph.graph import StateGraph, START, END
from src.graph.state import State

def build_mcp_graph():
    builder = StateGraph(State)

    builder.add_edge(START , END)
    
    return builder.compile()