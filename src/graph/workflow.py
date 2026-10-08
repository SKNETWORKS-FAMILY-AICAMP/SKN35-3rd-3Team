from langgraph.graph import StateGraph, START, END
from src.graph.state import State
from src.graph.rag_graph import build_rag_graph
from src.graph.mcp_graph import build_mcp_graph
from src.graph.general_agent import general_agent
from src.const.model import create_embedding
from src.const.config import EMBED_MODEL
from src.prompt.classify_prompt import CLASSIFY_PROMPT

def route_start(state: State) -> str:
    llm = create_embedding()
    message = [
        ("system", CLASSIFY_PROMPT),
        ("user", state["origin_query"])
    ]
    return llm.invoke(message).content

def build_workflow():
    builder = StateGraph(State)

    builder.add_node("general", general_agent)
    builder.add_node("rag", build_rag_graph)
    builder.add_node("mcp", build_mcp_graph)

    builder.add_conditional_edges(START, route_start,
        {"general": "general", "rag": "rag", "mcp" : "mcp"})
    
    builder.add_edge("general", END)
    builder.add_edge("rag", END)
    builder.add_edge("mcp", END)

    return builder.compile()