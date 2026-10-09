from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from src.graph.state import State
from src.graph.image_node import image_node
from src.graph.rag_graph import build_rag_graph
from src.graph.mcp_graph import build_mcp_graph
from src.graph.general_agent import general_agent
from src.const.model import create_llm , create_llm_gmn
from src.const.config import EMBED_MODEL
from src.prompt.classify_prompt import CLASSIFY_PROMPT

def route_start(state: State) -> str:
    llm = create_llm_gmn()
    message = [
        ("system", CLASSIFY_PROMPT),
        ("user", state["origin_query"])
    ]
    # label = llm.invoke(message).content.strip().lower()
    label = llm.invoke(message).text.strip().lower()
    return label if label in {"general", "rag", "mcp", "out_of_scope"} else "rag"

def build_workflow():
    builder = StateGraph(State)

    builder.add_node("image", image_node)
    builder.add_node("general", general_agent)
    builder.add_node("rag", build_rag_graph())
    builder.add_node("mcp", build_mcp_graph())

    builder.add_edge(START , "image")
    builder.add_conditional_edges("image", route_start,
        {"general": "general", "out_of_scope" : "general", "rag": "rag", "mcp" : "mcp"})
    
    builder.add_edge("general", END)
    builder.add_edge("rag", END)
    builder.add_edge("mcp", END)

    return builder.compile(checkpointer=InMemorySaver())