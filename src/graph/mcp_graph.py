from langgraph.graph import StateGraph, START, END
from src.graph.state import State
from src.graph.mcp_get_collection_info import mcp_get_collection_info
from src.graph.mcp_health_check_node import mcp_health_check_node
from src.graph.mcp_lookup_document import mcp_lookup_document
from src.graph.mcp_search_documents import mcp_search_documents
from src.graph.mcp_validation_node import mcp_validation_node

def route_after_validation(state: State):
    return "mcp_lookup_document"
    # if state["is_valid"]:
    #     return "mcp_lookup_document"
    # else:
    #     return "mcp_health_check_node"

def build_mcp_graph():
    builder = StateGraph(State)
    # 노드 생성
    builder.add_node("mcp_health_check_node" , mcp_health_check_node)
    builder.add_node("mcp_get_collection_info" , mcp_get_collection_info)
    builder.add_node("mcp_search_documents" , mcp_search_documents)
    builder.add_node("mcp_validation_node" , mcp_validation_node)
    builder.add_node("mcp_lookup_document" , mcp_lookup_document)
    print("노드 생성 완료")
    
    builder.add_edge(START , "mcp_health_check_node")
    builder.add_edge("mcp_health_check_node", "mcp_get_collection_info")
    builder.add_edge("mcp_get_collection_info", "mcp_search_documents")
    builder.add_edge("mcp_search_documents", "mcp_validation_node")
    builder.add_conditional_edges( "mcp_validation_node", route_after_validation,
            {
                "mcp_lookup_document": "mcp_lookup_document",
                "mcp_health_check_node": "mcp_health_check_node",
            }
        )
    builder.add_edge("mcp_lookup_document", END)
    
    return builder.compile()