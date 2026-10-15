from functools import cache

from langgraph.graph import StateGraph, START, END

from src.graph.state import State
from src.graph.mcp_get_collection_info import (
    create_mcp_collection_info_graph_node,
)
from src.graph.mcp_health_check_node import create_mcp_health_check_graph_node
from src.graph.mcp_lookup_document import create_mcp_lookup_document_graph_node
from src.graph.mcp_search_documents import create_mcp_search_documents_graph_node
from src.graph.mcp_validation_node import mcp_validation_node
from src.mcp_tools.nodes import MCPToolServiceLike
from src.mcp_tools.server import MCPToolService, create_tool_service
from src.vector_db import create_team_retriever


class MCPGraphState(State, total=False):
    """MCP 서브그래프 안에서만 유지하는 Tool별 실행 상태입니다."""

    mcp_health_status: str
    mcp_health_result: dict[str, object]
    mcp_healthy: bool
    mcp_health_error: dict[str, object] | None

    mcp_collection_status: str
    mcp_collection_result: dict[str, object]
    mcp_collection: dict[str, object]
    mcp_collection_error: dict[str, object] | None

    mcp_search_status: str
    mcp_search_result: dict[str, object]
    mcp_used_query: str | None
    mcp_search_documents: list[dict[str, object]]
    mcp_search_error: dict[str, object] | None

    mcp_lookup_status: str
    mcp_lookup_result: dict[str, object]
    mcp_lookup_documents: list[dict[str, object]]
    mcp_lookup_error: dict[str, object] | None

    mcp_document_id: str | None
    mcp_should_end: bool
    mcp_top_k: int
    mcp_score_threshold: float | None
    mcp_filters: dict[str, object]
    mcp_limit: int


@cache
def _default_mcp_service() -> MCPToolService:
    """MCP 분기가 실제 실행될 때 팀 공용 Tool 서비스를 한 번만 만듭니다."""

    retriever = create_team_retriever(validate_collection_on_startup=False)
    return create_tool_service(retriever)


def _health_check() -> str:
    return _default_mcp_service().health_check()


def _get_collection_info() -> str:
    return _default_mcp_service().get_collection_info()


def _search_documents(
    query: str,
    *,
    top_k: int = 5,
    score_threshold: float | None = None,
    filters: dict[str, object] | None = None,
) -> str:
    return _default_mcp_service().search_documents(
        query,
        top_k=top_k,
        score_threshold=score_threshold,
        filters=filters,
    )


def _lookup_document(document_id: str, *, limit: int = 20) -> str:
    return _default_mcp_service().lookup_document(document_id, limit=limit)


def route_after_mcp_validation(state: MCPGraphState) -> str:
    if state.get("mcp_should_end"):
        return "end"
    return "lookup" if state.get("is_valid") else "retry"


def build_mcp_graph(service: MCPToolServiceLike | None = None):
    """구현된 MCP Tool 네 개를 실행 순서대로 연결합니다."""

    health_check = service.health_check if service else _health_check
    get_collection_info = (
        service.get_collection_info if service else _get_collection_info
    )
    search_documents = service.search_documents if service else _search_documents
    lookup_document = service.lookup_document if service else _lookup_document

    builder = StateGraph(MCPGraphState)
    builder.add_node(
        "mcp_health_check_node",
        create_mcp_health_check_graph_node(health_check),
    )
    builder.add_node(
        "mcp_get_collection_info",
        create_mcp_collection_info_graph_node(get_collection_info),
    )
    builder.add_node(
        "mcp_search_documents",
        create_mcp_search_documents_graph_node(search_documents),
    )
    builder.add_node("mcp_validation_node", mcp_validation_node)
    builder.add_node(
        "mcp_lookup_document",
        create_mcp_lookup_document_graph_node(lookup_document),
    )

    builder.add_edge(START, "mcp_health_check_node")
    builder.add_edge("mcp_health_check_node", "mcp_get_collection_info")
    builder.add_edge("mcp_get_collection_info", "mcp_search_documents")
    builder.add_edge("mcp_search_documents", "mcp_validation_node")
    builder.add_conditional_edges(
        "mcp_validation_node",
        route_after_mcp_validation,
        {
            "lookup": "mcp_lookup_document",
            "retry": "mcp_health_check_node",
            "end": END,
        },
    )
    builder.add_edge("mcp_lookup_document", END)

    return builder.compile()
