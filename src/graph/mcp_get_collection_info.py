"""MCP ``get_collection_info`` Tool을 LangGraph 상태에 연결합니다."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.mcp_tools import create_collection_info_node


MCPToolCall = Callable[..., str]


def create_mcp_collection_info_graph_node(
    get_collection_info: MCPToolCall,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """컬렉션 정보 Tool 결과를 MCP 전용 상태 키로 반환합니다."""

    execute = create_collection_info_node(get_collection_info)

    def mcp_get_collection_info(
        state: Mapping[str, object],
    ) -> dict[str, object]:
        result = execute(state)
        collection = result.get("mcp_collection")
        error = result.get("mcp_error")
        return {
            "mcp_collection_status": result.get("mcp_status", "internal_error"),
            "mcp_collection_result": result.get("mcp_result", {}),
            "mcp_collection": (
                dict(collection) if isinstance(collection, Mapping) else {}
            ),
            "mcp_collection_error": (
                dict(error) if isinstance(error, Mapping) else None
            ),
        }

    return mcp_get_collection_info
