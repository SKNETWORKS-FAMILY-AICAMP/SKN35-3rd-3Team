"""MCP ``health_check`` Tool을 LangGraph 상태에 연결합니다."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.mcp_tools import create_health_check_node


MCPToolCall = Callable[..., str]


def create_mcp_health_check_graph_node(
    health_check: MCPToolCall,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """실제 Tool 호출 함수가 주입된 얇은 그래프 노드를 만듭니다."""

    execute = create_health_check_node(health_check)

    def mcp_health_check_node(
        state: Mapping[str, object],
    ) -> dict[str, object]:
        result = execute(state)
        error = result.get("mcp_error")
        return {
            "mcp_health_status": result.get("mcp_status", "internal_error"),
            "mcp_health_result": result.get("mcp_result", {}),
            "mcp_healthy": result.get("mcp_healthy", False),
            "mcp_health_error": dict(error) if isinstance(error, Mapping) else None,
        }

    return mcp_health_check_node
