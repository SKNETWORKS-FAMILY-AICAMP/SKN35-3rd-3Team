"""MCP ``search_documents`` Tool을 LangGraph 상태에 연결합니다."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.mcp_tools import create_search_documents_node


MCPToolCall = Callable[..., str]


def create_mcp_search_documents_graph_node(
    search_documents: MCPToolCall,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """검색 Tool을 실행하고 메타데이터가 보존된 문서를 반환합니다."""

    execute = create_search_documents_node(search_documents)

    def mcp_search_documents(
        state: Mapping[str, object],
    ) -> dict[str, object]:
        result = execute(state)
        documents = result.get("mcp_documents")
        error = result.get("mcp_error")
        normalized_documents = documents if isinstance(documents, list) else []
        return {
            "mcp_search_status": result.get("mcp_status", "internal_error"),
            "mcp_search_result": result.get("mcp_result", {}),
            "mcp_used_query": result.get("mcp_used_query"),
            "mcp_search_documents": normalized_documents,
            "mcp_search_error": (
                dict(error) if isinstance(error, Mapping) else None
            ),
            "documents": normalized_documents,
        }

    return mcp_search_documents
