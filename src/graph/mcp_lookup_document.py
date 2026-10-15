"""MCP ``lookup_document`` Tool을 LangGraph 상태에 연결합니다."""

from __future__ import annotations

from collections.abc import Callable, Mapping

from src.mcp_tools import create_lookup_document_node


MCPToolCall = Callable[..., str]


def create_mcp_lookup_document_graph_node(
    lookup_document: MCPToolCall,
) -> Callable[[Mapping[str, object]], dict[str, object]]:
    """검증 노드가 고른 document_id로 정확 조회를 수행합니다."""

    execute = create_lookup_document_node(lookup_document)

    def mcp_lookup_document(
        state: Mapping[str, object],
    ) -> dict[str, object]:
        result = execute(state)
        documents = result.get("mcp_documents")
        error = result.get("mcp_error")
        normalized_documents = documents if isinstance(documents, list) else []
        # 문서 ID 조회는 문서 전체의 일부만 반환합니다. 조문·질문과 직접 관련돼
        # 먼저 검색된 청크를 앞에 유지해야 조회의 임의 순서로 근거가 사라지지 않습니다.
        combined = []
        seen = set()
        for document in [*(state.get("mcp_search_documents") or []), *normalized_documents]:
            if not isinstance(document, dict) or not document.get("text"):
                continue
            meta = document.get("metadata") or {}
            identity = document.get("point_id") or (meta.get("document_id"), meta.get("chunk_id"))
            if identity not in seen:
                combined.append(document)
                seen.add(identity)
        return {
            "mcp_lookup_status": result.get("mcp_status", "internal_error"),
            "mcp_lookup_result": result.get("mcp_result", {}),
            "mcp_lookup_documents": normalized_documents,
            "mcp_lookup_error": (
                dict(error) if isinstance(error, Mapping) else None
            ),
            "documents": combined,
        }

    return mcp_lookup_document
