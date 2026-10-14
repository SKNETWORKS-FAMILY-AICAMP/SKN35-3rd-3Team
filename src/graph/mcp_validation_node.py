"""MCP Tool 실행 결과를 검사해 라우터 입력을 만듭니다."""

from __future__ import annotations

from collections.abc import Mapping

MAX_MCP_RETRIES = 2


def _document_id(document: object) -> str | None:
    if not isinstance(document, Mapping):
        return None

    metadata = document.get("metadata")
    if isinstance(metadata, Mapping):
        value = metadata.get("document_id")
        if isinstance(value, str) and value.strip():
            return value.strip()

    value = document.get("document_id")
    return value.strip() if isinstance(value, str) and value.strip() else None


def mcp_validation_node(state: Mapping[str, object]) -> dict[str, object]:
    """정확 조회 전에 Health, Collection, Search 결과를 검증합니다."""

    documents = state.get("mcp_search_documents")
    if not isinstance(documents, list):
        documents = []

    document_id = _document_id(documents[0]) if documents else None
    reasons: list[str] = []

    if state.get("mcp_health_status") != "success" or not state.get(
        "mcp_healthy"
    ):
        reasons.append("MCP 또는 Qdrant 연결을 확인할 수 없습니다.")
    if state.get("mcp_collection_status") != "success":
        reasons.append("검색 컬렉션 정보를 확인할 수 없습니다.")
    if state.get("mcp_search_status") != "success" or not documents:
        reasons.append("MCP 검색 결과가 없습니다.")
    elif document_id is None:
        reasons.append("검색 결과에 document_id가 없습니다.")

    is_valid = not reasons
    retry_count = int(state.get("retry_count", 0))
    if not is_valid:
        retry_count += 1

    return {
        "is_valid": is_valid,
        "validation_reason": (
            " ".join(reasons) if reasons else "MCP 검색 결과가 유효합니다."
        ),
        "retry_count": retry_count,
        "mcp_should_end": not is_valid and retry_count >= MAX_MCP_RETRIES,
        "mcp_document_id": document_id,
    }
