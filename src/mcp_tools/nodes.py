"""MCP Tool을 그래프에서 사용할 독립 노드 함수로 변환합니다."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Protocol, TypeAlias

from src.mcp_tools.client import decode_tool_response


MCPToolCall: TypeAlias = Callable[..., str]
MCPNode: TypeAlias = Callable[[Mapping[str, object]], dict[str, object]]


class MCPToolServiceLike(Protocol):
    """노드 생성에 필요한 Tool 서비스 인터페이스."""

    search_documents: MCPToolCall
    lookup_document: MCPToolCall
    health_check: MCPToolCall
    get_collection_info: MCPToolCall


def _text(state: Mapping[str, object], *keys: str) -> str:
    for key in keys:
        value = state.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"필수 입력이 없습니다: {', '.join(keys)}")


def _documents(value: object) -> list[dict[str, object]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError("MCP documents는 배열이어야 합니다.")

    documents: list[dict[str, object]] = []
    for index, document in enumerate(value):
        if not isinstance(document, Mapping):
            raise ValueError(f"MCP documents[{index}]는 객체여야 합니다.")
        text = document.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"MCP documents[{index}].text가 비어 있습니다.")
        documents.append(dict(document))
    return documents


def _result(
    payload: Mapping[str, object],
    **fields: object,
) -> dict[str, object]:
    result = {
        "mcp_status": payload["status"],
        "mcp_result": dict(payload),
        **fields,
    }
    if isinstance(error := payload.get("error"), Mapping):
        result["mcp_error"] = dict(error)
    return result


def _run(
    action: Callable[[], dict[str, object]],
    *,
    clear_documents: bool = False,
) -> dict[str, object]:
    try:
        return action()
    except (TypeError, ValueError) as error:
        status, message = "invalid_request", str(error)
    except Exception:
        status = "internal_error"
        message = "MCP Tool 호출 중 내부 오류가 발생했습니다."

    payload = {"status": status, "error": {"message": message}}
    fields = (
        {"mcp_documents": [], "documents": []}
        if clear_documents
        else {}
    )
    return _result(payload, **fields)


def create_search_documents_node(
    search_documents: MCPToolCall,
    *,
    default_top_k: int = 5,
    default_score_threshold: float | None = None,
) -> MCPNode:
    """문서 검색 Tool 노드를 만듭니다."""

    if not callable(search_documents):
        raise TypeError("search_documents는 호출 가능한 함수여야 합니다.")

    def node(state: Mapping[str, object]) -> dict[str, object]:
        def action() -> dict[str, object]:
            query = _text(state, "rewrite_query", "origin_query")
            payload = decode_tool_response(
                search_documents(
                    query,
                    top_k=state.get("mcp_top_k", default_top_k),
                    score_threshold=state.get(
                        "mcp_score_threshold",
                        default_score_threshold,
                    ),
                    filters=state.get("mcp_filters"),
                )
            )
            documents = _documents(payload.get("documents"))
            return _result(
                payload,
                mcp_used_query=payload.get("used_query", query),
                mcp_documents=documents,
                documents=documents,
            )

        return _run(action, clear_documents=True)

    return node


def create_lookup_document_node(
    lookup_document: MCPToolCall,
    *,
    default_limit: int = 20,
) -> MCPNode:
    """문서 ID 정확 조회 Tool 노드를 만듭니다."""

    if not callable(lookup_document):
        raise TypeError("lookup_document는 호출 가능한 함수여야 합니다.")

    def node(state: Mapping[str, object]) -> dict[str, object]:
        def action() -> dict[str, object]:
            document_id = _text(state, "mcp_document_id", "document_id")
            payload = decode_tool_response(
                lookup_document(
                    document_id,
                    limit=state.get("mcp_limit", default_limit),
                )
            )
            documents = _documents(payload.get("documents"))
            return _result(
                payload,
                mcp_document_id=document_id,
                mcp_documents=documents,
                documents=documents,
            )

        return _run(action, clear_documents=True)

    return node


def create_health_check_node(health_check: MCPToolCall) -> MCPNode:
    """연결 상태 확인 Tool 노드를 만듭니다."""

    if not callable(health_check):
        raise TypeError("health_check는 호출 가능한 함수여야 합니다.")

    def node(_state: Mapping[str, object]) -> dict[str, object]:
        def action() -> dict[str, object]:
            payload = decode_tool_response(health_check())
            healthy = payload.get("healthy")
            if not isinstance(healthy, bool):
                raise ValueError("MCP health_check 응답에 healthy가 없습니다.")
            return _result(payload, mcp_healthy=healthy)

        return _run(action)

    return node


def create_collection_info_node(get_collection_info: MCPToolCall) -> MCPNode:
    """컬렉션 정보 확인 Tool 노드를 만듭니다."""

    if not callable(get_collection_info):
        raise TypeError("get_collection_info는 호출 가능한 함수여야 합니다.")

    def node(_state: Mapping[str, object]) -> dict[str, object]:
        def action() -> dict[str, object]:
            payload = decode_tool_response(get_collection_info())
            collection = payload.get("collection")
            if collection is not None and not isinstance(collection, Mapping):
                raise ValueError("MCP collection 응답은 객체여야 합니다.")
            value = dict(collection) if isinstance(collection, Mapping) else {}
            return _result(payload, mcp_collection=value)

        return _run(action)

    return node


def create_mcp_tool_nodes(service: MCPToolServiceLike) -> dict[str, MCPNode]:
    """MCP Tool별 독립 노드를 이름과 함께 반환합니다."""

    return {
        "mcp_search_documents": create_search_documents_node(
            service.search_documents
        ),
        "mcp_lookup_document": create_lookup_document_node(
            service.lookup_document
        ),
        "mcp_health_check": create_health_check_node(service.health_check),
        "mcp_get_collection_info": create_collection_info_node(
            service.get_collection_info
        ),
    }
