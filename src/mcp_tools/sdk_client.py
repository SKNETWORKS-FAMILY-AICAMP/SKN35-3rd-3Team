"""공식 MCP Client로 조회 Tool을 점검하는 최소 헬퍼."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mcp import Client
from mcp.types import TextContent

from src.mcp_tools.client import decode_tool_response


def _decode_call_result(result: Any, name: str) -> dict[str, object]:
    if result.is_error:
        raise RuntimeError(f"MCP Tool 호출에 실패했습니다: {name}")

    for content in result.content:
        if isinstance(content, TextContent):
            return decode_tool_response(content.text)

    raise ValueError(f"MCP Tool 응답에 문자열 결과가 없습니다: {name}")


async def list_tool_names(target: Any) -> tuple[str, ...]:
    """MCP Server 객체나 Streamable HTTP URL의 Tool 이름을 조회합니다."""

    async with Client(target) as client:
        result = await client.list_tools()
        return tuple(tool.name for tool in result.tools)


async def call_json_tool(
    target: Any,
    name: str,
    arguments: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """문자열 JSON을 반환하는 MCP Tool을 호출하고 검증해 복원합니다."""

    async with Client(target) as client:
        result = await client.call_tool(name, dict(arguments or {}))

    return _decode_call_result(result, name)


async def probe_read_only_tools(
    target: Any,
    *,
    query: str | None = None,
    document_id: str | None = None,
    top_k: int = 5,
    filters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """한 연결에서 조회 Tool과 선택적 실제 검색을 확인합니다."""

    async with Client(target) as client:
        tools = await client.list_tools()
        health = await client.call_tool("health_check", {})
        collection = await client.call_tool("get_collection_info", {})

        search = None
        if query is not None:
            search = await client.call_tool(
                "search_documents",
                {
                    "query": query,
                    "top_k": top_k,
                    "filters": dict(filters or {}),
                },
            )

        lookup = None
        if document_id is not None:
            lookup = await client.call_tool(
                "lookup_document",
                {"document_id": document_id, "limit": top_k},
            )

    result = {
        "tools": [tool.name for tool in tools.tools],
        "health_check": _decode_call_result(health, "health_check"),
        "collection": _decode_call_result(
            collection,
            "get_collection_info",
        ),
    }
    if search is not None:
        result["search_documents"] = _decode_call_result(
            search,
            "search_documents",
        )
    if lookup is not None:
        result["lookup_document"] = _decode_call_result(
            lookup,
            "lookup_document",
        )
    return result
