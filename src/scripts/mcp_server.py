"""팀 Qdrant 검색 MCP Server 실행 진입점."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Literal, cast

from src.mcp_tools.contracts import parse_tool_response
from src.mcp_tools.runtime import create_team_mcp_server

Transport = Literal["stdio", "streamable-http"]
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})
EXPECTED_TOOLS = {
    "health_check",
    "get_collection_info",
    "search_documents",
    "lookup_document",
}
CHECK_QUERY = "계약 해지 시 환불 기준은 무엇인가요?"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qdrant 조회 기능을 MCP Tool로 제공합니다.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "streamable-http"),
        default="stdio",
        help="로컬 MCP Host는 stdio, HTTP 배포는 streamable-http를 사용합니다.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="기본 .env 대신 읽을 환경변수 파일 경로입니다.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--path", default="/mcp")
    parser.add_argument(
        "--check-tools",
        action="store_true",
        help="임시 stdio Server로 읽기 전용 Tool 네 개를 호출하고 종료합니다.",
    )
    return parser


def run_server(
    *,
    transport: Transport = "stdio",
    env_file: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    path: str = "/mcp",
) -> None:
    """전달받은 설정으로 팀 MCP Server를 실행합니다."""

    if transport not in {"stdio", "streamable-http"}:
        raise ValueError(f"지원하지 않는 MCP transport입니다: {transport}")
    if transport == "streamable-http" and host not in LOOPBACK_HOSTS:
        raise SystemExit(
            "인증이 없는 MCP HTTP 서버는 로컬 호스트에만 바인딩할 수 있습니다. "
            "외부 공개가 필요하면 인증·TLS를 제공하는 별도 게이트웨이를 구성하세요."
        )
    mcp = create_team_mcp_server(env_file)

    if transport == "stdio":
        mcp.run()
        return

    mcp.run(
        transport="streamable-http",
        host=host,
        port=port,
        streamable_http_path=path,
    )


def _tool_payload(result: object) -> dict[str, object]:
    """MCP TextContent를 팀 Tool 응답 객체로 변환합니다."""

    response = "".join(
        item.text
        for item in getattr(result, "content", ())
        if isinstance(getattr(item, "text", None), str)
    )
    return parse_tool_response(response)


async def check_tools(env_file: Path | None = None) -> None:
    """실제 stdio 연결로 MCP Tool 네 개를 한 번씩 호출합니다."""

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_args = ["-m", "src.scripts.mcp_server", "--transport", "stdio"]
    if env_file is not None:
        server_args.extend(("--env-file", str(env_file)))
    server = StdioServerParameters(command=sys.executable, args=server_args)

    async with stdio_client(server) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = {tool.name for tool in tools.tools}
            missing_tools = EXPECTED_TOOLS - tool_names
            if missing_tools:
                raise RuntimeError(
                    "등록되지 않은 MCP Tool: "
                    + ", ".join(sorted(missing_tools))
                )

            health = _tool_payload(await session.call_tool("health_check"))
            info = _tool_payload(
                await session.call_tool("get_collection_info")
            )
            search = _tool_payload(
                await session.call_tool(
                    "search_documents",
                    {"query": CHECK_QUERY, "top_k": 3},
                )
            )
            documents = search.get("documents")
            if not isinstance(documents, list) or not documents:
                raise RuntimeError("MCP 검색 결과가 없습니다.")
            metadata = documents[0].get("metadata", {})
            document_id = metadata.get("document_id")
            lookup = _tool_payload(
                await session.call_tool(
                    "lookup_document",
                    {"document_id": document_id, "limit": 20},
                )
            )

    results = {
        "health_check": health,
        "get_collection_info": info,
        "search_documents": search,
        "lookup_document": lookup,
    }
    failed = [
        name
        for name, result in results.items()
        if result["status"] != "success"
    ]
    if failed:
        raise RuntimeError("MCP Tool 호출 실패: " + ", ".join(failed))

    print("등록 Tool:", ", ".join(sorted(tool_names)))
    for name, result in results.items():
        documents = result.get("documents")
        count = f", {len(documents)}건" if isinstance(documents, list) else ""
        print(f"{name}: {result['status']}{count}")


def main() -> int:
    args = _build_parser().parse_args()
    if args.check_tools:
        asyncio.run(check_tools(args.env_file))
        return 0

    run_server(
        transport=cast(Transport, args.transport),
        env_file=args.env_file,
        host=args.host,
        port=args.port,
        path=args.path,
    )
    return 0
