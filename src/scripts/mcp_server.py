"""팀 Qdrant 검색 MCP Server 실행 진입점."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, cast

from src.mcp_tools.runtime import create_team_mcp_server

Transport = Literal["stdio", "streamable-http"]
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


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


def main() -> None:
    args = _build_parser().parse_args()
    run_server(
        transport=cast(Transport, args.transport),
        env_file=args.env_file,
        host=args.host,
        port=args.port,
        path=args.path,
    )


if __name__ == "__main__":
    main()
