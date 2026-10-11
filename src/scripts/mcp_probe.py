"""실제 설정으로 MCP Tool 목록과 Qdrant 연결 상태를 점검합니다."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import StdioServerParameters

from src.mcp_tools.runtime import create_team_mcp_server
from src.mcp_tools.sdk_client import probe_read_only_tools


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MCP SDK와 Qdrant의 읽기 전용 연결을 확인합니다.",
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "in-memory"),
        default="stdio",
        help="실제 하위 프로세스 연결 또는 테스트용 메모리 연결입니다.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="기본 .env 대신 읽을 환경변수 파일 경로입니다.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="입력하면 질문 임베딩과 Qdrant 검색까지 함께 확인합니다.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--document-id",
        default=None,
        help="입력하면 임베딩 없이 document_id 정확 조회까지 확인합니다.",
    )
    parser.add_argument(
        "--document-type",
        default=None,
        help="선택적 document_type 정확 일치 필터입니다.",
    )
    return parser


def _stdio_target(env_file: Path | None) -> StdioServerParameters:
    project_root = Path(__file__).resolve().parents[2]
    args = ["-m", "src.scripts.mcp_server"]
    if env_file is not None:
        args.extend(("--env-file", str(env_file)))
    return StdioServerParameters(
        command=sys.executable,
        args=args,
        cwd=project_root,
    )


async def _probe(
    transport: str,
    env_file: Path | None,
    query: str | None,
    document_id: str | None,
    top_k: int,
    document_type: str | None,
) -> dict[str, object]:
    target = (
        _stdio_target(env_file)
        if transport == "stdio"
        else create_team_mcp_server(env_file)
    )
    filters = (
        {"document_type": document_type}
        if document_type is not None
        else None
    )
    return await probe_read_only_tools(
        target,
        query=query,
        document_id=document_id,
        top_k=top_k,
        filters=filters,
    )


def main() -> None:
    args = _build_parser().parse_args()
    result = asyncio.run(
        _probe(
            args.transport,
            args.env_file,
            args.query,
            args.document_id,
            args.top_k,
            args.document_type,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
