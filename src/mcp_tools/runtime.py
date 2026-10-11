"""공식 MCP Python SDK에 검색 업무 서비스를 등록합니다.

이 모듈은 MCP SDK와 내부 업무 로직의 연결만 담당합니다. Qdrant 설정,
임베딩 생성 및 검색 구현은 기존 ``src.vector_db`` 계층에 위임합니다.
"""

from __future__ import annotations

from pathlib import Path

from mcp.server import MCPServer

from src.mcp_tools.server import MCPToolService, create_tool_service
from src.vector_db import create_team_retriever

SERVER_NAME = "SKN35-3rd-3Team Qdrant Search"


def create_mcp_server(service: MCPToolService) -> MCPServer:
    """주입받은 업무 서비스를 MCP 읽기 전용 Tool로 노출합니다."""

    mcp = MCPServer(SERVER_NAME)

    @mcp.tool()
    def search_documents(
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: dict[str, str | int | bool] | None = None,
    ) -> str:
        """질문과 유사한 문서를 Qdrant에서 한 번 검색합니다.

        filters는 팀 공용 Keyword 인덱스인 document_id, document_type,
        source_name만 지원합니다. 모든 MCP 사용자는 같은 규칙을 적용받습니다.
        """

        return service.search_documents(
            query,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
        )

    @mcp.tool()
    def lookup_document(
        document_id: str,
        limit: int = 20,
    ) -> str:
        """임베딩 없이 document_id가 같은 문서 청크를 정확 조회합니다."""

        return service.lookup_document(document_id, limit=limit)

    @mcp.tool()
    def health_check() -> str:
        """MCP 검색 서비스와 Qdrant 연결 상태를 확인합니다."""

        return service.health_check()

    @mcp.tool()
    def get_collection_info() -> str:
        """API Key를 제외한 검색 컬렉션 정보를 확인합니다."""

        return service.get_collection_info()

    return mcp


def create_team_mcp_server(
    env_path: str | Path | None = None,
) -> MCPServer:
    """팀 환경설정으로 Retriever와 MCP Server를 조립합니다.

    Qdrant가 일시적으로 응답하지 않아도 MCP 서버와 ``health_check`` Tool은
    시작할 수 있도록 컬렉션 원격 검증을 지연합니다. 컬렉션이나 Point를
    생성·수정·삭제하지 않습니다.
    """

    retriever = create_team_retriever(
        env_path,
        validate_collection_on_startup=False,
    )
    return create_mcp_server(create_tool_service(retriever))
