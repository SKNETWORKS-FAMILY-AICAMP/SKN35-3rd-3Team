"""공식 MCP SDK 등록부와 분리된 조회 Tool 업무 로직.

검색·상태 확인·컬렉션 조회 결과를 문자열 JSON으로 반환합니다. 실제 Tool
등록과 Transport 실행은 ``src.mcp_tools.runtime`` 및 ``src.scripts.mcp_server``가
담당하며, 이 모듈은 실행 환경과 테스트에서 같은 업무 규칙을 제공합니다.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass

from src.mcp_tools.contracts import (
    LookupDocumentRequest,
    SearchDocumentsRequest,
    serialize_tool_response,
)
from src.rag.retriever import SearchBackend
from src.vectorstore import VectorStoreError


logger = logging.getLogger(__name__)


def _external_error_status(error_code: str) -> str:
    """Vector DB 내부 오류를 MCP 외부 상태값으로 변환합니다."""

    if error_code == "connection_error":
        return "connection_error"
    if error_code in {"invalid_input", "invalid_query"}:
        return "invalid_request"
    if error_code == "no_hits":
        return "empty_result"
    return "internal_error"


def _vector_store_error_response(
    error: VectorStoreError,
    **fields: object,
) -> str:
    """Vector DB 오류를 일관된 MCP JSON 응답으로 변환합니다."""

    return serialize_tool_response(
        _external_error_status(error.code),
        **fields,
        error={"code": error.code, "message": error.safe_message},
    )


def _invalid_request_response(error: Exception) -> str:
    """잘못된 Tool 입력을 일관된 MCP JSON 응답으로 변환합니다."""

    return serialize_tool_response(
        "invalid_request",
        error={"message": str(error)},
    )


def _internal_error_response(
    message: str,
    **fields: object,
) -> str:
    """예상하지 못한 내부 오류의 안전한 MCP JSON 응답을 만듭니다."""

    logger.exception(message)
    return serialize_tool_response(
        "internal_error",
        **fields,
        error={"message": message},
    )


@dataclass
class MCPToolService:
    """SDK Tool 등록부와 분리된 MCP 업무 서비스."""

    backend: SearchBackend

    def search_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> str:
        """Vector DB 검색 한 번을 수행하고 JSON 문자열을 반환합니다."""

        try:
            request = SearchDocumentsRequest.create(
                query,
                top_k=top_k,
                score_threshold=score_threshold,
                filters=filters,
            )
            hits = self.backend.search_documents(
                request.query,
                top_k=request.top_k,
                score_threshold=request.score_threshold,
                filters=request.filters,
            )
        except (TypeError, ValueError) as error:
            return _invalid_request_response(error)
        except VectorStoreError as error:
            return _vector_store_error_response(error)
        except Exception:
            return _internal_error_response(
                "검색 처리 중 내부 오류가 발생했습니다."
            )

        documents = [hit.to_document() for hit in hits]
        if not documents:
            return serialize_tool_response(
                "empty_result",
                original_query=request.query,
                used_query=request.query,
                retry_count=0,
                documents=[],
            )

        return serialize_tool_response(
            "success",
            original_query=request.query,
            used_query=request.query,
            retry_count=0,
            documents=documents,
        )

    def lookup_document(
        self,
        document_id: str,
        *,
        limit: int = 20,
    ) -> str:
        """임베딩 없이 document_id가 일치하는 청크를 조회합니다."""

        try:
            request = LookupDocumentRequest.create(document_id, limit=limit)
            documents = self.backend.lookup_document(
                request.document_id,
                limit=request.limit,
            )
        except (TypeError, ValueError) as error:
            return _invalid_request_response(error)
        except VectorStoreError as error:
            return _vector_store_error_response(error)
        except Exception:
            return _internal_error_response(
                "문서 정확 조회 중 내부 오류가 발생했습니다."
            )

        response_documents = [document.to_document() for document in documents]
        status = "success" if response_documents else "empty_result"
        return serialize_tool_response(
            status,
            document_id=request.document_id,
            documents=response_documents,
        )

    def health_check(self) -> str:
        """MCP·Vector DB 연결 상태를 문자열 JSON으로 반환합니다."""

        try:
            healthy = self.backend.health_check()
        except VectorStoreError as error:
            return _vector_store_error_response(
                error,
                healthy=False,
            )
        except Exception:
            return _internal_error_response(
                "상태 확인 중 내부 오류가 발생했습니다.",
                healthy=False,
            )

        if not healthy:
            return serialize_tool_response("connection_error", healthy=False)
        return serialize_tool_response("success", healthy=True)

    def get_collection_info(self) -> str:
        """안전한 컬렉션 정보를 문자열 JSON으로 반환합니다."""

        try:
            collection_info = self.backend.get_collection_info()
        except VectorStoreError as error:
            return _vector_store_error_response(error)
        except Exception:
            return _internal_error_response(
                "컬렉션 조회 중 내부 오류가 발생했습니다."
            )

        return serialize_tool_response(
            "success",
            collection=collection_info.to_dict(),
        )


def create_tool_service(backend: SearchBackend) -> MCPToolService:
    """MCP Tool 등록과 테스트에서 공통으로 사용할 서비스를 만듭니다."""

    return MCPToolService(backend=backend)
