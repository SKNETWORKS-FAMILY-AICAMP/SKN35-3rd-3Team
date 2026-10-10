"""Vector DB 검색 서비스.

LangGraph 상태나 MCP SDK에 의존하지 않습니다. 실제 임베딩 모델과 Vector Store
구현체를 주입받아 검색 한 번을 수행합니다.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.vectorstore import (
    VectorCollectionInfo,
    VectorSearchHit,
    VectorSearchRequest,
    VectorStore,
)


@runtime_checkable
class QueryEmbedder(Protocol):
    """문서와 같은 모델로 질문 벡터를 만드는 인터페이스."""

    def embed_query(self, text: str) -> Sequence[float]:
        """질문 문자열을 벡터로 변환합니다."""


@runtime_checkable
class SearchBackend(Protocol):
    """MCP Tool 업무 로직이 의존하는 검색 인터페이스."""

    def health_check(self) -> bool:
        """하위 Vector DB 상태를 반환합니다."""

    def get_collection_info(self) -> VectorCollectionInfo:
        """안전한 컬렉션 정보를 반환합니다."""

    def search_documents(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        filters: Mapping[str, object],
    ) -> Sequence[VectorSearchHit]:
        """질문을 임베딩하고 Vector DB 검색 한 번을 수행합니다."""


@dataclass
class DocumentRetriever:
    """임베딩과 Vector Store를 연결하는 검색 서비스."""

    embedder: QueryEmbedder
    vector_store: VectorStore
    expected_dimension: int | None = None

    def __post_init__(self) -> None:
        """시작 시 컬렉션 차원을 한 번 확인해 검색 요청에 재사용합니다."""

        if self.expected_dimension is None:
            collection_info = self.vector_store.get_collection_info()
            self.expected_dimension = collection_info.vector_size
        if self.expected_dimension < 1:
            raise ValueError("expected_dimension은 1 이상이어야 합니다.")

    def health_check(self) -> bool:
        return self.vector_store.health_check()

    def get_collection_info(self) -> VectorCollectionInfo:
        return self.vector_store.get_collection_info()

    def search_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> Sequence[VectorSearchHit]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("query는 비어 있을 수 없습니다.")

        query_vector = self.embedder.embed_query(normalized_query)
        request = VectorSearchRequest.create(
            query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
            expected_dimension=self.expected_dimension,
        )
        return self.vector_store.search(request)


def retriever(state: object) -> dict[str, object]:
    """기존 자리 함수.

    LangGraph 연결은 별도 담당 범위이므로 현재 단계에서는 구현하지 않습니다.
    """

    return {}
