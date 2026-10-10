"""Vector DB 내부 인터페이스.

이 모듈은 특정 Qdrant Client 버전에 의존하지 않습니다. 실제 Qdrant 구현은
이 인터페이스를 따르며, MCP 계층은 Qdrant Client를 직접 호출하지 않습니다.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class VectorStoreError(RuntimeError):
    """Vector DB 내부 오류와 안전한 오류 코드를 함께 전달합니다."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.safe_message = message


def _validated_vector(values: Sequence[float]) -> tuple[float, ...]:
    """검색 벡터를 유한한 실수 튜플로 변환합니다."""

    if isinstance(values, (str, bytes)) or not values:
        raise ValueError("query_vector는 비어 있지 않은 숫자 배열이어야 합니다.")

    vector: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"query_vector[{index}]는 숫자여야 합니다.")

        numeric_value = float(value)
        if not math.isfinite(numeric_value):
            raise ValueError(f"query_vector[{index}]는 유한한 숫자여야 합니다.")
        vector.append(numeric_value)

    return tuple(vector)


@dataclass(frozen=True)
class VectorSearchRequest:
    """Vector Store가 받는 MCP 비종속 검색 요청."""

    query_vector: tuple[float, ...]
    top_k: int = 5
    score_threshold: float | None = None
    filters: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
        expected_dimension: int | None = None,
    ) -> VectorSearchRequest:
        """일반 입력값을 검증된 검색 요청으로 변환합니다."""

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k는 정수여야 합니다.")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k는 1 이상 100 이하여야 합니다.")

        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(
                score_threshold,
                (int, float),
            ):
                raise TypeError("score_threshold는 숫자 또는 null이어야 합니다.")
            score_threshold = float(score_threshold)
            if not math.isfinite(score_threshold):
                raise ValueError("score_threshold는 유한한 숫자여야 합니다.")

        vector = _validated_vector(query_vector)
        if expected_dimension is not None and len(vector) != expected_dimension:
            raise ValueError(
                "질문 벡터 차원이 컬렉션과 일치하지 않습니다: "
                f"expected={expected_dimension}, actual={len(vector)}"
            )

        return cls(
            query_vector=vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=dict(filters or {}),
        )


@dataclass(frozen=True)
class VectorSearchHit:
    """Vector DB가 상위 계층에 반환하는 검색 결과 한 건."""

    point_id: str
    score: float
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.point_id.strip():
            raise ValueError("point_id는 비어 있을 수 없습니다.")
        if not self.text.strip():
            raise ValueError("검색 결과 text는 비어 있을 수 없습니다.")
        if not math.isfinite(float(self.score)):
            raise ValueError("검색 score는 유한한 숫자여야 합니다.")

    def to_document(self) -> dict[str, object]:
        """MCP 이전 단계에서 사용할 구조화 문서로 변환합니다."""

        return {
            "point_id": self.point_id,
            "text": self.text,
            "score": float(self.score),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VectorCollectionInfo:
    """외부 비밀값을 포함하지 않는 컬렉션 정보."""

    name: str
    vector_size: int
    distance: str
    points_count: int
    status: str
    vector_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vector_size": self.vector_size,
            "distance": self.distance,
            "points_count": self.points_count,
            "status": self.status,
            "vector_name": self.vector_name,
        }


@runtime_checkable
class VectorStore(Protocol):
    """향후 Qdrant 구현체가 따라야 하는 최소 인터페이스."""

    def health_check(self) -> bool:
        """Vector DB 연결 가능 여부를 반환합니다."""

    def get_collection_info(self) -> VectorCollectionInfo:
        """현재 사용할 컬렉션 정보를 반환합니다."""

    def search(self, request: VectorSearchRequest) -> Sequence[VectorSearchHit]:
        """벡터 검색 한 번을 수행합니다."""

