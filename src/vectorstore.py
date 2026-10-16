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
    query_text: str | None = None
    sparse_indices: tuple[int, ...] | None = None
    sparse_values: tuple[float, ...] | None = None

    @classmethod
    def create(
        cls,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
        expected_dimension: int | None = None,
        query_text: str | None = None,
        sparse_indices: Sequence[int] | None = None,
        sparse_values: Sequence[float] | None = None,
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

        if (sparse_indices is None) != (sparse_values is None):
            raise ValueError(
                "Sparse Vector의 indices와 values는 함께 지정해야 합니다."
            )
        normalized_sparse_indices = None
        normalized_sparse_values = None
        if sparse_indices is not None and sparse_values is not None:
            normalized_sparse_indices = tuple(int(item) for item in sparse_indices)
            normalized_sparse_values = tuple(float(item) for item in sparse_values)
            if len(normalized_sparse_indices) != len(normalized_sparse_values):
                raise ValueError(
                    "Sparse Vector의 indices와 values 길이가 다릅니다."
                )
            if not normalized_sparse_indices:
                raise ValueError("Sparse Vector는 비어 있을 수 없습니다.")

        return cls(
            query_vector=vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=dict(filters or {}),
            query_text=query_text.strip() if query_text is not None else None,
            sparse_indices=normalized_sparse_indices,
            sparse_values=normalized_sparse_values,
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
class VectorDocument:
    """점수 계산 없이 Payload 조건으로 정확 조회한 문서 청크."""

    point_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.point_id.strip():
            raise ValueError("point_id는 비어 있을 수 없습니다.")
        if not self.text.strip():
            raise ValueError("조회 결과 text는 비어 있을 수 없습니다.")

    def to_document(self) -> dict[str, object]:
        """MCP 응답에서 사용할 구조화 문서로 변환합니다."""

        return {
            "point_id": self.point_id,
            "text": self.text,
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
    search_mode: str = "dense"
    sparse_vector_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vector_size": self.vector_size,
            "distance": self.distance,
            "points_count": self.points_count,
            "status": self.status,
            "vector_name": self.vector_name,
            "search_mode": self.search_mode,
            "sparse_vector_name": self.sparse_vector_name,
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

    def lookup(
        self,
        filters: Mapping[str, object],
        *,
        limit: int,
    ) -> Sequence[VectorDocument]:
        """임베딩 없이 Payload 조건과 정확히 일치하는 청크를 조회합니다."""

"""Vector DB 내부 인터페이스.

이 모듈은 특정 Qdrant Client 버전에 의존하지 않습니다. 실제 Qdrant 구현은
이 인터페이스를 따르며, MCP 계층은 Qdrant Client를 직접 호출하지 않습니다.
"""

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
    # dev의 Qdrant Sparse 검색에서도 같은 요청 객체를 사용합니다.
    query_text: str | None = None
    sparse_indices: tuple[int, ...] | None = None
    sparse_values: tuple[float, ...] | None = None

    @classmethod
    def create(
        cls,
        query_vector: Sequence[float],
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
        expected_dimension: int | None = None,
        query_text: str | None = None,
        sparse_indices: Sequence[int] | None = None,
        sparse_values: Sequence[float] | None = None,
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

        if (sparse_indices is None) != (sparse_values is None):
            raise ValueError("Sparse indices와 values는 함께 지정해야 합니다.")
        indices = values = None
        if sparse_indices is not None and sparse_values is not None:
            indices = tuple(int(value) for value in sparse_indices)
            values = tuple(float(value) for value in sparse_values)
            if not indices or len(indices) != len(values):
                raise ValueError("Sparse Vector는 비어 있지 않고 길이가 일치해야 합니다.")
            if any(value < 0 for value in indices) or any(not math.isfinite(value) for value in values):
                raise ValueError("Sparse Vector 인덱스는 음수가 아니며 값은 유한해야 합니다.")
        return cls(
            query_vector=vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=dict(filters or {}),
            query_text=query_text.strip() if query_text is not None else None,
            sparse_indices=indices,
            sparse_values=values,
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
    search_mode: str = "dense"
    sparse_vector_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "vector_size": self.vector_size,
            "distance": self.distance,
            "points_count": self.points_count,
            "status": self.status,
            "vector_name": self.vector_name,
            "search_mode": self.search_mode,
            "sparse_vector_name": self.sparse_vector_name,
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

    def lookup(self, filters: Mapping[str, object], *, limit: int) -> Sequence[VectorDocument]:
        """dev의 문서 ID 정확 조회 기능과 호환되는 공용 인터페이스."""


@dataclass(frozen=True)
class VectorDocument:
    """dev의 점수 없는 문서 조회 결과. 검색 결과와 별도로 유지합니다."""

    point_id: str
    text: str
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.point_id.strip() or not self.text.strip():
            raise ValueError("조회 문서의 ID와 본문은 비어 있을 수 없습니다.")

    def to_document(self) -> dict[str, object]:
        return {"point_id": self.point_id, "text": self.text, "metadata": dict(self.metadata)}

