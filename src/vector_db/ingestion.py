"""전처리 청크를 Qdrant 적재 직전 Point로 변환합니다.

이 모듈은 Qdrant Client와 MCP를 import하지 않습니다. 따라서 전처리 결과의
형식과 벡터 차원을 서버 연결 전에 단위 테스트할 수 있습니다.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from src.const.qdrant_config import EmbeddingSettings
from src.const.vector_data_contract import (
    PreparedChunk,
    infer_vector_dimension,
    normalize_vector,
    validate_unique_records,
)


@runtime_checkable
class DocumentEmbeddingProvider(Protocol):
    """적재할 문서 본문을 벡터로 변환하는 최소 인터페이스."""

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> Sequence[Sequence[float]]:
        """입력 순서와 같은 순서의 문서 벡터를 반환합니다."""


@dataclass(frozen=True)
class PreparedPoint:
    """Qdrant SDK 객체로 바꾸기 전의 저장 독립 Point."""

    point_id: str
    vector: tuple[float, ...]
    payload: Mapping[str, object]


def _utc_now_iso() -> str:
    """초 단위 UTC ISO 8601 시각을 반환합니다."""

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _validated_batch_size(batch_size: int) -> int:
    """배치 크기를 검사하고 정상 값을 반환합니다."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size는 정수여야 합니다.")
    if batch_size < 1:
        raise ValueError("batch_size는 1 이상이어야 합니다.")
    return batch_size


def _validate_existing_vector_metadata(
    record: PreparedChunk,
    settings: EmbeddingSettings,
) -> None:
    """기존 벡터가 현재 팀 임베딩 규격과 같은지 확인합니다."""

    expected_values = {
        "embedding_provider": settings.provider,
        "embedding_model": settings.model,
        "embedding_dimension": settings.dimension,
        "embedding_normalized": settings.normalized,
    }
    actual_values = {
        "embedding_provider": record.embedding_provider,
        "embedding_model": record.embedding_model,
        "embedding_dimension": record.embedding_dimension,
        "embedding_normalized": record.embedding_normalized,
    }

    for field_name, expected_value in expected_values.items():
        actual_value = actual_values[field_name]
        if actual_value is None or expected_value is None:
            continue
        if actual_value != expected_value:
            raise ValueError(
                "기존 벡터의 임베딩 정보가 현재 설정과 다릅니다: "
                f"field={field_name}, expected={expected_value}, "
                f"actual={actual_value}"
            )


def generate_document_vectors(
    records: Iterable[PreparedChunk],
    *,
    embedder: DocumentEmbeddingProvider,
    settings: EmbeddingSettings,
    batch_size: int = 64,
) -> tuple[PreparedChunk, ...]:
    """벡터가 없는 문서만 배치 임베딩해 적재 가능한 레코드로 만듭니다.

    이미 벡터가 있는 레코드는 다시 API에 보내지 않습니다. 대신 현재 설정과
    차원이 호환되는지 확인합니다. 반환 순서는 입력 순서와 같습니다.
    """

    settings.validate_for_vector_work()
    assert settings.provider is not None
    assert settings.model is not None
    assert settings.dimension is not None

    validated_batch_size = _validated_batch_size(batch_size)
    record_list = list(records)
    validate_unique_records(record_list)

    generated_records = list(record_list)
    pending_indexes: list[int] = []
    for index, record in enumerate(record_list):
        record.validate(require_vector=False)
        if record.vector is None:
            pending_indexes.append(index)
            continue

        record.validate(
            require_vector=True,
            expected_dimension=settings.dimension,
        )
        _validate_existing_vector_metadata(record, settings)

    for start in range(0, len(pending_indexes), validated_batch_size):
        batch_indexes = pending_indexes[
            start : start + validated_batch_size
        ]
        batch_texts = [record_list[index].text for index in batch_indexes]
        batch_vectors = list(embedder.embed_documents(batch_texts))

        if len(batch_vectors) != len(batch_indexes):
            raise RuntimeError(
                "임베딩 응답 개수가 입력 문서 개수와 다릅니다: "
                f"expected={len(batch_indexes)}, actual={len(batch_vectors)}"
            )

        for record_index, vector in zip(
            batch_indexes,
            batch_vectors,
            strict=True,
        ):
            embedded_record = replace(
                record_list[record_index],
                vector=normalize_vector(vector),
                embedding_provider=settings.provider,
                embedding_model=settings.model,
                embedding_dimension=settings.dimension,
                embedding_normalized=settings.normalized,
            )
            embedded_record.validate(
                require_vector=True,
                expected_dimension=settings.dimension,
            )
            generated_records[record_index] = embedded_record

    return tuple(generated_records)


def prepare_upsert_points(
    records: Iterable[PreparedChunk],
    *,
    expected_dimension: int,
    indexed_at: str | None = None,
) -> tuple[PreparedPoint, ...]:
    """검증된 청크를 Qdrant 적재 후보 Point로 변환합니다."""

    if isinstance(expected_dimension, bool) or not isinstance(
        expected_dimension,
        int,
    ):
        raise TypeError("expected_dimension은 정수여야 합니다.")
    if expected_dimension < 1:
        raise ValueError("expected_dimension은 1 이상이어야 합니다.")

    record_list = list(records)
    validate_unique_records(record_list)

    inferred_dimension = infer_vector_dimension(record_list)
    if inferred_dimension is not None and inferred_dimension != expected_dimension:
        raise ValueError(
            "입력 벡터 차원이 컬렉션 차원과 일치하지 않습니다: "
            f"expected={expected_dimension}, actual={inferred_dimension}"
        )

    indexed_time = indexed_at or _utc_now_iso()
    points: list[PreparedPoint] = []
    for record in record_list:
        record.validate(
            require_vector=True,
            expected_dimension=expected_dimension,
        )
        assert record.vector is not None
        points.append(
            PreparedPoint(
                point_id=record.point_id(),
                vector=record.vector,
                payload=record.payload(indexed_at=indexed_time),
            )
        )

    return tuple(points)


def iter_point_batches(
    points: Sequence[PreparedPoint],
    *,
    batch_size: int = 64,
) -> Iterator[tuple[PreparedPoint, ...]]:
    """대량 적재를 위해 Point를 일정한 크기로 나눕니다."""

    validated_batch_size = _validated_batch_size(batch_size)

    for start in range(0, len(points), validated_batch_size):
        yield tuple(points[start : start + validated_batch_size])
