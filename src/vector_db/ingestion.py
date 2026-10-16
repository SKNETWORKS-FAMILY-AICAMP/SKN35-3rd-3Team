"""전처리 청크를 Qdrant 적재 직전 Point로 변환합니다.

이 모듈은 Qdrant Client와 MCP를 import하지 않습니다. 따라서 전처리 결과의
형식과 벡터 차원을 서버 연결 전에 단위 테스트할 수 있습니다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Protocol, runtime_checkable

import tiktoken

from src.const.qdrant_config import EmbeddingSettings
from src.const.vector_data_contract import (
    PreparedChunk,
    create_content_hash,
    infer_vector_dimension,
    normalize_vector,
    validate_unique_records,
)
from src.vector_db.sparse_embedder import SparseEmbedding


_OPENAI_EMBEDDING_TOKEN_LIMIT = 8_191
_OFFICIAL_PDF_PAGE_MARKER = re.compile(
    r"(?m)^\[공식 PDF 페이지 (\d+) 공통기준\]"
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
    sparse_indices: tuple[int, ...] | None = None
    sparse_values: tuple[float, ...] | None = None
    sparse_vector_name: str | None = None


@dataclass(frozen=True)
class _EmbeddingInput:
    """API 제한을 반영한 Dense 임베딩 입력과 감사 정보."""

    text: str
    original_token_count: int | None = None

    @property
    def truncated(self) -> bool:
        return self.original_token_count is not None


@lru_cache(maxsize=None)
def _openai_encoding(model: str) -> object:
    """OpenAI 모델에 맞는 토크나이저를 재사용합니다."""

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")


def _embedding_input(
    record: PreparedChunk,
    settings: EmbeddingSettings,
) -> _EmbeddingInput:
    """전처리 문맥을 사용하고 OpenAI 입력 한도를 안전하게 적용합니다."""

    text = record.embedding_input_text
    if settings.provider != "openai" or settings.model is None:
        return _EmbeddingInput(text=text)

    encoding = _openai_encoding(settings.model)
    tokens = encoding.encode(text)
    if len(tokens) <= _OPENAI_EMBEDDING_TOKEN_LIMIT:
        return _EmbeddingInput(text=text)
    return _EmbeddingInput(
        text=encoding.decode(tokens[:_OPENAI_EMBEDDING_TOKEN_LIMIT]),
        original_token_count=len(tokens),
    )


def _split_passages(
    passages: object,
    *,
    source_text: str,
    part_start: int,
    part_end: int,
) -> list[dict[str, object]]:
    """부모 본문의 passage를 분할 본문 기준 위치로 옮깁니다."""

    if not isinstance(passages, Sequence) or isinstance(
        passages,
        (str, bytes),
    ):
        return []

    selected: list[dict[str, object]] = []
    for value in passages:
        if not isinstance(value, Mapping):
            continue
        start = value.get("start")
        end = value.get("end")
        if not isinstance(start, int) or not isinstance(end, int):
            continue

        overlap_start = max(start, part_start)
        overlap_end = min(end, part_end)
        if overlap_start >= overlap_end:
            continue

        passage = dict(value)
        passage.update(
            {
                "parent_start": start,
                "parent_end": end,
                "start": overlap_start - part_start,
                "end": overlap_end - part_start,
                "text": source_text[overlap_start:overlap_end],
            }
        )
        if overlap_start != start or overlap_end != end:
            passage.pop("search_text", None)
        selected.append(passage)
    return selected


def _page_tables(value: object, page_number: int) -> object:
    """페이지 번호가 있는 표 목록을 현재 파생 청크에 맞게 제한합니다."""

    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return value
    return [
        item
        for item in value
        if not isinstance(item, Mapping)
        or item.get("page_number") in (None, page_number)
    ]


def _split_openai_oversized_record(
    record: PreparedChunk,
    *,
    model: str,
) -> tuple[PreparedChunk, ...]:
    """페이지 경계가 있는 OpenAI 초과 청크를 안정적인 ID로 분할합니다.

    페이지 경계를 식별할 수 없는 청크는 기존 입력 제한 처리에 맡겨 이전
    동작을 유지합니다.
    """

    if record.vector is not None:
        return (record,)

    encoding = _openai_encoding(model)
    embedding_text = record.embedding_input_text
    original_token_count = len(encoding.encode(embedding_text))
    if original_token_count <= _OPENAI_EMBEDDING_TOKEN_LIMIT:
        return (record,)

    if not embedding_text.endswith(record.text):
        return (record,)
    context_prefix = embedding_text[: -len(record.text)]
    markers = list(_OFFICIAL_PDF_PAGE_MARKER.finditer(record.text))
    if len(markers) < 2 or markers[0].start() != 0:
        return (record,)

    spans = [
        (
            int(marker.group(1)),
            marker.start(),
            (
                markers[index + 1].start()
                if index + 1 < len(markers)
                else len(record.text)
            ),
        )
        for index, marker in enumerate(markers)
    ]
    parent_metadata = dict(record.extra_metadata)
    parent_start = parent_metadata.get("start", 0)
    if not isinstance(parent_start, int):
        parent_start = 0

    split_records: list[PreparedChunk] = []
    for split_index, (page_number, part_start, part_end) in enumerate(
        spans,
        start=1,
    ):
        part_text = record.text[part_start:part_end]
        part_embedding_text = context_prefix + part_text
        part_token_count = len(encoding.encode(part_embedding_text))
        if part_token_count > _OPENAI_EMBEDDING_TOKEN_LIMIT:
            raise ValueError(
                "페이지 단위 분할 후에도 OpenAI 입력 한도를 초과합니다: "
                f"chunk_id={record.chunk_id}, page={page_number}, "
                f"tokens={part_token_count}"
            )

        part_metadata = dict(parent_metadata)
        part_metadata.update(
            {
                "embedding_text": part_embedding_text,
                "page_numbers": [page_number],
                "start": parent_start + part_start,
                "end": parent_start + part_end,
                "atomic": False,
                "oversize_atomic_unit": False,
                "split_from_chunk_id": record.chunk_id,
                "split_index": split_index,
                "split_count": len(spans),
                "split_reason": "openai_embedding_token_limit",
                "embedding_input_original_token_count": (
                    original_token_count
                ),
                "embedding_input_token_limit": (
                    _OPENAI_EMBEDDING_TOKEN_LIMIT
                ),
            }
        )
        part_metadata["search_passages"] = _split_passages(
            parent_metadata.get("search_passages"),
            source_text=record.text,
            part_start=part_start,
            part_end=part_end,
        )
        if "pdf_tables" in part_metadata:
            part_metadata["pdf_tables"] = _page_tables(
                part_metadata["pdf_tables"],
                page_number,
            )

        split_records.append(
            replace(
                record,
                chunk_id=f"{record.chunk_id}__page_{page_number:04d}",
                text=part_text,
                page=page_number,
                content_hash=create_content_hash(part_text),
                extra_metadata=part_metadata,
            )
        )
    return tuple(split_records)


def expand_embedding_records(
    records: Iterable[PreparedChunk],
    *,
    settings: EmbeddingSettings,
) -> tuple[PreparedChunk, ...]:
    """제공자 입력 제한에 맞춰 적재용 청크를 결정적으로 확장합니다."""

    record_list = list(records)
    validate_unique_records(record_list)
    if settings.provider != "openai" or settings.model is None:
        return tuple(record_list)

    expanded_records = tuple(
        child
        for record in record_list
        for child in _split_openai_oversized_record(
            record,
            model=settings.model,
        )
    )
    validate_unique_records(expanded_records)
    return expanded_records


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


def _with_embedding(
    record: PreparedChunk,
    embedding_input: _EmbeddingInput,
    vector: Sequence[float],
    settings: EmbeddingSettings,
) -> PreparedChunk:
    """생성한 벡터와 임베딩 정보를 원본 청크에 반영합니다."""

    extra_metadata = dict(record.extra_metadata)
    if embedding_input.truncated:
        extra_metadata.update(
            {
                "embedding_input_truncated": True,
                "embedding_input_token_count": (
                    embedding_input.original_token_count
                ),
                "embedding_input_token_limit": _OPENAI_EMBEDDING_TOKEN_LIMIT,
            }
        )

    embedded_record = replace(
        record,
        vector=normalize_vector(vector),
        embedding_provider=settings.provider,
        embedding_model=settings.model,
        embedding_dimension=settings.dimension,
        embedding_normalized=settings.normalized,
        extra_metadata=extra_metadata,
    )
    embedded_record.validate(
        require_vector=True,
        expected_dimension=settings.dimension,
    )
    return embedded_record


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
    record_list = list(
        expand_embedding_records(records, settings=settings)
    )

    generated_records = list(record_list)
    pending_records: list[tuple[int, _EmbeddingInput]] = []
    for index, record in enumerate(record_list):
        record.validate(require_vector=False)
        if record.vector is None:
            pending_records.append((index, _embedding_input(record, settings)))
            continue

        record.validate(
            require_vector=True,
            expected_dimension=settings.dimension,
        )
        _validate_existing_vector_metadata(record, settings)

    for start in range(0, len(pending_records), validated_batch_size):
        batch = pending_records[
            start : start + validated_batch_size
        ]
        batch_texts = [embedding_input.text for _, embedding_input in batch]
        batch_vectors = list(embedder.embed_documents(batch_texts))

        if len(batch_vectors) != len(batch):
            raise RuntimeError(
                "임베딩 응답 개수가 입력 문서 개수와 다릅니다: "
                f"expected={len(batch)}, actual={len(batch_vectors)}"
            )

        for (record_index, embedding_input), vector in zip(
            batch,
            batch_vectors,
            strict=True,
        ):
            generated_records[record_index] = _with_embedding(
                record_list[record_index],
                embedding_input,
                vector,
                settings,
            )

    return tuple(generated_records)


def prepare_upsert_points(
    records: Iterable[PreparedChunk],
    *,
    expected_dimension: int,
    indexed_at: str | None = None,
    sparse_vector_name: str | None = None,
    sparse_vectors: Sequence[SparseEmbedding] | None = None,
) -> tuple[PreparedPoint, ...]:
    """검증된 청크를 Qdrant 적재 후보 Point로 변환합니다."""

    if isinstance(expected_dimension, bool) or not isinstance(
        expected_dimension,
        int,
    ):
        raise TypeError("expected_dimension은 정수여야 합니다.")
    if expected_dimension < 1:
        raise ValueError("expected_dimension은 1 이상이어야 합니다.")
    if (sparse_vector_name is None) != (sparse_vectors is None):
        raise ValueError(
            "Sparse Vector 이름과 벡터 목록은 함께 설정해야 합니다."
        )

    record_list = list(records)
    validate_unique_records(record_list)
    if sparse_vectors is not None and len(sparse_vectors) != len(record_list):
        raise ValueError(
            "Sparse Vector 개수가 입력 청크 개수와 다릅니다."
        )

    inferred_dimension = infer_vector_dimension(record_list)
    if inferred_dimension is not None and inferred_dimension != expected_dimension:
        raise ValueError(
            "입력 벡터 차원이 컬렉션 차원과 일치하지 않습니다: "
            f"expected={expected_dimension}, actual={inferred_dimension}"
        )

    indexed_time = indexed_at or _utc_now_iso()
    points: list[PreparedPoint] = []
    for index, record in enumerate(record_list):
        record.validate(
            require_vector=True,
            expected_dimension=expected_dimension,
        )
        assert record.vector is not None
        sparse_vector = sparse_vectors[index] if sparse_vectors else None
        points.append(
            PreparedPoint(
                point_id=record.point_id(),
                vector=record.vector,
                payload=record.payload(indexed_at=indexed_time),
                sparse_indices=(
                    sparse_vector.indices if sparse_vector is not None else None
                ),
                sparse_values=(
                    sparse_vector.values if sparse_vector is not None else None
                ),
                sparse_vector_name=sparse_vector_name,
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
