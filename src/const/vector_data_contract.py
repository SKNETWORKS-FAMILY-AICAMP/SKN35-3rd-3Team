"""전처리 담당자와 Vector DB 담당자 사이의 공통 데이터 계약.

이 모듈은 Qdrant에 연결하거나 데이터를 저장하지 않습니다.
텍스트 청크 또는 미리 계산된 벡터를 여러 사용자가 같은 형식으로
전달할 수 있도록 입력값을 검증하고 안정적인 식별자를 만듭니다.
"""

from __future__ import annotations

import hashlib
import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final


POINT_ID_NAMESPACE: Final = uuid.UUID(
    "d66a4bc4-ce87-4c7d-bc59-c937799d4d83"
)


def _required_text(value: object, field_name: str) -> str:
    """필수 문자열을 정리하고 빈 값이면 오류를 발생시킵니다."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name}은 문자열이어야 합니다.")

    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError(f"{field_name}은 비어 있을 수 없습니다.")

    return normalized_value


def _optional_text(value: object, field_name: str) -> str | None:
    """선택 문자열의 공백 값을 ``None``으로 변환합니다."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{field_name}은 문자열 또는 null이어야 합니다.")

    normalized_value = value.strip()
    return normalized_value or None


def _optional_page(value: object) -> int | None:
    """선택 페이지 번호를 검사합니다."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("metadata.page는 정수 또는 null이어야 합니다.")
    if value < 1:
        raise ValueError("metadata.page는 1 이상이어야 합니다.")

    return value


def _optional_positive_int(value: object, field_name: str) -> int | None:
    """선택 양의 정수를 검사합니다."""

    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name}은 정수 또는 null이어야 합니다.")
    if value < 1:
        raise ValueError(f"{field_name}은 1 이상이어야 합니다.")

    return value


def _optional_boolean(value: object, field_name: str) -> bool | None:
    """선택 불리언 값을 검사합니다."""

    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError(f"{field_name}은 true/false 또는 null이어야 합니다.")

    return value


def normalize_vector(value: object) -> tuple[float, ...]:
    """벡터 입력을 유한한 실수 튜플로 변환합니다."""

    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("vector는 숫자 배열이어야 합니다.")
    if not value:
        raise ValueError("vector는 빈 배열일 수 없습니다.")

    vector: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"vector[{index}]는 숫자여야 합니다.")

        numeric_item = float(item)
        if not math.isfinite(numeric_item):
            raise ValueError(f"vector[{index}]는 유한한 숫자여야 합니다.")
        vector.append(numeric_item)

    return tuple(vector)


def _optional_vector(value: object) -> tuple[float, ...] | None:
    """선택 벡터를 유한한 실수 튜플로 변환합니다."""

    if value is None:
        return None
    return normalize_vector(value)


def create_content_hash(text: str) -> str:
    """정제된 청크 본문의 SHA-256 해시를 반환합니다."""

    normalized_text = _required_text(text, "text")
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedChunk:
    """Qdrant 적재 직전 단계에서 사용하는 표준 청크.

    ``vector``가 ``None``이면 텍스트 청크를 임베딩해야 합니다.
    ``vector``가 존재하면 미리 계산된 벡터 입력으로 처리할 수 있습니다.
    """

    document_id: str
    chunk_id: str
    text: str
    vector: tuple[float, ...] | None = None
    source_name: str | None = None
    source_path: str | None = None
    page: int | None = None
    section: str | None = None
    content_hash: str | None = None
    document_updated_at: str | None = None
    published_at: str | None = None
    embedding_provider: str | None = None
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_normalized: bool | None = None
    extra_metadata: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> PreparedChunk:
        """JSON·JSONL 등에서 읽은 사전을 표준 청크로 변환합니다."""

        metadata_value = data.get("metadata", {})
        if metadata_value is None:
            metadata_value = {}
        if not isinstance(metadata_value, Mapping):
            raise TypeError("metadata는 객체 형태여야 합니다.")

        known_metadata_names = {
            "source_name",
            "source_path",
            "source_url",
            "page",
            "section",
            "content_hash",
            "document_updated_at",
            "published_at",
            "indexed_at",
            "embedding_provider",
            "embedding_model",
            "embedding_dimension",
            "embedding_normalized",
            # Payload 핵심 필드는 extra_metadata로 덮어쓸 수 없습니다.
            "text",
            "document_id",
            "chunk_id",
            "vector",
        }

        source_path = metadata_value.get("source_path")
        if source_path is None:
            source_path = metadata_value.get("source_url")

        text = _required_text(data.get("text"), "text")
        content_hash = _optional_text(
            metadata_value.get("content_hash"),
            "metadata.content_hash",
        )

        return cls(
            document_id=_required_text(
                data.get("document_id"),
                "document_id",
            ),
            chunk_id=_required_text(data.get("chunk_id"), "chunk_id"),
            text=text,
            vector=_optional_vector(data.get("vector")),
            source_name=_optional_text(
                metadata_value.get("source_name"),
                "metadata.source_name",
            ),
            source_path=_optional_text(
                source_path,
                "metadata.source_path",
            ),
            page=_optional_page(metadata_value.get("page")),
            section=_optional_text(
                metadata_value.get("section"),
                "metadata.section",
            ),
            content_hash=content_hash or create_content_hash(text),
            document_updated_at=_optional_text(
                metadata_value.get("document_updated_at"),
                "metadata.document_updated_at",
            ),
            published_at=_optional_text(
                metadata_value.get("published_at"),
                "metadata.published_at",
            ),
            embedding_provider=_optional_text(
                metadata_value.get("embedding_provider"),
                "metadata.embedding_provider",
            ),
            embedding_model=_optional_text(
                metadata_value.get("embedding_model"),
                "metadata.embedding_model",
            ),
            embedding_dimension=_optional_positive_int(
                metadata_value.get("embedding_dimension"),
                "metadata.embedding_dimension",
            ),
            embedding_normalized=_optional_boolean(
                metadata_value.get("embedding_normalized"),
                "metadata.embedding_normalized",
            ),
            extra_metadata={
                str(key): value
                for key, value in metadata_value.items()
                if key not in known_metadata_names
            },
        )

    @property
    def vector_dimension(self) -> int | None:
        """벡터가 있으면 차원을 반환합니다."""

        return len(self.vector) if self.vector is not None else None

    @property
    def requires_embedding(self) -> bool:
        """임베딩 생성이 필요한 텍스트 청크인지 반환합니다."""

        return self.vector is None

    def validate(
        self,
        *,
        require_vector: bool = False,
        expected_dimension: int | None = None,
        require_source_name: bool = True,
    ) -> None:
        """현재 처리 단계에 필요한 값과 벡터 차원을 검사합니다."""

        _required_text(self.document_id, "document_id")
        _required_text(self.chunk_id, "chunk_id")
        _required_text(self.text, "text")

        if require_source_name and self.source_name is None:
            raise ValueError("metadata.source_name이 설정되지 않았습니다.")
        if require_vector and self.vector is None:
            raise ValueError("미리 계산된 vector가 필요합니다.")
        if (
            self.vector is not None
            and self.embedding_dimension is not None
            and len(self.vector) != self.embedding_dimension
        ):
            raise ValueError(
                "vector와 metadata.embedding_dimension이 일치하지 않습니다: "
                f"metadata={self.embedding_dimension}, actual={len(self.vector)}"
            )
        if expected_dimension is not None:
            if expected_dimension <= 0:
                raise ValueError("expected_dimension은 1 이상이어야 합니다.")
            if self.vector is None:
                raise ValueError("차원을 검사할 vector가 없습니다.")
            if len(self.vector) != expected_dimension:
                raise ValueError(
                    "벡터 차원이 일치하지 않습니다: "
                    f"expected={expected_dimension}, actual={len(self.vector)}"
                )

    def point_id(self) -> str:
        """동일 청크 재적재 시에도 같은 Qdrant Point ID를 반환합니다."""

        content_hash = self.content_hash or create_content_hash(self.text)
        identity = f"{self.document_id}:{self.chunk_id}:{content_hash}"
        return str(uuid.uuid5(POINT_ID_NAMESPACE, identity))

    def payload(self, *, indexed_at: str) -> dict[str, object]:
        """Qdrant Payload 후보를 만듭니다.

        ``indexed_at``은 실제 Upsert를 수행하는 시점에 주입합니다.
        """

        payload: dict[str, object] = {
            "text": self.text,
            "document_id": self.document_id,
            "chunk_id": self.chunk_id,
            "content_hash": self.content_hash
            or create_content_hash(self.text),
            "indexed_at": _required_text(indexed_at, "indexed_at"),
        }

        optional_values = {
            "source_name": self.source_name,
            "source_path": self.source_path,
            "page": self.page,
            "section": self.section,
            "document_updated_at": self.document_updated_at,
            "published_at": self.published_at,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "embedding_normalized": self.embedding_normalized,
        }
        payload.update(
            {
                key: value
                for key, value in optional_values.items()
                if value is not None
            }
        )
        payload.update(self.extra_metadata)
        return payload


def infer_vector_dimension(records: Iterable[PreparedChunk]) -> int | None:
    """벡터가 있는 레코드의 공통 차원을 확인합니다.

    모든 레코드가 텍스트 청크라면 ``None``을 반환합니다. 서로 다른 차원의
    벡터가 섞여 있으면 Qdrant 적재 전에 오류를 발생시킵니다.
    """

    dimensions = {
        record.vector_dimension
        for record in records
        if record.vector_dimension is not None
    }

    if not dimensions:
        return None
    if len(dimensions) > 1:
        dimension_list = ", ".join(str(value) for value in sorted(dimensions))
        raise ValueError(f"서로 다른 벡터 차원이 섞여 있습니다: {dimension_list}")

    return next(iter(dimensions))


def validate_unique_records(records: Iterable[PreparedChunk]) -> None:
    """한 입력 묶음 안에서 문서·청크 ID 중복을 검사합니다."""

    seen_keys: set[tuple[str, str]] = set()
    for record in records:
        key = (record.document_id, record.chunk_id)
        if key in seen_keys:
            raise ValueError(
                "중복된 document_id와 chunk_id 조합입니다: "
                f"{record.document_id}/{record.chunk_id}"
            )
        seen_keys.add(key)
