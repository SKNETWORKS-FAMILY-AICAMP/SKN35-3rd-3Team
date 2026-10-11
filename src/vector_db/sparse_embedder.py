"""Hybrid 검색용 BM25 Sparse Vector 생성기.

Dense 임베딩 API와 독립적으로 로컬 FastEmbed를 사용합니다. 한국어 원문에
영어 stemming과 stopword가 적용되지 않도록 stemmer를 비활성화합니다.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SparseEmbedding:
    """Qdrant SDK와 분리된 Sparse Vector 한 건."""

    indices: tuple[int, ...]
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError("Sparse Vector의 indices와 values 길이가 다릅니다.")
        if not self.indices:
            raise ValueError("Sparse Vector는 비어 있을 수 없습니다.")
        if any(index < 0 for index in self.indices):
            raise ValueError("Sparse Vector index는 0 이상이어야 합니다.")
        if any(not math.isfinite(value) for value in self.values):
            raise ValueError("Sparse Vector value는 유한한 숫자여야 합니다.")


@runtime_checkable
class SparseEmbeddingProvider(Protocol):
    """문서와 질문을 같은 Sparse 모델로 변환하는 인터페이스."""

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 256,
    ) -> Sequence[SparseEmbedding]: ...

    def embed_query(self, text: str) -> SparseEmbedding: ...


def _to_sparse_embedding(value: object) -> SparseEmbedding:
    """FastEmbed 결과를 저장 독립 SparseEmbedding으로 변환합니다."""

    indices = getattr(value, "indices", None)
    values = getattr(value, "values", None)
    if indices is None or values is None:
        raise RuntimeError("BM25 Sparse 임베딩 응답 형식이 올바르지 않습니다.")
    return SparseEmbedding(
        indices=tuple(int(index) for index in indices),
        values=tuple(float(item) for item in values),
    )


@dataclass
class FastEmbedBm25:
    """Qdrant/bm25를 지연 로드하는 로컬 Sparse 임베더."""

    model_name: str = "Qdrant/bm25"
    _model: object | None = field(default=None, init=False, repr=False)

    def _get_model(self) -> object:
        if self._model is not None:
            return self._model
        try:
            from fastembed import SparseTextEmbedding
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Hybrid 검색에는 qdrant-client[fastembed] 설치가 필요합니다."
            ) from error

        self._model = SparseTextEmbedding(
            model_name=self.model_name,
            disable_stemmer=True,
        )
        return self._model

    def _embed(
        self,
        texts: Iterable[str],
        *,
        batch_size: int,
    ) -> tuple[SparseEmbedding, ...]:
        normalized_texts = [text.strip() for text in texts]
        if any(not text for text in normalized_texts):
            raise ValueError("Sparse 임베딩 입력은 비어 있을 수 없습니다.")
        if not normalized_texts:
            return ()

        model = self._get_model()
        embed = getattr(model, "embed")
        return tuple(
            _to_sparse_embedding(value)
            for value in embed(normalized_texts, batch_size=batch_size)
        )

    def embed_documents(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 256,
    ) -> Sequence[SparseEmbedding]:
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("Sparse batch_size는 정수여야 합니다.")
        if batch_size < 1:
            raise ValueError("Sparse batch_size는 1 이상이어야 합니다.")
        return self._embed(texts, batch_size=batch_size)

    def embed_query(self, text: str) -> SparseEmbedding:
        embeddings = self._embed([text], batch_size=1)
        return embeddings[0]


def create_sparse_embedding_provider(model_name: str) -> FastEmbedBm25:
    """팀 고정 BM25 모델의 Sparse 임베더를 만듭니다."""

    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("Sparse 모델 이름은 비어 있을 수 없습니다.")
    return FastEmbedBm25(model_name=model_name.strip())
