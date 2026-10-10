"""로컬 MCP에서 사용할 교체 가능한 임베딩 구현.

기본값은 OpenAI Embeddings API입니다. 개인 API 키 값은 서로 달라도
같은 모델이면 같은 Qdrant 컬렉션을 사용할 수 있습니다. 다른 모델을 쓰면
벡터 공간이 달라지므로 반드시 별도 컬렉션과 재임베딩이 필요합니다.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast

from src.const.qdrant_config import EmbeddingSettings
from src.const.vector_data_contract import normalize_vector
from src.const.vector_db_defaults import NVIDIA_EMBEDDING_BASE_URL


class EmbeddingResponseItemLike(Protocol):
    index: int
    embedding: Sequence[float]


class EmbeddingResponseLike(Protocol):
    data: Sequence[EmbeddingResponseItemLike]


class EmbeddingEndpointLike(Protocol):
    def create(self, **kwargs: object) -> EmbeddingResponseLike: ...


class EmbeddingClientLike(Protocol):
    embeddings: EmbeddingEndpointLike


class SentenceTransformerModelLike(Protocol):
    """로컬 Sentence Transformers 모델의 최소 실행 인터페이스."""

    def encode(self, sentences: Sequence[str], **kwargs: object) -> object: ...


@dataclass
class OpenAICompatibleEmbedder:
    """OpenAI 호환 Embeddings API를 공통 인터페이스로 연결합니다."""

    client: EmbeddingClientLike
    model: str
    dimension: int
    query_extra_body: dict[str, object] | None = None
    document_extra_body: dict[str, object] | None = None

    def _embed(
        self,
        texts: Sequence[str],
        *,
        extra_body: dict[str, object] | None,
    ) -> list[list[float]]:
        normalized_texts = [text.strip() for text in texts]
        if not normalized_texts or any(not text for text in normalized_texts):
            raise ValueError("임베딩할 텍스트는 비어 있을 수 없습니다.")

        request: dict[str, object] = {
            "model": self.model,
            "input": normalized_texts,
        }
        if extra_body:
            request["extra_body"] = dict(extra_body)

        response = self.client.embeddings.create(**request)
        ordered_items = sorted(response.data, key=lambda item: item.index)
        vectors = [list(map(float, item.embedding)) for item in ordered_items]
        if len(vectors) != len(normalized_texts):
            raise RuntimeError("임베딩 응답 개수가 입력 개수와 다릅니다.")
        for vector in vectors:
            if len(vector) != self.dimension:
                raise ValueError(
                    "임베딩 응답 차원이 설정과 다릅니다: "
                    f"expected={self.dimension}, actual={len(vector)}"
                )
        return vectors

    def embed_query(self, text: str) -> Sequence[float]:
        """검색 질문 한 건을 ``query`` 벡터로 변환합니다."""

        return self._embed(
            [text],
            extra_body=self.query_extra_body,
        )[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """문서 청크를 ``passage`` 벡터로 변환합니다."""

        return self._embed(
            texts,
            extra_body=self.document_extra_body,
        )


class NvidiaEmbedder(OpenAICompatibleEmbedder):
    """NVIDIA NIM의 passage/query 입력 유형을 적용한 임베더."""

    def __init__(
        self,
        client: EmbeddingClientLike,
        model: str,
        dimension: int,
    ) -> None:
        super().__init__(
            client=client,
            model=model,
            dimension=dimension,
            query_extra_body={"input_type": "query", "truncate": "END"},
            document_extra_body={
                "input_type": "passage",
                "truncate": "END",
            },
        )


@dataclass
class LocalHuggingFaceEmbedder:
    """개인 GPU 또는 CPU에서 Sentence Transformers 모델을 실행합니다."""

    model: SentenceTransformerModelLike
    model_name: str
    dimension: int
    batch_size: int = 32
    normalize_embeddings: bool = True
    query_prefix: str = ""
    document_prefix: str = ""

    def _embed(
        self,
        texts: Sequence[str],
        *,
        prefix: str,
    ) -> list[list[float]]:
        normalized_texts = [text.strip() for text in texts]
        if not normalized_texts or any(not text for text in normalized_texts):
            raise ValueError("임베딩할 텍스트는 비어 있을 수 없습니다.")

        encoded = self.model.encode(
            [f"{prefix}{text}" for text in normalized_texts],
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        to_list = getattr(encoded, "tolist", None)
        raw_vectors = to_list() if callable(to_list) else encoded
        if not isinstance(raw_vectors, Sequence) or isinstance(
            raw_vectors,
            (str, bytes),
        ):
            raise TypeError("로컬 임베딩 결과는 벡터 배열이어야 합니다.")
        if len(raw_vectors) != len(normalized_texts):
            raise RuntimeError("임베딩 응답 개수가 입력 개수와 다릅니다.")

        vectors: list[list[float]] = []
        for raw_vector in raw_vectors:
            vector = list(normalize_vector(raw_vector))
            if len(vector) != self.dimension:
                raise ValueError(
                    "로컬 임베딩 결과 차원이 설정과 다릅니다: "
                    f"expected={self.dimension}, actual={len(vector)}"
                )
            vectors.append(vector)
        return vectors

    def embed_query(self, text: str) -> Sequence[float]:
        """검색 질문 한 건을 로컬 벡터로 변환합니다."""

        return self._embed([text], prefix=self.query_prefix)[0]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """문서 청크를 로컬 벡터로 변환합니다."""

        return self._embed(texts, prefix=self.document_prefix)


def _first_api_key(*environment_names: str) -> str:
    """공통 키를 우선하고 제공자별 기존 키를 대체값으로 읽습니다."""

    for environment_name in environment_names:
        value = os.getenv(environment_name, "").strip()
        if value:
            return value
    expected_names = " 또는 ".join(environment_names)
    raise ValueError(f"{expected_names}가 설정되지 않았습니다.")


def _create_openai_client(
    *,
    api_key: str,
    base_url: str | None,
) -> EmbeddingClientLike:
    """SDK를 지연 import해 OpenAI 호환 Client를 만듭니다."""

    try:
        from openai import OpenAI
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "임베딩 API 호출에는 openai 설치가 필요합니다. "
            "프로젝트 루트에서 `uv sync`를 실행하세요."
        ) from error

    client_options: dict[str, object] = {"api_key": api_key}
    if base_url:
        client_options["base_url"] = base_url
    return cast(EmbeddingClientLike, OpenAI(**client_options))


def _positive_integer_env(name: str, default: int) -> int:
    """선택 환경변수를 양의 정수로 읽습니다."""

    raw_value = os.getenv(name, "").strip()
    if not raw_value:
        return default
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name}은 정수여야 합니다.") from error
    if value < 1:
        raise ValueError(f"{name}은 1 이상이어야 합니다.")
    return value


def _create_local_huggingface_embedder(
    settings: EmbeddingSettings,
) -> LocalHuggingFaceEmbedder:
    """선택 설치된 Sentence Transformers 모델을 지연 생성합니다."""

    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "개인 GPU 임베딩에는 sentence-transformers 설치가 필요합니다. "
            "`uv pip install sentence-transformers` 실행 후 다시 시도하세요."
        ) from error

    assert settings.model is not None
    assert settings.dimension is not None

    device = os.getenv("EMBEDDING_DEVICE", "").strip()
    model_options: dict[str, object] = {}
    if device:
        model_options["device"] = device

    model = SentenceTransformer(settings.model, **model_options)
    return LocalHuggingFaceEmbedder(
        model=cast(SentenceTransformerModelLike, model),
        model_name=settings.model,
        dimension=settings.dimension,
        batch_size=_positive_integer_env("EMBEDDING_BATCH_SIZE", 32),
        normalize_embeddings=settings.normalized is not False,
        query_prefix=os.getenv("EMBEDDING_QUERY_PREFIX", ""),
        document_prefix=os.getenv("EMBEDDING_DOCUMENT_PREFIX", ""),
    )


def create_embedding_provider(
    settings: EmbeddingSettings,
) -> OpenAICompatibleEmbedder:
    """환경 설정에 맞는 임베딩 제공자를 생성합니다.

    지원 값은 ``openai``, ``nvidia``, ``openai_compatible``,
    ``local_huggingface``입니다.
    API 키는 공통 ``EMBEDDING_API_KEY``를 우선하며 기존 제공자별 키도
    지원합니다. 비밀값은 객체 외부로 출력하지 않습니다.
    """

    settings.validate_for_vector_work()
    assert settings.provider is not None
    assert settings.model is not None
    assert settings.dimension is not None

    provider = settings.provider.strip().lower().replace("-", "_")
    configured_base_url = os.getenv("EMBEDDING_BASE_URL", "").strip() or None

    if provider == "nvidia":
        api_key = _first_api_key("EMBEDDING_API_KEY", "NVIDIA_API_KEY")
        client = _create_openai_client(
            api_key=api_key,
            base_url=configured_base_url or NVIDIA_EMBEDDING_BASE_URL,
        )
        return NvidiaEmbedder(
            client=client,
            model=settings.model,
            dimension=settings.dimension,
        )

    if provider == "openai":
        api_key = _first_api_key("EMBEDDING_API_KEY", "OPENAI_API_KEY")
        client = _create_openai_client(
            api_key=api_key,
            base_url=configured_base_url,
        )
        return OpenAICompatibleEmbedder(
            client=client,
            model=settings.model,
            dimension=settings.dimension,
        )

    if provider == "openai_compatible":
        api_key = _first_api_key("EMBEDDING_API_KEY")
        if configured_base_url is None:
            raise ValueError(
                "openai_compatible에는 EMBEDDING_BASE_URL이 필요합니다."
            )
        client = _create_openai_client(
            api_key=api_key,
            base_url=configured_base_url,
        )
        return OpenAICompatibleEmbedder(
            client=client,
            model=settings.model,
            dimension=settings.dimension,
        )

    if provider in {"local_huggingface", "sentence_transformers"}:
        return _create_local_huggingface_embedder(settings)

    raise ValueError(
        "지원하지 않는 EMBEDDING_PROVIDER입니다. "
        "openai, nvidia, openai_compatible, local_huggingface 중 "
        "하나를 사용하세요."
    )


def create_nvidia_embedder(settings: EmbeddingSettings) -> NvidiaEmbedder:
    """기존 NVIDIA 전용 호출부와 호환되는 팩토리."""

    if settings.provider is None or settings.provider.lower() != "nvidia":
        raise ValueError("EMBEDDING_PROVIDER=nvidia가 필요합니다.")
    return cast(NvidiaEmbedder, create_embedding_provider(settings))


def embedder(text: str) -> Sequence[float]:
    """기존 호출부와 호환되는 질문 임베딩 편의 함수."""

    from src.const.env_loader import load_project_env
    from src.const.qdrant_config import load_qdrant_settings

    load_project_env()
    settings = load_qdrant_settings()
    return create_embedding_provider(settings.embedding).embed_query(text)
