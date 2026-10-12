"""Qdrant 연결 및 컬렉션 설정.

이 모듈은 환경변수를 설정 객체로 변환하고 값의 유효성만 검사합니다.
Qdrant 연결, 컬렉션 생성, 데이터 적재는 수행하지 않습니다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import urlparse

from src.const.config import load_project_environment
from src.const.vector_db_defaults import (
    API_EMBEDDING_PROFILES,
    TEAM_EMBEDDING_DIMENSION,
    TEAM_EMBEDDING_MODEL,
    TEAM_EMBEDDING_NORMALIZED,
    TEAM_EMBEDDING_PROVIDER,
    TEAM_QDRANT_COLLECTION,
    TEAM_QDRANT_DISTANCE,
    TEAM_HYBRID_PREFETCH_LIMIT,
    TEAM_QDRANT_SPARSE_MODEL,
    TEAM_QDRANT_SPARSE_VECTOR_NAME,
    TEAM_QDRANT_URL,
    TEAM_QDRANT_VECTOR_NAME,
    TEAM_SEARCH_MODE,
    ApiEmbeddingProfile,
    get_api_embedding_profile,
)

DEFAULT_QDRANT_URL: Final = TEAM_QDRANT_URL
DEFAULT_DISTANCE: Final = TEAM_QDRANT_DISTANCE
DEFAULT_TIMEOUT_SECONDS: Final = 10.0

SUPPORTED_DISTANCES: Final = frozenset(
    {
        "cosine",
        "dot",
        "euclid",
        "manhattan",
    }
)
SUPPORTED_SEARCH_MODES: Final = frozenset({"dense", "hybrid"})

TRUE_VALUES: Final = frozenset({"1", "true", "yes", "on"})
FALSE_VALUES: Final = frozenset({"0", "false", "no", "off"})


def _optional_env(name: str) -> str | None:
    """공백 문자열을 ``None``으로 바꿔 반환합니다."""

    value = os.getenv(name)
    if value is None:
        return None

    stripped_value = value.strip()
    return stripped_value or None


def _optional_positive_int(name: str) -> int | None:
    """선택 환경변수를 양의 정수로 변환합니다."""

    value = _optional_env(name)
    if value is None:
        return None

    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ValueError(f"{name}은 정수여야 합니다.") from error

    if parsed_value <= 0:
        raise ValueError(f"{name}은 1 이상의 정수여야 합니다.")

    return parsed_value


def _positive_float(name: str, default: float) -> float:
    """환경변수를 양의 실수로 변환합니다."""

    value = _optional_env(name)
    if value is None:
        return default

    try:
        parsed_value = float(value)
    except ValueError as error:
        raise ValueError(f"{name}은 숫자여야 합니다.") from error

    if parsed_value <= 0:
        raise ValueError(f"{name}은 0보다 커야 합니다.")

    return parsed_value


def _boolean_env(name: str, default: bool) -> bool:
    """일반적인 문자열 환경변수를 불리언으로 변환합니다."""

    value = _optional_env(name)
    if value is None:
        return default

    normalized_value = value.lower()
    if normalized_value in TRUE_VALUES:
        return True
    if normalized_value in FALSE_VALUES:
        return False

    raise ValueError(
        f"{name}은 true/false, yes/no, on/off 또는 1/0이어야 합니다."
    )


def _optional_boolean_env(name: str) -> bool | None:
    """선택 불리언 환경변수를 변환합니다."""

    value = _optional_env(name)
    if value is None:
        return None

    normalized_value = value.lower()
    if normalized_value in TRUE_VALUES:
        return True
    if normalized_value in FALSE_VALUES:
        return False

    raise ValueError(
        f"{name}은 true/false, yes/no, on/off 또는 1/0이어야 합니다."
    )


def _api_profile_from_env() -> ApiEmbeddingProfile | None:
    """개인 API 키에 맞는 임베딩 기본 프로필을 선택합니다.

    OpenAI와 NVIDIA 모델명은 ``.env``에서 확인하되, 차원과 컬렉션은
    검증된 팀 프로필을 기본값으로 사용합니다. 로컬 GPU와 OpenAI 호환
    제공자는 사용자가 모든 규격을 명시합니다.
    """

    # 사용자가 직접 입력한 차원은 프로필 적용 전에도 형식을 검증합니다.
    if _optional_env("EMBEDDING_DIMENSION") is not None:
        _optional_positive_int("EMBEDDING_DIMENSION")

    provider = _optional_env("EMBEDDING_PROVIDER")
    openai_key_exists = _optional_env("OPENAI_API_KEY") is not None
    nvidia_key_exists = _optional_env("NVIDIA_API_KEY") is not None
    if provider is not None:
        normalized_provider = provider.lower().replace("-", "_")
        if normalized_provider in API_EMBEDDING_PROFILES:
            if (
                normalized_provider == "openai"
                and not openai_key_exists
                and nvidia_key_exists
            ):
                return get_api_embedding_profile("nvidia")
            if (
                normalized_provider == "nvidia"
                and not nvidia_key_exists
                and openai_key_exists
            ):
                return get_api_embedding_profile("openai")
            return get_api_embedding_profile(normalized_provider)
        return None

    if openai_key_exists and nvidia_key_exists:
        raise ValueError(
            "OPENAI_API_KEY와 NVIDIA_API_KEY가 모두 설정되어 있습니다. "
            "EMBEDDING_PROVIDER를 openai 또는 nvidia로 지정하세요."
        )
    if nvidia_key_exists:
        return get_api_embedding_profile("nvidia")
    return get_api_embedding_profile("openai")


def _api_profile_model(profile: ApiEmbeddingProfile) -> str:
    """제공자별 모델 환경변수와 공통 재정의 값을 순서대로 읽습니다."""

    provider_model_name = f"{profile.provider.upper()}_EMBEDDING_MODEL"
    return (
        _optional_env("EMBEDDING_MODEL")
        or _optional_env(provider_model_name)
        or profile.model
    )


def _validate_api_profile_overrides(profile: ApiEmbeddingProfile) -> None:
    """고정 API 프로필과 다른 모델을 기존 컬렉션에 섞지 못하게 합니다."""

    configured_model = _api_profile_model(profile)
    if configured_model == profile.model:
        return

    configured_dimension = _optional_positive_int("EMBEDDING_DIMENSION")
    configured_collection = _optional_env("QDRANT_COLLECTION")
    if configured_dimension is None or configured_collection is None:
        raise ValueError(
            "팀 기본값과 다른 EMBEDDING_MODEL을 사용하려면 "
            "EMBEDDING_DIMENSION과 QDRANT_COLLECTION을 함께 지정하세요."
        )
    if configured_collection == profile.collection:
        raise ValueError(
            "다른 임베딩 모델은 팀 기본 컬렉션과 벡터 공간을 공유할 수 "
            "없습니다. 별도 QDRANT_COLLECTION을 지정하세요."
        )


@dataclass(frozen=True)
class QdrantConnectionSettings:
    """Qdrant 서버 연결 설정."""

    url: str
    api_key: str | None
    timeout_seconds: float
    prefer_grpc: bool

    @classmethod
    def from_env(cls) -> QdrantConnectionSettings:
        """현재 환경변수에서 연결 설정을 읽습니다."""

        return cls(
            url=_optional_env("QDRANT_URL") or DEFAULT_QDRANT_URL,
            api_key=_optional_env("QDRANT_API_KEY"),
            timeout_seconds=_positive_float(
                "QDRANT_TIMEOUT_SECONDS",
                DEFAULT_TIMEOUT_SECONDS,
            ),
            prefer_grpc=_boolean_env(
                "QDRANT_PREFER_GRPC",
                default=False,
            ),
        )

    def validate(self) -> None:
        """연결 전에 URL 설정을 검사합니다."""

        parsed_url = urlparse(self.url)
        if parsed_url.scheme not in {"http", "https"}:
            raise ValueError(
                "QDRANT_URL은 http:// 또는 https://로 시작해야 합니다."
            )
        if not parsed_url.netloc:
            raise ValueError("QDRANT_URL에 호스트가 없습니다.")

    def safe_summary(self) -> dict[str, object]:
        """API 키를 노출하지 않는 확인용 정보를 반환합니다."""

        return {
            "url": self.url,
            "api_key_configured": self.api_key is not None,
            "timeout_seconds": self.timeout_seconds,
            "prefer_grpc": self.prefer_grpc,
        }


@dataclass(frozen=True)
class QdrantCollectionSettings:
    """Qdrant 컬렉션 설정.

    임베딩 모델이 확정되기 전에는 ``vector_size``가 ``None``일 수 있습니다.
    실제 컬렉션 생성 직전에 ``validate_for_creation()``을 호출합니다.
    """

    name: str | None
    vector_size: int | None
    distance: str
    vector_name: str | None
    search_mode: str = "dense"
    sparse_vector_name: str | None = None
    sparse_model: str | None = None
    hybrid_prefetch_limit: int = TEAM_HYBRID_PREFETCH_LIMIT

    @classmethod
    def from_env(
        cls,
        api_profile: ApiEmbeddingProfile | None = None,
    ) -> QdrantCollectionSettings:
        """현재 환경변수에서 컬렉션 설정을 읽습니다."""

        distance = (
            _optional_env("QDRANT_DISTANCE") or DEFAULT_DISTANCE
        ).lower()
        search_mode = (
            _optional_env("SEARCH_MODE") or TEAM_SEARCH_MODE
        ).lower()
        hybrid_enabled = search_mode == "hybrid"

        collection_name = _optional_env("QDRANT_COLLECTION") or (
            api_profile.collection
            if api_profile is not None
            else TEAM_QDRANT_COLLECTION
        )
        vector_size = _optional_positive_int("EMBEDDING_DIMENSION") or (
            api_profile.dimension
            if api_profile is not None
            else TEAM_EMBEDDING_DIMENSION
        )

        return cls(
            name=collection_name,
            # 문서와 질문 임베딩 차원, 컬렉션 벡터 차원은 같아야 합니다.
            vector_size=vector_size,
            distance=distance,
            vector_name=(
                _optional_env("QDRANT_VECTOR_NAME")
                or TEAM_QDRANT_VECTOR_NAME
            ),
            search_mode=search_mode,
            sparse_vector_name=(
                TEAM_QDRANT_SPARSE_VECTOR_NAME if hybrid_enabled else None
            ),
            sparse_model=(
                TEAM_QDRANT_SPARSE_MODEL if hybrid_enabled else None
            ),
            hybrid_prefetch_limit=TEAM_HYBRID_PREFETCH_LIMIT,
        )

    def validate_for_creation(self) -> None:
        """컬렉션 생성에 필요한 값이 확정됐는지 검사합니다."""

        if self.name is None:
            raise ValueError("QDRANT_COLLECTION이 설정되지 않았습니다.")
        if any(character.isspace() for character in self.name):
            raise ValueError("QDRANT_COLLECTION에는 공백을 사용할 수 없습니다.")
        if self.vector_size is None:
            raise ValueError("EMBEDDING_DIMENSION이 설정되지 않았습니다.")
        if self.distance not in SUPPORTED_DISTANCES:
            supported_values = ", ".join(sorted(SUPPORTED_DISTANCES))
            raise ValueError(
                "QDRANT_DISTANCE는 다음 중 하나여야 합니다: "
                f"{supported_values}"
            )
        if self.search_mode not in SUPPORTED_SEARCH_MODES:
            supported_modes = ", ".join(sorted(SUPPORTED_SEARCH_MODES))
            raise ValueError(
                "SEARCH_MODE는 다음 중 하나여야 합니다: "
                f"{supported_modes}"
            )
        if self.search_mode == "hybrid":
            if not self.sparse_vector_name:
                raise ValueError("Hybrid 검색에는 Sparse Vector 이름이 필요합니다.")
            if not self.sparse_model:
                raise ValueError("Hybrid 검색에는 Sparse 모델이 필요합니다.")
            if self.hybrid_prefetch_limit < 1:
                raise ValueError("Hybrid prefetch limit은 1 이상이어야 합니다.")

    def safe_summary(self) -> dict[str, object]:
        """컬렉션 설정 확인용 정보를 반환합니다."""

        return {
            "name": self.name,
            "vector_size": self.vector_size,
            "distance": self.distance,
            "vector_name": self.vector_name,
            "search_mode": self.search_mode,
            "sparse_vector_name": self.sparse_vector_name,
            "sparse_model": self.sparse_model,
            "hybrid_prefetch_limit": self.hybrid_prefetch_limit,
        }


@dataclass(frozen=True)
class EmbeddingSettings:
    """문서·질문 벡터의 생성 규격.

    텍스트 청크를 임베딩할 때뿐 아니라 미리 계산된 벡터를 받을 때도
    모델·차원·정규화 여부를 기록해야 같은 규격의 질문 벡터를 만들 수 있습니다.
    """

    provider: str | None
    model: str | None
    dimension: int | None
    normalized: bool | None

    @classmethod
    def from_env(
        cls,
        api_profile: ApiEmbeddingProfile | None = None,
    ) -> EmbeddingSettings:
        """현재 환경변수에서 임베딩 규격을 읽습니다."""

        if api_profile is not None:
            return cls(
                provider=api_profile.provider,
                model=_api_profile_model(api_profile),
                dimension=(
                    _optional_positive_int("EMBEDDING_DIMENSION")
                    or api_profile.dimension
                ),
                normalized=(
                    _optional_boolean_env("EMBEDDING_NORMALIZED")
                    if _optional_env("EMBEDDING_NORMALIZED") is not None
                    else api_profile.normalized
                ),
            )

        return cls(
            provider=(
                _optional_env("EMBEDDING_PROVIDER")
                or TEAM_EMBEDDING_PROVIDER
            ),
            model=_optional_env("EMBEDDING_MODEL") or TEAM_EMBEDDING_MODEL,
            dimension=(
                _optional_positive_int("EMBEDDING_DIMENSION")
                or TEAM_EMBEDDING_DIMENSION
            ),
            normalized=(
                _optional_boolean_env("EMBEDDING_NORMALIZED")
                if _optional_env("EMBEDDING_NORMALIZED") is not None
                else TEAM_EMBEDDING_NORMALIZED
            ),
        )

    def validate_for_vector_work(self) -> None:
        """실제 임베딩·적재·검색에 필요한 규격을 검사합니다."""

        missing_names: list[str] = []
        if self.provider is None:
            missing_names.append("EMBEDDING_PROVIDER")
        if self.model is None:
            missing_names.append("EMBEDDING_MODEL")
        if self.dimension is None:
            missing_names.append("EMBEDDING_DIMENSION")

        if missing_names:
            raise ValueError(
                "벡터 작업에 필요한 환경변수가 설정되지 않았습니다: "
                + ", ".join(missing_names)
            )

    def safe_summary(self) -> dict[str, object]:
        """임베딩 규격 확인용 정보를 반환합니다."""

        return {
            "provider": self.provider,
            "model": self.model,
            "dimension": self.dimension,
            "normalized": self.normalized,
        }


@dataclass(frozen=True)
class QdrantSettings:
    """Qdrant 연결 설정과 컬렉션 설정을 묶은 객체."""

    connection: QdrantConnectionSettings
    collection: QdrantCollectionSettings
    embedding: EmbeddingSettings

    @classmethod
    def from_env(cls) -> QdrantSettings:
        """환경변수에서 전체 Qdrant 설정을 읽습니다."""

        api_profile = _api_profile_from_env()
        if api_profile is not None:
            _validate_api_profile_overrides(api_profile)

        return cls(
            connection=QdrantConnectionSettings.from_env(),
            collection=QdrantCollectionSettings.from_env(api_profile),
            embedding=EmbeddingSettings.from_env(api_profile),
        )

    def validate_for_collection_creation(self) -> None:
        """실제 컬렉션 생성 요청 전에 전체 설정을 검사합니다."""

        self.connection.validate()
        self.collection.validate_for_creation()

    def validate_for_vector_work(self) -> None:
        """실제 적재·검색 요청 전에 전체 벡터 규격을 검사합니다."""

        self.validate_for_collection_creation()
        self.embedding.validate_for_vector_work()
        if self.collection.vector_size != self.embedding.dimension:
            raise ValueError(
                "컬렉션 벡터 차원과 임베딩 차원이 일치하지 않습니다."
            )

    def safe_summary(self) -> dict[str, object]:
        """보안정보를 제외한 전체 설정 정보를 반환합니다."""

        return {
            "connection": self.connection.safe_summary(),
            "collection": self.collection.safe_summary(),
            "embedding": self.embedding.safe_summary(),
        }


def load_qdrant_settings(
    env_path: str | Path | None = None,
) -> QdrantSettings:
    """공용 ``.env``에서 설정을 읽되 서버 연결은 하지 않습니다."""

    load_project_environment(env_path)
    return QdrantSettings.from_env()
