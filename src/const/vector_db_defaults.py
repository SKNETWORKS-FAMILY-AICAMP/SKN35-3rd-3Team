"""팀에서 공통으로 사용하는 Vector DB 규격.

비밀값은 이 파일에 두지 않습니다. API 제공자별 모델·차원·컬렉션은
한곳에서 고정해 팀원이 개인 API 키만 설정할 수 있게 합니다.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

# 팀 공용 Qdrant Cloud 주소입니다. 비밀 키는 개인 ``.env``에서만 관리합니다.
TEAM_QDRANT_URL: Final = (
    "https://31ff3cbb-4b6c-42a7-b0c1-09a7f3d59d65."
    "us-west-1-0.aws.cloud.qdrant.io"
)
TEAM_QDRANT_COLLECTION: Final = "team_documents_openai_dev"
TEAM_QDRANT_VECTOR_NAME: Final[str | None] = None
TEAM_QDRANT_DISTANCE: Final = "cosine"
TEAM_SEARCH_MODE: Final = "hybrid"
TEAM_QDRANT_SPARSE_VECTOR_NAME: Final = "bm25"
TEAM_QDRANT_SPARSE_MODEL: Final = "Qdrant/bm25"
TEAM_HYBRID_PREFETCH_LIMIT: Final = 100

TEAM_EMBEDDING_PROVIDER: Final = "openai"
TEAM_EMBEDDING_MODEL: Final = "text-embedding-3-small"
TEAM_EMBEDDING_DIMENSION: Final = 1536
TEAM_EMBEDDING_NORMALIZED: Final = True
NVIDIA_EMBEDDING_BASE_URL: Final = "https://integrate.api.nvidia.com/v1"


@dataclass(frozen=True)
class ApiEmbeddingProfile:
    """API 제공자와 전용 Qdrant 컬렉션의 팀 고정 규격."""

    provider: str
    model: str
    dimension: int
    collection: str
    normalized: bool = True


API_EMBEDDING_PROFILES: Final[Mapping[str, ApiEmbeddingProfile]] = (
    MappingProxyType(
        {
            "openai": ApiEmbeddingProfile(
                provider="openai",
                model="text-embedding-3-small",
                dimension=1536,
                collection="team_documents_openai_dev",
            ),
            "nvidia": ApiEmbeddingProfile(
                provider="nvidia",
                model="nvidia/nemotron-3-embed-1b",
                dimension=2048,
                collection="team_documents_nvidia_dev",
            ),
        }
    )
)


def get_api_embedding_profile(provider: str) -> ApiEmbeddingProfile:
    """API 제공자에 대응하는 팀 고정 임베딩 규격을 반환합니다."""

    normalized_provider = provider.strip().lower().replace("-", "_")
    try:
        return API_EMBEDDING_PROFILES[normalized_provider]
    except KeyError as error:
        supported = ", ".join(sorted(API_EMBEDDING_PROFILES))
        raise ValueError(
            f"고정 API 프로필은 다음만 지원합니다: {supported}"
        ) from error
