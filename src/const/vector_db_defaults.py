"""팀에서 공통으로 사용하는 Vector DB 규격.

비밀값은 이 파일에 두지 않습니다. 원격 Qdrant 주소가 확정되면
``TEAM_QDRANT_URL``만 팀 공용 주소로 바꾸면 팀원은 개인 키만 설정해
같은 컬렉션과 임베딩 규격을 사용할 수 있습니다.
"""

from typing import Final


# 원격 서버 배포 전까지는 수업과 같은 로컬 Docker 주소를 사용합니다.
TEAM_QDRANT_URL: Final = "http://localhost:6333"
TEAM_QDRANT_COLLECTION: Final = "team_documents_openai_dev"
TEAM_QDRANT_VECTOR_NAME: Final[str | None] = None
TEAM_QDRANT_DISTANCE: Final = "cosine"

TEAM_EMBEDDING_PROVIDER: Final = "openai"
TEAM_EMBEDDING_MODEL: Final = "text-embedding-3-small"
TEAM_EMBEDDING_DIMENSION: Final = 1536
TEAM_EMBEDDING_NORMALIZED: Final = True
NVIDIA_EMBEDDING_BASE_URL: Final = "https://integrate.api.nvidia.com/v1"
