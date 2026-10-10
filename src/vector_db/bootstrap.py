"""팀 공용 설정으로 Qdrant 검색 계층을 한 번에 조립합니다."""

from __future__ import annotations

from pathlib import Path

from src.const.env_loader import load_project_env
from src.const.qdrant_config import load_qdrant_settings
from src.embedder import create_embedding_provider
from src.retriever import DocumentRetriever
from src.vector_db.qdrant_store import QdrantVectorStore


def create_team_retriever(
    env_path: str | Path | None = None,
) -> DocumentRetriever:
    """개인 키와 팀 공용 규격으로 바로 사용할 검색기를 만듭니다.

    컬렉션을 변경하지 않습니다. 시작 시 실제 컬렉션 규격을 한 번 확인하고,
    검색할 때는 확인한 벡터 차원을 재사용합니다.
    """

    load_project_env(env_path)
    settings = load_qdrant_settings()
    settings.validate_for_vector_work()

    embedder = create_embedding_provider(settings.embedding)
    vector_store = QdrantVectorStore.from_settings(settings)
    collection_info = vector_store.validate_collection_contract()
    return DocumentRetriever(
        embedder=embedder,
        vector_store=vector_store,
        expected_dimension=collection_info.vector_size,
    )
