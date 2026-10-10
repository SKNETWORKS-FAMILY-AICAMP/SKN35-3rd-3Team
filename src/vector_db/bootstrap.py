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

    이 함수 자체는 검색하거나 컬렉션을 변경하지 않습니다. 호출부는 반환된
    객체의 ``search_documents()``를 사용하면 됩니다.
    """

    load_project_env(env_path)
    settings = load_qdrant_settings()
    settings.validate_for_vector_work()

    embedder = create_embedding_provider(settings.embedding)
    vector_store = QdrantVectorStore.from_settings(settings)
    return DocumentRetriever(
        embedder=embedder,
        vector_store=vector_store,
    )
