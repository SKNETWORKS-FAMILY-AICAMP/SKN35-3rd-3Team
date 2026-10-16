"""팀 공용 설정으로 Qdrant 검색 계층을 한 번에 조립합니다."""

from __future__ import annotations

from pathlib import Path

from src.const.qdrant_config import load_qdrant_settings
from src.embedder import create_embedding_provider
from src.rag.retriever import DocumentRetriever
from src.vector_db.qdrant_store import QdrantVectorStore
from src.vector_db.sparse_embedder import create_sparse_embedding_provider


def create_team_retriever(
    env_path: str | Path | None = None,
    *,
    validate_collection_on_startup: bool = True,
) -> DocumentRetriever:
    """개인 키와 팀 공용 규격으로 바로 사용할 검색기를 만듭니다.

    컬렉션을 변경하지 않습니다. 기본값은 시작 시 실제 컬렉션 규격을 한 번
    확인합니다. MCP처럼 서버 자체는 먼저 시작해야 하는 호출부에서는 검증을
    지연하고 설정에 확정된 벡터 차원을 검색 요청 검증에 사용합니다.
    """

    settings = load_qdrant_settings(env_path)
    settings.validate_for_vector_work()

    embedder = create_embedding_provider(settings.embedding)
    vector_store = QdrantVectorStore.from_settings(settings)
    expected_dimension = settings.embedding.dimension
    if validate_collection_on_startup:
        collection_info = vector_store.validate_collection_contract()
        expected_dimension = collection_info.vector_size

    assert expected_dimension is not None
    sparse_embedder = None
    collection_settings = getattr(settings, "collection", None)
    if (
        collection_settings is not None
        and collection_settings.search_mode == "hybrid"
    ):
        assert collection_settings.sparse_model is not None
        sparse_embedder = create_sparse_embedding_provider(
            collection_settings.sparse_model
        )

    return DocumentRetriever(
        embedder=embedder,
        vector_store=vector_store,
        expected_dimension=expected_dimension,
        sparse_embedder=sparse_embedder,
    )
