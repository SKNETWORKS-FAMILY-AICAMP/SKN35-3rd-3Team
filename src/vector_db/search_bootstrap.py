"""개인 .env와 2번 담당자의 Qdrant 계층을 Hybrid 검색에 연결합니다.

DB에는 읽기 요청만 보냅니다. 컬렉션 생성/삭제, 문서 적재는 수행하지 않습니다.
BM25에는 Dense 검색 결과 일부가 아니라 컬렉션의 청크 전체를 준비합니다.
"""
from __future__ import annotations
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from src.retriever import BM25Retriever, DocumentRetriever, HybridRetriever, tokenize_keywords
from src.vectorstore import VectorSearchHit


def load_keyword_documents(store: QdrantVectorStore, *, page_size: int = 256) -> tuple[VectorSearchHit, ...]:
    """Qdrant scroll로 벡터 없이 ID/본문/출처를 내려받습니다.

    초기화 때만 실행합니다. 데이터 적재가 바뀌면 검색기를 다시 조립해야 합니다.
    대규모 자료에는 메모리 색인 대신 별도 키워드 검색 서버를 고려해야 합니다.
    """
    if isinstance(page_size, bool) or not isinstance(page_size, int) or page_size < 1:
        raise ValueError("page_size는 양의 정수여야 합니다.")
    documents = []
    offset = None
    seen_offsets = set()
    while True:
        points, next_offset = store.client.scroll(
            collection_name=store.collection_name,
            limit=page_size, offset=offset, with_payload=True, with_vectors=False,
        )
        for point in points:
            payload = dict(point.payload or {})
            text = payload.pop("text", None)
            if not isinstance(text, str) or not text.strip():
                raise ValueError("저장된 청크에 유효한 text가 없습니다.")
            documents.append(VectorSearchHit(str(point.id), 0.0, text, payload))
        if next_offset is None:
            break
        # 서버가 같은 페이지를 반복 반환하면 끝없는 루프 대신 오류를 냅니다.
        marker = str(next_offset)
        if marker in seen_offsets:
            raise RuntimeError("Qdrant scroll이 같은 offset을 반복했습니다.")
        seen_offsets.add(marker)
        offset = next_offset
    return tuple(documents)


def create_team_hybrid_retriever(
    env_path: str | Path | None = None, *,
    candidate_k: int = 20,
    tokenizer: Callable[[str], list[str]] = tokenize_keywords,
) -> HybridRetriever:
    """설정 확인 → 서버 규격 확인 → 청크 읽기 → Dense/BM25 조립."""
    # 공용 모듈은 dev에서 공급합니다. jem에 다른 팀원의 파일을 복사하지 않습니다.
    # import만 하는 단위 테스트와 실제 서버 연결을 분리합니다.
    from src.const.model import load_project_env
    from src.const.qdrant_config import load_qdrant_settings
    from src.embedder import create_embedding_provider
    from src.vector_db.qdrant_store import QdrantVectorStore
    load_project_env(env_path)
    settings = load_qdrant_settings()
    settings.validate_for_vector_work()
    store = QdrantVectorStore.from_settings(settings)
    try:
        store.validate_collection_contract()
        documents = load_keyword_documents(store)
        if not documents:
            # 데이터가 없는데 임베딩 API를 호출하거나 평가 점수를 내지 않습니다.
            raise ValueError("컬렉션에 문서가 없습니다. 2번 담당자의 데이터 적재가 필요합니다.")
        embedding = create_embedding_provider(settings.embedding)
        # dev의 기본값이 서버 Hybrid여도 이 비교의 Dense는 Dense만 조회합니다.
        # BM25를 여기서 결합하므로 서버 Sparse까지 동시에 반영하지 않습니다.
        # 컬렉션/.env는 변경하지 않고 이 조회 객체의 설정만 복사합니다.
        store.settings = replace(settings, collection=replace(settings.collection, search_mode="dense"))
    except Exception:
        store.client.close()
        raise
    return HybridRetriever(
        dense=DocumentRetriever(embedding, store),
        keyword=BM25Retriever(documents, tokenizer=tokenizer),
        candidate_k=candidate_k,
    )


def create_team_multi_query_retriever(llm: object, env_path: str | Path | None = None):
    """사용자가 선택한 LLM과 팀 Hybrid 검색을 연결합니다.

    LLM 모델/키를 이 파일에 고정하지 않습니다. 원격 문서가 비어 있다면
    Hybrid 초기화에서 중단하므로 질문 확장 API는 호출하지 않습니다.
    """
    from src.multi_query import create_multi_query_searcher

    hybrid = create_team_hybrid_retriever(env_path)
    return create_multi_query_searcher(hybrid, llm)


def create_team_reranking_retriever(env_path=None, *, llm=None, candidate_k=20):
    from types import SimpleNamespace
    from src.vector_db.bootstrap import create_team_retriever
    from src.multi_query import create_multi_query_searcher
    from src.rerank import rerank
    base = create_team_retriever(env_path, search_mode="hybrid")
    searcher = create_multi_query_searcher(base, llm) if llm is not None else base

    def search_documents(query, *, top_k=5, score_threshold=None, filters=None):
        from src.vectorstore import VectorSearchHit
        hits = searcher.search_documents(query, top_k=max(candidate_k, top_k), filters=filters)
        result = rerank({"origin_query": query, "documents": [hit.to_document() for hit in hits]}, top_k)
        return tuple(VectorSearchHit(**doc) for doc in result["documents"]
                     if score_threshold is None or doc["score"] >= score_threshold)

    return SimpleNamespace(search_documents=search_documents, vector_store=base.vector_store)
