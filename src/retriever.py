"""Vector DB 검색 서비스.

LangGraph 상태나 MCP SDK에 의존하지 않습니다. 실제 임베딩 모델과 Vector Store
구현체를 주입받아 검색 한 번을 수행합니다.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from src.rrf import reciprocal_rank_fusion
from src.vectorstore import (
    VectorCollectionInfo,
    VectorDocument,
    VectorSearchHit,
    VectorSearchRequest,
    VectorStore,
)


# --------------------------------------------------------------------------
# 인터페이스(Protocol)
# --------------------------------------------------------------------------
@runtime_checkable
class QueryEmbedder(Protocol):
    """문서와 같은 모델로 질문 벡터를 만드는 인터페이스."""

    def embed_query(self, text: str) -> Sequence[float]:
        """질문 문자열을 벡터로 변환합니다."""


class SparseQueryVector(Protocol):
    """검색에 필요한 Sparse Vector 최소 구조."""

    indices: Sequence[int]
    values: Sequence[float]


class SparseQueryEmbedder(Protocol):
    """질문을 Sparse Vector로 만드는 최소 인터페이스."""

    def embed_query(self, text: str) -> SparseQueryVector:
        """질문 문자열을 Sparse Vector로 변환합니다."""


@runtime_checkable
class SearchBackend(Protocol):
    """MCP Tool 업무 로직이 의존하는 검색 인터페이스."""

    def health_check(self) -> bool:
        """하위 Vector DB 상태를 반환합니다."""

    def get_collection_info(self) -> VectorCollectionInfo:
        """안전한 컬렉션 정보를 반환합니다."""

    def search_documents(
        self,
        query: str,
        *,
        top_k: int,
        score_threshold: float | None,
        filters: Mapping[str, object],
    ) -> Sequence[VectorSearchHit]:
        """질문을 임베딩하고 Vector DB 검색 한 번을 수행합니다."""

    def lookup_document(
        self,
        document_id: str,
        *,
        limit: int,
    ) -> Sequence[VectorDocument]:
        """질문 임베딩 없이 문서 ID가 같은 청크를 정확 조회합니다."""


# --------------------------------------------------------------------------
# 공통 검증 헬퍼 (BM25Retriever / HybridRetriever에서 함께 사용)
# --------------------------------------------------------------------------
def _validate_query(query: object) -> str:
    """질문 타입과 공백 여부를 검사하고, 앞뒤 공백을 제거한 문자열을 반환합니다."""

    if not isinstance(query, str):
        raise TypeError("query는 문자열이어야 합니다.")
    normalized = query.strip()
    if not normalized:
        raise ValueError("query는 비어 있을 수 없습니다.")
    return normalized


def _validate_filters(filters: Mapping[str, object] | None) -> dict[str, object]:
    """메타데이터 정확 일치 필터를 검증하고 복사본을 반환합니다."""

    if filters is not None and not isinstance(filters, Mapping):
        raise TypeError("filters는 매핑 또는 None이어야 합니다.")
    conditions = dict(filters or {})
    for key, value in conditions.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("필터 키는 비어 있지 않은 문자열이어야 합니다.")
        if not isinstance(value, (str, int, bool)):
            raise TypeError("필터 값은 문자열·정수·불리언만 지원합니다.")
    return conditions


# --------------------------------------------------------------------------
# Dense 검색
# --------------------------------------------------------------------------
@dataclass
class DocumentRetriever:
    """임베딩과 Vector Store를 연결하는 검색 서비스.

    embedder는 질문을 숫자 벡터로, vector_store는 그 벡터를 검색 결과로
    바꿉니다. 객체를 외부에서 받으므로 OpenAI/NVIDIA 설정을 고정하지 않습니다.
    """

    embedder: QueryEmbedder
    vector_store: VectorStore
    expected_dimension: int | None = None
    # [택1-A] Qdrant 내부 Sparse 검색(SEARCH_MODE=hybrid, 팀 기본값). 사용 중.
    # 메모리 BM25(HybridRetriever)와 동시에 쓰면 키워드 검색이 이중 반영됩니다.
    sparse_embedder: SparseQueryEmbedder | None = None

    def __post_init__(self) -> None:
        """시작 시 컬렉션 차원을 한 번 확인해 검색 요청에 재사용합니다."""

        if self.expected_dimension is None:
            collection_info = self.vector_store.get_collection_info()
            self.expected_dimension = collection_info.vector_size
        if self.expected_dimension < 1:
            raise ValueError("expected_dimension은 1 이상이어야 합니다.")

    def health_check(self) -> bool:
        return self.vector_store.health_check()

    def get_collection_info(self) -> VectorCollectionInfo:
        return self.vector_store.get_collection_info()

    def search_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> Sequence[VectorSearchHit]:
        """Dense 검색을 수행하고 DB의 순위 및 출처를 보존합니다.

        score_threshold는 검색 점수의 하한이며 Precision 점수가 아닙니다.
        모델과 컬렉션의 일치는 호출부에서 보장해야 합니다.
        """
        normalized_query = _validate_query(query)

        # [택1] 필터 검증 방식. 현재는 기존 방식(VectorSearchRequest.create에
        # 위임, 필터 값 타입은 검사하지 않음)을 사용합니다.
        # BM25/Hybrid와 같은 엄격한 규칙(문자열·정수·불리언만 허용)으로 통일하려면
        # 아래 두 줄을 해제하고, create의 filters 인자를 conditions로 바꾸세요.
        # conditions = _validate_filters(filters)

        # 1. 잘못된 옵션은 임베딩 API를 호출하기 전에 거절합니다(비용 방지).
        VectorSearchRequest.create(
            [0.0], top_k=top_k, score_threshold=score_threshold, filters=filters
        )

        dimension = self.expected_dimension
        if dimension is None:
            dimension = self.vector_store.get_collection_info().vector_size

        # 2. 저장된 문서와 같은 모델로 질문을 한 번만 임베딩합니다.
        query_vector = self.embedder.embed_query(normalized_query)
        sparse_vector = (
            self.sparse_embedder.embed_query(normalized_query)
            if self.sparse_embedder is not None
            else None
        )

        # 3. 차원(시작 시 확인한 expected_dimension)과 옵션을 검증해
        #    공용 요청 객체를 만들고, 실제 검색은 Vector Store 구현에 맡깁니다.
        #    반환된 문서 순위, 본문, 점수, 출처를 변경하지 않습니다.
        request = VectorSearchRequest.create(
            query_vector,
            top_k=top_k,
            score_threshold=score_threshold,
            filters=filters,
            expected_dimension=self.expected_dimension,
            query_text=normalized_query,
            sparse_indices=(
                sparse_vector.indices if sparse_vector is not None else None
            ),
            sparse_values=(
                sparse_vector.values if sparse_vector is not None else None
            ),
        )
        return self.vector_store.search(request)

    def lookup_document(
        self,
        document_id: str,
        *,
        limit: int = 20,
    ) -> Sequence[VectorDocument]:
        """질문 임베딩 없이 document_id가 같은 청크를 정확 조회합니다."""

        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id는 비어 있지 않은 문자열이어야 합니다.")
        return self.vector_store.lookup(
            {"document_id": document_id.strip()},
            limit=limit,
        )


# --------------------------------------------------------------------------
# LangGraph 노드
# --------------------------------------------------------------------------
def retriever(
    state: Mapping[str, object],
    *,
    searcher: SearchBackend,
    top_k: int = 5,
    score_threshold: float | None = None,
    filters: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """주입한 검색기로 재작성 질문 또는 원본 질문을 검색합니다.

    documents는 ID, 본문, 점수, 메타데이터를 포함합니다. 그래프 통합 시
    State.documents의 list[str] 선언을 구조화 문서 목록에 맞춰야 합니다.
    """
    query = state.get("rewrite_query") or state.get("origin_query")
    if not isinstance(query, str):
        raise ValueError("문자열 origin_query 또는 rewrite_query가 필요합니다.")
    hits = searcher.search_documents(
        query, top_k=top_k, score_threshold=score_threshold, filters=filters
    )
    return {"documents": [hit.to_document() for hit in hits]}


# --------------------------------------------------------------------------
# 키워드(BM25) 검색
# --------------------------------------------------------------------------
def tokenize_keywords(text: str) -> list[str]:
    """기본 키워드 분리: 유니코드 정규화 후 한글·영문·숫자 단위로 분리.

    형태소 분석기는 아닙니다. '보증금을'과 '보증금'을 다른 단어로 취급하므로
    한국어 품질 개선 시 tokenizer 인자로 형태소 분석 함수를 주입합니다.
    문서와 질문에는 반드시 같은 함수를 사용합니다.
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    return re.findall(r"[가-힣a-z0-9]+", normalized)


class BM25Retriever:
    """청크 전체를 메모리에 색인하는 키워드 검색기.

    documents에는 Qdrant와 같은 ID, 본문, 메타데이터를 가진
    VectorSearchHit 목록을 전달합니다. 입력 score는 사용하지 않습니다.
    초기화 시 색인을 만들며, 자료 변경 시 검색기를 다시 만들어야 합니다.
    API 호출이나 Qdrant 데이터 변경은 하지 않습니다.
    """

    def __init__(
        self,
        documents: Sequence[VectorSearchHit],
        *,
        tokenizer: Callable[[str], list[str]] = tokenize_keywords,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        # k1: 단어가 반복될수록 점수가 증가하지만 과도하게 커지지 않게 조절.
        # b: 긴 문서에 검색어가 많이 나오는 효과를 문서 길이로 보정.
        for name, value in (("k1", k1), ("b", b)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name}은 숫자여야 합니다.")
            if not math.isfinite(value):
                raise ValueError(f"{name}은 유한한 숫자여야 합니다.")
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("k1은 양수, b는 0 이상 1 이하여야 합니다.")

        self._tokenizer = tokenizer
        self._k1 = k1
        self._b = b
        # 외부에서 메타데이터를 바꿔도 준비한 색인이 달라지지 않게 복사합니다.
        self._documents = tuple(
            VectorSearchHit(doc.point_id, doc.score, doc.text, dict(doc.metadata))
            for doc in documents
        )
        ids = [doc.point_id for doc in self._documents]
        if len(ids) != len(set(ids)):
            raise ValueError("중복 point_id가 있습니다. 청크별 고유 ID가 필요합니다.")

        # 문서별 단어 등장 횟수(tf), 길이, 단어가 등장한 문서 수(df)를 저장.
        self._frequencies = tuple(
            Counter(tokenizer(doc.text)) for doc in self._documents
        )
        self._lengths = tuple(sum(tf.values()) for tf in self._frequencies)
        self._average_length = (
            sum(self._lengths) / len(self._documents) if self._documents else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for frequencies in self._frequencies:
            document_frequency.update(frequencies.keys())

        # 희귀한 단어일수록 높은 IDF. 항상 양수인 log(1 + ...) 형태를 사용.
        count = len(self._documents)
        self._idf = {
            term: math.log1p((count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> Sequence[VectorSearchHit]:
        """BM25 점수가 높은 순서로 최대 top_k개 청크를 반환합니다.

        score_threshold는 BM25 점수 기준입니다. Dense 점수와 범위가 다르므로
        Hybrid에서는 각각 따로 설정해야 합니다. 기본값은 제한 없음입니다.
        필터는 메타데이터의 문자열·정수·불리언 정확 일치 조건(AND)입니다.
        """
        normalized_query = _validate_query(query)
        conditions = _validate_filters(filters)
        # Dense와 동일한 top_k 및 유한한 점수 검증 규칙을 재사용합니다.
        VectorSearchRequest.create(
            [0.0], top_k=top_k, score_threshold=score_threshold
        )

        # 같은 검색어를 반복 입력해도 점수를 중복 가산하지 않습니다.
        terms = set(self._tokenizer(normalized_query))
        if not terms or not self._average_length:
            return ()

        hits: list[VectorSearchHit] = []
        for doc, frequencies, length in zip(
            self._documents, self._frequencies, self._lengths
        ):
            # 예: document_type='law'이면 판례/안내서는 검색 결과에서 제외.
            # 통계는 전체 색인 기준으로 유지하고, 결과 후보에만 필터를 적용.
            # type까지 비교하므로 True와 1은 서로 다른 값으로 취급합니다.
            if any(
                type(doc.metadata.get(key)) is not type(value)
                or doc.metadata.get(key) != value
                for key, value in conditions.items()
            ):
                continue
            score = 0.0
            length_penalty = self._k1 * (
                1 - self._b + self._b * length / self._average_length
            )
            for term in sorted(terms):
                frequency = frequencies.get(term, 0)
                if frequency:
                    score += self._idf[term] * (
                        frequency * (self._k1 + 1) / (frequency + length_penalty)
                    )
            # 무관한 문서를 top_k 개수 맞추기 위해 억지로 반환하지 않습니다.
            if score <= 0 or (score_threshold is not None and score < score_threshold):
                continue
            hits.append(
                VectorSearchHit(doc.point_id, score, doc.text, dict(doc.metadata))
            )

        # 점수 내림차순. 동점이면 ID 순서로 고정하여 비교 평가를 재현합니다.
        hits.sort(key=lambda hit: (-hit.score, hit.point_id))
        return tuple(hits[:top_k])


# --------------------------------------------------------------------------
# Hybrid 검색 (Dense + 메모리 BM25 → RRF)
# [택1-B] 아래 HybridRetriever는 Qdrant Sparse([택1-A])와 동시에 쓰면 안 되어
# 현재 주석 처리했습니다. 사용하려면:
#   1) 이 블록의 주석을 해제하고
#   2) SEARCH_MODE=dense 로 실행하며(Qdrant가 Sparse 벡터를 요구하지 않도록)
#   3) DocumentRetriever에는 sparse_embedder를 넘기지 않아야 합니다.
# (src/vector_db/search_bootstrap.py, scripts/search_evaluate.py가 이 클래스를
#  import하므로 해제 전에는 해당 스크립트가 동작하지 않습니다.)
# --------------------------------------------------------------------------
@dataclass
class HybridRetriever:
    """Dense + BM25 검색을 실행하고 RRF로 통합하는 검색기.

    Dense는 Qdrant, BM25는 같은 청크 전체의 메모리 색인을 사용합니다.
    두 검색의 데이터/ID/출처를 동기화하는 것은 검색기 조립 단계의 책임입니다.
    기존 DocumentRetriever를 변경하지 않고 내부 구성 요소로 사용합니다.
    """

    dense: DocumentRetriever
    keyword: BM25Retriever
    candidate_k: int = 20  # 각 검색의 후보 수. 최종 반환 수 top_k와 다릅니다.
    rank_constant: float = 60.0

    def health_check(self) -> bool:
        return self.dense.health_check()

    def get_collection_info(self) -> VectorCollectionInfo:
        return self.dense.get_collection_info()

    def lookup_document(
        self,
        document_id: str,
        *,
        limit: int = 20,
    ) -> Sequence[VectorDocument]:
        """정확 조회는 검색 방식과 무관하므로 Dense(Qdrant)에 위임합니다."""

        return self.dense.lookup_document(document_id, limit=limit)

    def search_documents(
        self,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
        dense_score_threshold: float | None = None,
        bm25_score_threshold: float | None = None,
    ) -> Sequence[VectorSearchHit]:
        """후보 검색 → RRF 통합 → 점수 하한 적용 → 상위 top_k 반환.

        score_threshold는 RRF 점수 하한이며 Dense/BM25 점수와 다릅니다.
        기본값 None을 권장하며, 개별 검색 점수 하한은 별도 인자로 지정합니다.
        """
        # API를 호출하기 전에 입력과 설정을 확인해 잘못된 입력으로 인한 비용을 막습니다.
        normalized_query = _validate_query(query)
        conditions = _validate_filters(filters)
        for count, threshold in (
            (top_k, score_threshold),
            (self.candidate_k, dense_score_threshold),
            (self.candidate_k, bm25_score_threshold),
        ):
            VectorSearchRequest.create([0.0], top_k=count, score_threshold=threshold)
        reciprocal_rank_fusion({}, top_k=top_k, rank_constant=self.rank_constant)

        # 각 검색에서 처음부터 top_k개만 찾으면 통합에 필요한 후보를 놓칠 수
        # 있습니다. 각각 넓게 찾고 통합 후 최종 반환 개수를 제한합니다.
        candidates = max(self.candidate_k, top_k)
        dense_hits = self.dense.search_documents(
            normalized_query,
            top_k=candidates,
            score_threshold=dense_score_threshold,
            filters=conditions,
        )
        keyword_hits = self.keyword.search_documents(
            normalized_query,
            top_k=candidates,
            score_threshold=bm25_score_threshold,
            filters=conditions,
        )
        # 정상적인 빈 결과는 허용하지만 연결/API 오류는 숨기지 않습니다.
        # 한쪽이 비면 다른 쪽 결과를 사용하고, 양쪽이 비면 빈 튜플입니다.
        fused = reciprocal_rank_fusion(
            {"dense": dense_hits, "bm25": keyword_hits},
            top_k=top_k,
            rank_constant=self.rank_constant,
        )
        return tuple(
            hit
            for hit in fused
            if score_threshold is None or hit.score >= score_threshold
        )