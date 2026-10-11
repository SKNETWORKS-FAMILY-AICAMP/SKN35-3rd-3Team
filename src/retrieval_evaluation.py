"""정답 청크 ID가 있는 질문으로 검색 성능을 계산합니다.

LLM Judge와 별개인 오프라인 평가입니다. 같은 청크가 반복되면 첫 등장만 남깁니다.
정답이 없는 질문은 이 평가에서 제외하거나 별도 무응답 평가로 관리해야 합니다.
"""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter

from src.vectorstore import VectorSearchHit


def chunk_key(hit: VectorSearchHit) -> str:
    """정답 파일에서도 같은 키 형식을 사용합니다: document_id::chunk_id."""
    document = hit.metadata.get("document_id")
    chunk = hit.metadata.get("chunk_id")
    if isinstance(document, str) and document and isinstance(chunk, str) and chunk:
        return f"{document}::{chunk}"
    return hit.point_id


def evaluate_ranking(retrieved_ids: Sequence[str], relevant_ids: Sequence[str], *, k: int = 5) -> dict[str, float]:
    """질문 하나의 Recall/Precision/Hit/Reciprocal Rank를 계산합니다.

    Precision 분모는 K: 결과가 K개보다 적어도 부족한 자리를 정답으로 보지 않습니다.
    RR은 첫 정답 순위의 역수, MRR은 여러 질문의 RR 평균입니다.
    """
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 100:
        raise ValueError("k는 1 이상 100 이하 정수여야 합니다.")
    if isinstance(retrieved_ids, (str, bytes)) or isinstance(relevant_ids, (str, bytes)):
        raise TypeError("ID 목록을 전달해야 합니다.")
    if any(not isinstance(value, str) or not value.strip() for value in (*retrieved_ids, *relevant_ids)):
        raise ValueError("문서 ID는 비어 있지 않은 문자열이어야 합니다.")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("정답 문서가 없는 질문은 별도로 평가해야 합니다.")
    ranked = list(dict.fromkeys(retrieved_ids))[:k]
    matches = sum(identifier in relevant for identifier in ranked)
    rr = next((1 / rank for rank, identifier in enumerate(ranked, 1) if identifier in relevant), 0.0)
    return {"recall": matches / len(relevant), "precision": matches / k, "hit": float(matches > 0), "rr": rr}


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    relevant_ids: tuple[str, ...]
    filters: Mapping[str, object] | None = None


def compare_retrievers(searchers: Mapping[str, object], cases: Sequence[EvaluationCase], *, k: int = 5) -> dict:
    """모든 방식에 같은 질문/정답/필터/K를 적용하고 평균과 질문별 결과를 반환.

    실패를 0점으로 감추지 않습니다. 호출 오류나 불완전한 정답은 평가를 중단합니다.
    latency에는 임베딩, 검색, Multi-Query 생성 등 실제 호출 시간이 포함됩니다.
    """
    if not cases:
        raise ValueError("평가 질문이 없습니다.")
    # API를 호출하기 전에 모든 정답과 질문을 검사합니다.
    for case in cases:
        if not isinstance(case.query, str) or not case.query.strip():
            raise ValueError("평가 질문은 비어 있을 수 없습니다.")
        evaluate_ranking([], case.relevant_ids, k=k)
    report = {}
    for name, searcher in searchers.items():
        rows = []
        for case in cases:
            start = perf_counter()
            hits = searcher.search_documents(case.query, top_k=k, filters=case.filters)
            elapsed = perf_counter() - start
            ids = [chunk_key(hit) for hit in hits]
            metrics = evaluate_ranking(ids, case.relevant_ids, k=k)
            rows.append({"query": case.query, "retrieved_ids": ids, **metrics, "seconds": elapsed})
        report[name] = {
            "recall": sum(row["recall"] for row in rows) / len(rows),
            "precision": sum(row["precision"] for row in rows) / len(rows),
            "hit_rate": sum(row["hit"] for row in rows) / len(rows),
            "mrr": sum(row["rr"] for row in rows) / len(rows),
            "mean_seconds": sum(row["seconds"] for row in rows) / len(rows),
            "per_query": rows,
        }
    return {"k": k, "query_count": len(cases), "results": report}
