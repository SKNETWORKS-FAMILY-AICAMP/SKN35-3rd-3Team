# rrf 구현

def rrf(state):
    # state에 따라서 rrf를 수행하는 기능을 구현
    return {}


# 기존 그래프용 자리 함수는 유지하고, 실제 순위 통합은 아래 함수로 제공합니다.
import math
from collections.abc import Mapping, Sequence

from src.vectorstore import VectorSearchHit


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[VectorSearchHit]],
    *,
    top_k: int = 5,
    rank_constant: float = 60.0,
) -> tuple[VectorSearchHit, ...]:
    """점수 단위가 다른 검색 목록을 순위로 통합합니다.

    입력 예: {'dense': dense_hits, 'bm25': keyword_hits}.
    각 목록은 이미 관련성이 높은 순으로 정렬되어 있어야 합니다.
    RRF 점수 = 각 목록에서 1 / (rank_constant + 순위)를 더한 값.
    순위는 1부터 시작합니다. 60은 조정 가능한 초기값이며 통과 기준이 아닙니다.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k는 정수여야 합니다.")
    if not 1 <= top_k <= 100:
        raise ValueError("top_k는 1 이상 100 이하여야 합니다.")
    if isinstance(rank_constant, bool) or not isinstance(rank_constant, (int, float)):
        raise TypeError("rank_constant는 숫자여야 합니다.")
    if not math.isfinite(rank_constant) or rank_constant <= 0:
        raise ValueError("rank_constant는 유한한 양수여야 합니다.")

    scores = {}
    originals = {}
    evidence = {}
    for source, hits in rankings.items():
        if not isinstance(source, str) or not source.strip():
            raise ValueError("검색 목록 이름은 비어 있지 않은 문자열이어야 합니다.")
        seen = set()
        for rank, hit in enumerate(hits, start=1):
            document_id = hit.metadata.get("document_id")
            chunk_id = hit.metadata.get("chunk_id")
            # 같은 원본 문서의 서로 다른 청크는 합치지 않습니다.
            # 양쪽 모두 팀의 고유 식별자를 보존해야 정확히 중복을 제거합니다.
            if isinstance(document_id, str) and document_id and isinstance(chunk_id, str) and chunk_id:
                key = ("chunk", document_id, chunk_id)
            else:
                key = ("point", hit.point_id)
            # 한 검색 목록의 중복 결과가 점수를 여러 번 올리지 않게 합니다.
            if key in seen:
                continue
            seen.add(key)
            if key in originals and originals[key].text != hit.text:
                raise ValueError("동일한 청크 ID의 본문이 다릅니다. 두 색인을 동기화하세요.")
            if key not in originals:
                originals[key] = hit
                scores[key] = 0.0
                evidence[key] = {}
            scores[key] += 1.0 / (rank_constant + rank)
            # 원점수는 통합에 쓰지 않고 디버깅/평가를 위해 함께 보존합니다.
            evidence[key][source] = {"rank": rank, "score": float(hit.score)}

    # 동점은 식별자로 정렬하여 실행할 때마다 순위가 바뀌지 않게 합니다.
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    results = []
    for key in ordered:
        original = originals[key]
        metadata = dict(original.metadata)
        metadata["retrieval_method"] = "rrf"
        metadata["rrf_sources"] = evidence[key]
        # 본문과 출처는 유지하고 score만 RRF 통합 점수로 바꿉니다.
        results.append(VectorSearchHit(original.point_id, scores[key], original.text, metadata))
    return tuple(results)
