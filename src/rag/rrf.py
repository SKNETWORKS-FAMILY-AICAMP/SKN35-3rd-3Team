"""여러 질문의 검색 결과를 순위 기반 RRF로 통합합니다."""
import math

from src.vectorstore import VectorSearchHit

# 검색기와 RRF의 상호 참조를 피하고, 직접 실행할 때만 검색기를 가져옵니다.
if __name__ == "__main__":
    from src.vector_db.bootstrap import create_team_retriever

def print_hits(hits, width=80):
    if not hits:
        print("  (결과 없음)")
        return
    for rank, hit in enumerate(hits, 1):
        text = " ".join(hit.text.split())
        if len(text) > width:
            text = text[:width] + "..."
        print(f"    [{rank}] {text}")
    print()

# 질문별 검색 순위를 RRF 점수로 합치고, 중복 청크를 제거해 상위 결과를 반환합니다.
def reciprocal_rank_fusion(rankings, *, top_k=5, rank_constant=60.0):
    """질문별 검색 목록을 받아 상위 청크를 반환합니다. 점수는 확률이 아닙니다."""
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k는 1~100 정수여야 합니다.")
    if (isinstance(rank_constant, bool) or not isinstance(rank_constant, (int, float))
            or not math.isfinite(rank_constant) or rank_constant <= 0):
        raise ValueError("rank_constant는 유한한 양수여야 합니다.")

    scores, documents, sources = {}, {}, {}
    for name, hits in rankings.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("검색 목록 이름은 비어 있지 않은 문자열이어야 합니다.")
        seen = set()
        for rank, hit in enumerate(hits, 1):
            # 문서 ID와 청크 ID로 같은 청크를 식별합니다. 없으면 Qdrant ID를 씁니다.
            ids = (hit.metadata.get("document_id"), hit.metadata.get("chunk_id"))
            key = ("chunk", *ids) if all(isinstance(x, str) and x for x in ids) else ("point", hit.point_id)
            # 한 검색 목록에 같은 청크가 반복돼도 점수는 한 번만 더합니다.
            if key in seen:
                continue
            seen.add(key)
            if key in documents and documents[key].text != hit.text:
                raise ValueError("같은 청크 ID의 본문이 다릅니다. 색인을 확인하세요.")
            documents.setdefault(key, hit)
            # 높은 순위일수록 점수가 높고, 여러 목록에 등장하면 점수가 누적됩니다.
            scores[key] = scores.get(key, 0.0) + 1 / (rank_constant + rank)
            sources.setdefault(key, {})[name] = {"rank": rank, "score": float(hit.score)}

    # 통합 점수순으로 정렬합니다. 동점이면 ID로 순서를 고정합니다.
    ordered = sorted(scores, key=lambda key: (-scores[key], key))[:top_k]
    response = []
    for key in ordered:
        hit = documents[key]
        # 원본 본문·출처는 보존하고 통합 점수와 검색별 순위를 기록합니다.
        metadata = {**hit.metadata, "retrieval_method": "rrf", "rrf_sources": sources[key]}
        response.append(VectorSearchHit(hit.point_id, scores[key], hit.text, metadata))
    response = tuple(response)
    # print_hits(response)
    return response

# 그래프 State의 rankings를 받아 RRF를 실행하고, documents 형태로 반환합니다.
def rrf(state):
    # 그래프 State의 질문별 결과를 통합해 다음 rerank 노드에 전달합니다.
    hits = reciprocal_rank_fusion(state["rankings"], top_k=20)
    response = {"documents": [hit.to_document() for hit in hits]}
    # print_hits(response)
    return response

