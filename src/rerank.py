"""RRF 후보를 원본 질문과 비교해 CrossEncoder로 재정렬합니다."""
import math
from src.const.model import create_rerank_model
from src.multi_query import multi_query
from src.rrf import reciprocal_rank_fusion


# State의 원본 질문과 후보 본문을 비교해 상위 문서를 반환합니다.
def rerank(state, top_k=5):
    question = state.get("origin_query")
    documents = state.get("documents", [])
    if not isinstance(question, str) or not question.strip():
        raise ValueError("문자열 origin_query가 필요합니다.")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k는 1~100 정수여야 합니다.")
    if not isinstance(documents, list):
        raise ValueError("documents는 문서 딕셔너리 목록이어야 합니다.")
    if not documents:
        response = {"documents": []}
        print(response)
        return response
    for doc in documents:
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError("각 후보에는 비어 있지 않은 text가 필요합니다.")

    # 모델은 model.py에서 생성·재사용하며, 질문과 본문을 함께 읽습니다.
    model = create_rerank_model()
    tokenizer = model.tokenizer
    query_tokens = tokenizer.encode(question.strip(), add_special_tokens=False)
    if len(query_tokens) > 100:
        raise ValueError("질문이 재정렬 입력 한도를 초과합니다(100 토큰).")
    pairs, owners = [], []
    for index, doc in enumerate(documents):
        # 긴 표·법령의 뒷부분도 평가하도록 본문을 겹치는 토큰 구간으로 나눕니다.
        tokens = tokenizer.encode(doc["text"], add_special_tokens=False)
        for start in range(0, len(tokens), 350):
            passage = tokenizer.decode(tokens[start:start + 400], skip_special_tokens=True)
            pairs.append((question.strip(), passage))
            owners.append(index)
            if start + 400 >= len(tokens):
                break
    scores = model.predict(pairs, batch_size=1, show_progress_bar=False)
    if len(scores) != len(pairs):
        raise ValueError("재정렬 점수와 입력 구간 개수가 다릅니다.")
    best = [float("-inf")] * len(documents)
    for index, score in zip(owners, scores):
        score = float(score)
        if not math.isfinite(score):
            raise ValueError("재정렬 모델이 유한하지 않은 점수를 반환했습니다.")
        # 청크의 가장 관련성 높은 구간 점수를 사용합니다. 정답 확률이 아닙니다.
        best[index] = max(best[index], score)

    ranked = []
    for index, doc in enumerate(documents):
        # 원본 본문과 출처를 유지하고, 이전 순위와 점수를 기록합니다.
        metadata = {**doc.get("metadata", {}), "retrieval_method": "rerank",
                    "before_rerank": {"rank": index + 1, "score": doc.get("score")},
                    "rerank_score_method": "max_window"}
        ranked.append({**doc, "score": best[index], "metadata": metadata})
    ranked.sort(key=lambda doc: -doc["score"])  # 동점은 RRF 순서를 유지합니다.
    response = {"documents": ranked[:top_k]}
    print(response)
    return response


if __name__ == "__main__":
    # 실제 멀티쿼리·Qdrant·RRF 후보로 재정렬 전후를 비교합니다.
    question = input("질문을 입력하세요: ")
    results = multi_query(question)
    hits = reciprocal_rank_fusion(results["rankings"], top_k=20)
    state = {"origin_query": question, "documents": [hit.to_document() for hit in hits]}
    response = rerank(state)
    for label, docs in [("재정렬 전", state["documents"][:5]),
                        ("재정렬 후", response["documents"])]:
        print(f"\n=== {label} ===")
        for index, doc in enumerate(docs, 1):
            print(f"[{index}] id={doc['point_id']}, score={doc['score']:.4f}\n{doc['text'][:200]}\n")
