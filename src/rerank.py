"""RRF 후보를 원본 질문과 비교해 CrossEncoder(BGE)로 재정렬합니다."""
import math
from time import perf_counter
from src.const.model import create_rerank_model
from src.multi_query import multi_query
from src.rrf import reciprocal_rank_fusion


# State의 원본 질문과 후보 본문을 비교해 상위 문서를 반환합니다.
def rerank(state, top_k=5, *, model=None, timings=None, verbose=True):
    started = perf_counter()
    question = state.get("rewrite_query") or state.get("origin_query")
    documents = state.get("documents", [])
    if not isinstance(question, str) or not question.strip():
        raise ValueError("문자열 origin_query가 필요합니다.")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k는 1~100 정수여야 합니다.")
    if not isinstance(documents, list):
        raise ValueError("documents는 문서 딕셔너리 목록이어야 합니다.")
    if not documents:
        response = {"documents": [], "score" : None}
        if timings is not None:
            timings.update(model_load_seconds=0.0, processing_seconds=perf_counter() - started,
                           inference_seconds=0.0, window_count=0)
        if verbose:
            print(response)
        return response
    for doc in documents:
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError("각 후보에는 비어 있지 않은 text가 필요합니다.")

    # 평가에서는 로드된 모델을 주입할 수 있고, 일반 호출은 공통 로더를 사용합니다.
    load_started = perf_counter()
    supplied_model = model is not None
    model = create_rerank_model() if model is None else model
    load_seconds = 0.0 if supplied_model else perf_counter() - load_started
    processing_started = perf_counter()
    tokenizer = model.tokenizer
    # 분할 준비용 토큰화는 아직 모델 입력이 아니므로 원문 길이 경고를 발생시키지 않습니다.
    query_tokens = tokenizer.encode(question.strip(), add_special_tokens=False, verbose=False)
    if len(query_tokens) > 1000:
        raise ValueError("질문이 재정렬 입력 한도를 초과합니다(1000 토큰).")
    max_length = model.max_seq_length
    pairs, owners, pair_lengths = [], [], []
    for index, doc in enumerate(documents):
        # 긴 표·법령의 뒷부분도 평가하도록 본문을 겹치는 토큰 구간으로 나눕니다.
        tokens = tokenizer.encode(doc["text"], add_special_tokens=False, verbose=False)
        if not tokens:
            raise ValueError("재정렬 본문에서 토큰을 생성하지 못했습니다.")
        start = 0
        while start < len(tokens):
            size = min(400, len(tokens) - start)
            # decode 후 재토큰화 길이는 달라질 수 있으므로 질문·특수 토큰까지 실제로 계산합니다.
            while True:
                passage = tokenizer.decode(tokens[start:start + size], skip_special_tokens=True)
                length = len(tokenizer(question.strip(), passage, add_special_tokens=True,
                                       truncation=False, verbose=False)["input_ids"])
                if length <= max_length:
                    break
                if size == 1:
                    raise ValueError("질문과 본문 구간을 모델 입력 한도 안에 구성하지 못했습니다.")
                size = max(1, size - max(1, length - max_length))
            pairs.append((question.strip(), passage))
            owners.append(index)
            pair_lengths.append(length)
            if start + size >= len(tokens):
                break
            start += max(1, size - 50)  # 축소된 구간도 끝부분을 빠뜨리지 않고 겹쳐 이동합니다.
    inference_started = perf_counter()
    scores = model.predict(pairs, batch_size=1, show_progress_bar=False)
    inference_seconds = perf_counter() - inference_started
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
    top_docs = ranked[:top_k]
    response = {"documents": top_docs, "score": top_docs[0]["score"]}
    if timings is not None:
        timings.update(model_load_seconds=load_seconds,
                       processing_seconds=perf_counter() - processing_started,
                       inference_seconds=inference_seconds, window_count=len(pairs),
                       max_pair_tokens=max(pair_lengths), model_max_tokens=max_length)
    if verbose:
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
