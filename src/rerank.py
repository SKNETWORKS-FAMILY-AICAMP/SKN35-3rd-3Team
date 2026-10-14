"""RRF 후보를 원본 질문과 비교해 CrossEncoder(BGE)로 재정렬합니다."""
import math
import re
from time import perf_counter
from src.const.model import create_rerank_model
from src.multi_query import multi_query
from src.rrf import reciprocal_rank_fusion


DEFAULT_BATCH_SIZE = 8
PRIMARY_LAW_MIN_COUNT = 2


def _rerank_question(state):
    """자연어 질문에 법률 핵심어를 덧붙여 재정렬 의도를 선명하게 합니다."""
    question = state.get("rewrite_query") or state.get("origin_query")
    if not isinstance(question, str) or not question.strip():
        return question
    queries = state.get("queries") or []
    if queries and isinstance(queries[-1], str):
        legal_query = queries[-1].strip()
        if legal_query and legal_query.casefold() != question.strip().casefold():
            question = f"{question.strip()}\n검색 핵심어: {legal_query}"
    return question


def _legal_search_rank(document):
    source = document.get("metadata", {}).get("rrf_sources", {}).get("legal_primary_source", {})
    rank = source.get("rank")
    return rank if isinstance(rank, int) and rank > 0 else math.inf


def _preserve_primary_laws(ranked, top_k, question="", minimum=PRIMARY_LAW_MIN_COUNT):
    """명시된 조문과 법령 원문이 최종 근거에 포함되도록 출처 균형을 맞춥니다."""
    top_docs = list(ranked[:top_k])
    target = min(minimum, top_k)
    protected: set[str] = set()

    def key(document):
        return str(document.get("point_id") or id(document))

    def include(candidate, reason):
        candidate_key = key(candidate)
        if any(key(document) == candidate_key for document in top_docs):
            protected.add(candidate_key)
            return
        replace_at = next(
            (index for index in range(len(top_docs) - 1, -1, -1)
             if key(top_docs[index]) not in protected),
            None,
        )
        if replace_at is None:
            return
        candidate["metadata"] = {**candidate.get("metadata", {}), "selection_reason": reason}
        top_docs[replace_at] = candidate
        protected.add(candidate_key)
        top_docs.sort(key=lambda doc: -doc["score"])

    # 질문 생성기가 확실한 조문 번호를 제시했다면 각 조문의 가장 관련도 높은
    # 청크를 하나씩 남깁니다. 복합 질문에서 한 조문에 결과가 몰리는 것을 막습니다.
    article_numbers = list(dict.fromkeys(re.findall(r"제\s*(\d+)\s*조", question or "")))
    for article_number in article_numbers[:top_k]:
        candidates = [
            doc for doc in ranked
            if doc.get("metadata", {}).get("document_type") == "law"
            and str(doc.get("metadata", {}).get("article_number") or "") == article_number
        ]
        if candidates:
            include(max(candidates, key=lambda doc: doc["score"]), "referenced_article_coverage")

    current = sum(doc.get("metadata", {}).get("document_type") == "law" for doc in top_docs)
    selected_ids = {key(doc) for doc in top_docs}
    law_candidates = sorted(
        (doc for doc in ranked
         if key(doc) not in selected_ids and doc.get("metadata", {}).get("document_type") == "law"),
        key=lambda doc: (_legal_search_rank(doc), -doc["score"]),
    )
    for candidate in law_candidates[:max(0, target - current)]:
        include(candidate, "primary_law_source_coverage")
    return top_docs


# State의 원본 질문과 후보 본문을 비교해 상위 문서를 반환합니다.
def rerank(state, top_k=5, *, model=None, batch_size=DEFAULT_BATCH_SIZE,
           timings=None, verbose=True):
    started = perf_counter()
    question = _rerank_question(state)
    documents = state.get("documents", [])
    if not isinstance(question, str) or not question.strip():
        raise ValueError("문자열 origin_query가 필요합니다.")
    if type(top_k) is not int or not 1 <= top_k <= 100:
        raise ValueError("top_k는 1~100 정수여야 합니다.")
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError("batch_size는 1~128 정수여야 합니다.")
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
    scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
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
    top_docs = _preserve_primary_laws(ranked, top_k, question)
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
