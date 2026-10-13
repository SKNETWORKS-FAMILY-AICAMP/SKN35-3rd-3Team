"""질문 생성과 질문별 검색. 통합은 다음 RRF 노드에서 수행합니다."""
import re
from src.prompt.multi_query_prompt import MULTI_QUERY_PROMPT
from types import SimpleNamespace
from src.rrf import reciprocal_rank_fusion
from src.vector_db.bootstrap import create_team_retriever
from src.const.model import create_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


# 모델이 만든 여러 줄 문자열을 실제 검색 질의 리스트로 바꿉니다.
def parse_queries(text, question, max_queries=3):
    # 입력 질문을 첫 번째 질의로 남기고, 확장 표현을 뒤에 추가합니다.
    queries = [question.strip()]
    seen = {" ".join(queries[0].split()).casefold()}
    for line in text.splitlines():
        # 불릿과 번호를 제거하고, 빈 표현과 중복 표현은 제외합니다.
        query = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
        key = " ".join(query.split()).casefold()
        if query and key not in seen and len(queries) < max_queries:
            queries.append(query)
            seen.add(key)
    response = queries
    #print(response)
    return response


# 공통 로더로 프로젝트의 .env를 읽고 실제 GPT-4o-mini를 연결합니다.
def create_multi_query_llm():
    return create_llm()


# 입력 질문에서 추가 검색 표현을 만들고, 원본을 포함한 질의 리스트를 반환합니다.
def create_queries(question, llm=None, max_queries=3):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("질문은 비어 있지 않은 문자열이어야 합니다.")
    if type(max_queries) is not int or not 1 <= max_queries <= 5:
        raise ValueError("max_queries는 1~5 정수여야 합니다.")
    if max_queries == 1:
        # 원본만 검색할 때는 모델을 호출하지 않습니다.
        response = [question.strip()]
    else:
        # 프롬프트 → 모델 → 문자열 흐름을 하나의 체인으로 묶습니다.
        chain = PromptTemplate.from_template(MULTI_QUERY_PROMPT) | (llm if llm is not None else create_multi_query_llm()) | StrOutputParser()
        text = chain.invoke({"question": question, "count": max_queries - 1})
        # 모델의 문자열 출력을 검색에 사용할 질의 리스트로 바꿉니다.
        response = parse_queries(text, question, max_queries)
    print("\n[검색 질문]")

    for i, query in enumerate(response, 1):
        print(f"{i}. {query}")

    return response


# 원본과 확장 질문을 각각 검색하고, 질문별 결과를 다음 RRF 노드로 전달합니다.
def multi_query(question, llm=None, searcher=None, *, candidate_k=20, filters=None):
    """질문 확장 → 원격 Qdrant Dense/Sparse 검색 → 질문별 결과 반환."""
    if type(candidate_k) is not int or not 1 <= candidate_k <= 100:
        raise ValueError("candidate_k는 1~100 정수여야 합니다.")
    # 공통 설정으로 두 검색을 Qdrant 서버에서 실행합니다.
    owns_searcher = searcher is None
    if owns_searcher:
        searcher = create_team_retriever()
    try:
        queries = create_queries(question, llm)
        rankings = {}
        for index, query in enumerate(queries):
            # 각 질문으로 원격 Qdrant의 Dense와 Sparse 검색을 실행합니다.
            # 질문별 Dense/Sparse 결과는 Qdrant 내부 RRF로 통합됩니다.
            # 다음 그래프 RRF 노드는 여러 질문의 결과를 통합합니다.
            rankings[f"query_{index}"] = list(searcher.search_documents(
                query, top_k=candidate_k, filters=filters,
            ))
    finally:
        # 이 함수에서 만든 검색 연결은 작업이 끝나면 닫습니다.
        if owns_searcher:
            searcher.vector_store.client.close()
    response = {"queries": queries, "rankings": rankings}
    # 원본(query_0)과 확장 질문의 결과를 구분해서 출력합니다.
    #print(response)
    return response


# 기존 평가·재정렬 코드가 함수 기반 멀티쿼리를 호출할 수 있도록 연결합니다.
def create_multi_query_searcher(searcher, llm=None):
    """평가·재정렬 호출부에서 함수 구현을 사용하는 연결 객체입니다."""

    def search_documents(query, *, top_k=5, score_threshold=None, filters=None):
        result = multi_query(query, llm, searcher, candidate_k=max(20, top_k), filters=filters)
        # 그래프 밖의 평가 경로에서는 질문별 결과를 RRF로 합치고 중복 청크를 제거합니다.
        hits = reciprocal_rank_fusion(result["rankings"], top_k=top_k)
        # 통합 점수의 하한이 지정되면 기준을 만족하는 결과만 남깁니다.
        response = tuple(hit for hit in hits if score_threshold is None or hit.score >= score_threshold)
        #print(response)
        return response

    return SimpleNamespace(search_documents=search_documents)

if __name__ == "__main__":
    question = input("질문을 입력하세요: ")
    multi_query(question)
