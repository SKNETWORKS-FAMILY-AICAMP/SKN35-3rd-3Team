"""생성 전 검색 결과의 구조와 본문 유무를 확인합니다. LLM Judge는 수행하지 않습니다."""
import math

from src.const.config import RAG_MAX_RETRIES
from src.graph.state import State
from src.rag.multi_query import multi_query
from src.rag.rerank import rerank
from src.rag.rrf import reciprocal_rank_fusion


# 재정렬 결과를 검사하고, 생성 진행 여부 또는 빈 결과 재검색 상태를 반환합니다.
def validation(state: State) -> dict:
    question = state.get("origin_query")
    documents = state.get("documents")
    retries = state.get("retry_count", 0)
    # 데이터 연결 오류는 검색 실패로 숨기지 않고 즉시 알립니다.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("문자열 origin_query가 필요합니다.")
    if not isinstance(documents, list):
        raise ValueError("documents는 재정렬된 문서 목록이어야 합니다.")
    if type(retries) is not int or retries < 0:
        raise ValueError("retry_count는 0 이상의 정수여야 합니다.")
    for doc in documents:
        if (not isinstance(doc, dict) or not isinstance(doc.get("text"), str)
                or not doc["text"].strip() or not isinstance(doc.get("metadata"), dict)):
            raise ValueError("각 문서에는 비어 있지 않은 text와 metadata 딕셔너리가 필요합니다.")
        score = doc.get("score")
        if (isinstance(score, bool) or not isinstance(score, (int, float))
                or not math.isfinite(score)):
            raise ValueError("각 재정렬 문서에는 유한한 숫자 score가 필요합니다.")

    # 문서가 있으면 생성 단계로 보냅니다. 관련성·사실성·인용 정확성을 보증하지 않습니다.
    if documents:
        response = {"is_valid": True, "validation_reason": "생성에 사용할 문서가 존재합니다.",
                    "retry_count": retries}
    else:
        # 기존 False → query_rewrite 엣지를 사용하되 무한 재검색은 차단합니다.
        if retries >= RAG_MAX_RETRIES:
            raise RuntimeError("검색 결과가 없어 재검색 한도에 도달했습니다. 답변 생성을 중단합니다.")
        response = {"is_valid": False, "validation_reason": "검색 결과가 없어 재검색합니다.",
                    "retry_count": retries + 1}
    print(response)
    return response
