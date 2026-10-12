# State 클래스 정의
from typing import NotRequired

from langgraph.graph import MessagesState
from src.vectorstore import VectorSearchHit

class State(MessagesState):
    origin_query : str # 원본
    retry_count : int

    images: NotRequired[list[str]]        # base64 또는 URL 리스트

    rewrite_query: NotRequired[str] # 재작성
    queries : NotRequired[list[str]] # 멀티쿼리 용
    rankings: NotRequired[dict[str, list[VectorSearchHit]]]
    documents : NotRequired[list[dict]] # 가져온 문서

    is_valid: NotRequired[bool] # 0.8보다 높은지 체크
    precision_at_k : NotRequired[float] # Precision@K 평가용

    answer : str # 답변