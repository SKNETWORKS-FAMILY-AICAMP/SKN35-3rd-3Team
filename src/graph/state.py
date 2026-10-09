# State 클래스 정의
from typing import TypedDict

class State(TypedDict):
    origin_query : str # 원본
    rewrite_query: str # 재작성
    documents : list[str] # 가져온 문서
    is_valid: bool # 0.8보다 높은지 체크
    precision_at_k : float # Precision@K 평가용