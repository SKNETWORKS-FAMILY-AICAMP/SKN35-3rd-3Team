# State 클래스 정의
from typing import TypedDict

class State(TypedDict):
    origin_query : str
    rewrite_query: str
    documents : list[str]
    is_valid: bool
    precision_at_k : float