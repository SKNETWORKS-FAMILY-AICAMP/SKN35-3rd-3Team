from src.graph.state import State
from src.validation import validation


# 검색 결과를 검사하고, 판정과 재검색 횟수를 State에 반영합니다.
def validation_node(state: State):
    return validation(state)
