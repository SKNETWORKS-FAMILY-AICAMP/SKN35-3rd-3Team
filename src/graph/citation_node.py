from src.graph.state import State
from src.citation import citation
# 노드 입니다.

# state에 따라서 노드의 상태를 정의합니다.

def citation_node(state : State):
    # 현재는 인용 구조 검사만 실행합니다. Judge는 연결하지 않습니다.
    return citation(state)
