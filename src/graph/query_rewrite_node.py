from src.graph.state import State
from src.query_rewriter import query_rewriter
# 노드 입니다.

# state에 따라서 노드의 상태를 정의합니다.

def query_rewrite_node(state:State):
    # query_rewrite.py에서 정의한 함수의 리턴값만 받아서 다룰 예정
    query = state["origin_query"]

    response = query_rewriter(query)

    print(f"재작성 결과 : {response}")

    return {'rewrite_query': response}