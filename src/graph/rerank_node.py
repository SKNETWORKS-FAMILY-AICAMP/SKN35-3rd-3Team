from src.graph.state import State
# 노드 입니다.
# Precision@K 로 평가 진행
# state에 따라서 노드의 상태를 정의합니다.
# rerank 평가 후 다시 query_rewrite

def rerank_node(state : State):
    # rerank.py에서 정의한 함수의 리턴값만 받아서 다룰 예정
    documents = "ddd"
    precision = 0.5
    return {"documents": documents, "precision_at_k": precision,}