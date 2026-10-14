from src.graph.state import State

RERANK_MIN_SCORE = 0.8 # 테스트 후 수정 진행

def validation_node(state: State):
    score = state["score"]

    if state["retry_count"] >= 2:
        return { "is_valid": True }

    return {
        "is_valid": score >= RERANK_MIN_SCORE and score is not None
    }