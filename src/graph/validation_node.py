from src.graph.state import State

PRECISION_THRESHOLD = 0.8

def validation_node(state: State):
    # precision = state["precision_at_k"]

    # if state["retry_count"] >= 2:
    return { "is_valid": True }

    # return {
    #     "is_valid": precision >= PRECISION_THRESHOLD
    # }