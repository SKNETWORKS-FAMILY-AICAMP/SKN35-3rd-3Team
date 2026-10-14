from src.graph.state import State
import math

# 이전 임계값은 리랭커 점수를 확률처럼 해석해 정상 근거도 반복 검색했습니다.
# 호환용 상수는 남기되, 검증은 점수가 아니라 사용 가능한 근거의 존재로 판단합니다.
RERANK_MIN_SCORE = None

def validation_node(state: State):
    documents = state.get("documents") or []
    score = state.get("score")
    has_evidence = bool(documents) and score is not None and math.isfinite(float(score))
    if has_evidence:
        return {"is_valid": True, "validation_reason": "usable_evidence_found"}

    # 빈 검색 결과는 한 번만 다시 작성·검색하고, 두 번째에는 루프를 끝냅니다.
    # 생성기는 근거 없음 상태를 처리하므로 무한 재시도를 피할 수 있습니다.
    exhausted = state.get("retry_count", 0) >= 2
    return {
        "is_valid": exhausted,
        "validation_reason": "empty_evidence_retry_exhausted" if exhausted else "empty_evidence_retry",
    }
