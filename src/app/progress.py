"""UI state derived from completed graph nodes, without simulated timers."""


def new_progress():
    return {"steps": ["active", "waiting", "waiting", "waiting"],
            "documents": [], "status": "running", "note": "질문과 첨부 자료를 분석하고 있어요"}


def advance_progress(progress, node, update):
    steps = progress["steps"]
    if node in {"query_rewrite", "multi_query", "rrf", "rerank"}:
        steps[:] = ["done", "active", "waiting", "waiting"]
        progress["note"] = "관련 자료를 검색하고 재정렬하고 있어요"
    if node in {"rrf", "rerank"} and "documents" in update:
        progress["documents"] = update["documents"] or []
    if node == "rerank":
        steps[:] = ["done", "done", "active", "waiting"]
        progress["note"] = "검색된 문서의 본문과 구조를 확인하고 있어요"
    if node == "validation":
        if update.get("is_valid"):
            steps[:] = ["done", "done", "done", "active"]
            progress["note"] = "검색된 근거를 바탕으로 답변을 작성하고 있어요"
        else:
            steps[:] = ["done", "active", "waiting", "waiting"]
            progress["documents"] = []
            progress["note"] = "근거가 부족해 다시 검색하고 있어요"
    if node == "general":
        steps[:] = ["done", "skipped", "skipped", "done"]
    if node == "generator":
        steps[3] = "done"


def finish_progress(progress, result):
    failed = bool(result.get("error_type")) or result.get("generation_status") in {"llm_error", "invalid_output"}
    limited = result.get("generation_status") in {"insufficient_evidence", "needs_clarification", "conflicting_evidence"}
    if failed:
        progress["status"] = "error"
        progress["note"] = "처리 중 오류가 발생했어요. 잠시 후 다시 시도해 주세요"
        if "active" not in progress["steps"]:
            progress["steps"][3] = "error"
        progress["steps"] = ["error" if step == "active" else step for step in progress["steps"]]
    elif not result.get("answer"):
        progress["status"] = "limited"
        progress["note"] = "이 요청에서 답변이 생성되지 않았어요"
        progress["steps"] = ["skipped" if step in {"active", "waiting"} else step for step in progress["steps"]]
    else:
        progress["status"] = "limited" if limited else "complete"
        progress["note"] = "추가 정보나 근거 확인이 필요해요" if limited else "답변 작성 완료"
        progress["steps"] = ["skipped" if step in {"active", "waiting"} else step for step in progress["steps"]]
