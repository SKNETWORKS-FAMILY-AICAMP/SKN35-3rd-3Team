"""실제 그래프 노드 완료 이벤트를 우측 패널의 네 단계 진행 상태로 변환한다."""

SEARCH_NODES = {"query_rewrite", "multi_query", "rrf", "rerank"}
DOCUMENT_NODES = {"rrf", "rerank"}
FAILED_STATUSES = {"llm_error", "invalid_output"}
LIMITED_STATUSES = {"insufficient_evidence", "needs_clarification", "conflicting_evidence"}


def new_progress():
    # 질문마다 새 목록을 생성해 이전 상담의 단계나 문서가 공유되지 않게 한다.
    return {
        "steps": ["active", "waiting", "waiting", "waiting"],
        "documents": [],
        "status": "running",
        "note": "질문과 첨부 자료를 분석하고 있어요",
    }


def advance_progress(progress, node, update):
    """진행 중인 객체를 갱신한다. 검색 재시도 시 이전 후보 문서는 초기화한다."""
    steps = progress["steps"]
    if node in SEARCH_NODES:
        steps[:] = ["done", "active", "waiting", "waiting"]
        progress["note"] = "관련 자료를 검색하고 재정렬하고 있어요"
    if node in DOCUMENT_NODES and "documents" in update:
        progress["documents"] = update["documents"] or []
    # rerank는 검색 완료와 근거 확인 시작을 모두 알리므로 위 조건과 독립적으로 처리한다.
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
    if node == "citation":
        status = update.get("citation_validation_status")
        steps[2] = "error" if status == "failed" else "done" if status in {"passed", "passed_with_missing_metadata"} else "skipped"


def finish_progress(progress, result):
    """오류·무응답·제한 답변을 구분하고 수행하지 않은 단계를 건너뜀으로 표시한다."""
    status = result.get("generation_status")
    failed = bool(result.get("error_type")) or status in FAILED_STATUSES or result.get("citation_validation_status") == "failed"
    limited = status in LIMITED_STATUSES
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
