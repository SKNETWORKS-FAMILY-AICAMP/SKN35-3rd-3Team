"""검색된 근거와 답변에서 인용 검증을 통과한 출처를 구분합니다."""

VALID_CITATION_STATUSES = {"passed", "passed_with_missing_metadata"}


def verified_sources(result):
    if result.get("citation_validation_status") not in VALID_CITATION_STATUSES:
        return []
    return result.get("sources") or result.get("citations") or []


def _identity(source):
    if source.get("document_id") and source.get("chunk_id"):
        return source["document_id"], source["chunk_id"]
    return source.get("point_id") or (source.get("source_url"), source.get("source_id"))


def display_sources(result):
    """검증 출처를 먼저, 나머지 실제 검색 자료를 후보로 표시합니다."""
    items = [dict(source, is_cited=True) for source in verified_sources(result)]
    seen = {_identity(source) for source in items}
    contexts = {_identity(source): source for source in result.get("context_sources") or []}
    for document in result.get("documents") or []:
        if not isinstance(document, dict) or not document.get("text"):
            continue
        meta = document.get("metadata") or {}
        candidate = {
            **meta,
            "point_id": document.get("point_id"),
            "source_name": meta.get("law_name") or meta.get("source_name") or meta.get("title"),
            "article": meta.get("article_label") or meta.get("article_number"),
            "text": document["text"],
        }
        identity = _identity(candidate)
        if identity in seen:
            continue
        # Context에 포함됐다는 사실과 답변에 인용됐다는 사실은 다릅니다.
        candidate = {**contexts.get(identity, {}), **candidate, "is_cited": False}
        candidate.pop("source_id", None)
        items.append(candidate)
        seen.add(identity)
    return items
