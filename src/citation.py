"""생성 답변의 인용을 실제 검색 결과와 대조합니다. 의미적 Judge는 호출하지 않습니다."""
import re
import runpy
from time import perf_counter


# 인용 번호·문서 ID·본문·메타데이터 일치를 LLM 없이 확인합니다.
def validate_citations(state: dict) -> dict:
    """Generator 결과와 검색 documents를 받아 인용·오류·구조 지표를 반환합니다."""
    answer = state.get("answer", "")
    if not isinstance(answer, str):
        raise ValueError("answer는 문자열이어야 합니다.")
    references = re.findall(r"\[(S[^\]\s]*)\]", answer)
    errors, warnings, citations = [], [], []
    docs = state.get("documents", [])
    context_docs = state.get("context_documents", [])
    mapping = {}
    for source in state.get("context_sources", []):
        number = source.get("source_id")
        if number in mapping:
            errors.append({"code": "duplicate_source_number", "source_id": number})
        mapping[number] = source
    if not docs:
        errors.append({"code": "no_search_documents"})
    if not references:
        errors.append({"code": "missing_citation"})
    for number in dict.fromkeys(references):
        source = mapping.get(number)
        if not re.fullmatch(r"S[1-9]\d*", number) or source is None:
            errors.append({"code": "unknown_citation", "source_id": number})
            continue
        if not source.get("document_id"):
            errors.append({"code": "missing_document_id", "source_id": number})
            continue
        matches = [d for d in docs if d.get("metadata", {}).get("document_id") == source["document_id"]
                   and d.get("metadata", {}).get("chunk_id") == source.get("chunk_id")
                   and (not source.get("point_id") or d.get("point_id") == source["point_id"])]
        contexts = [d for d in context_docs if d.get("metadata", {}).get("document_id") == source["document_id"]
                    and d.get("metadata", {}).get("chunk_id") == source.get("chunk_id")
                    and (not source.get("point_id") or d.get("point_id") == source["point_id"])]
        if len(matches) != 1 or len(contexts) != 1:
            errors.append({"code": "document_mapping_mismatch", "source_id": number})
            continue
        doc, context_doc = matches[0], contexts[0]
        text, excerpt = doc.get("text"), context_doc.get("text")
        if not isinstance(text, str) or not text.strip() or not isinstance(excerpt, str) or not excerpt.strip():
            errors.append({"code": "missing_document_text", "source_id": number})
            continue
        if (source.get("truncated") and not text.startswith(excerpt)) or (not source.get("truncated") and text != excerpt):
            errors.append({"code": "context_text_mismatch", "source_id": number})
            continue
        meta = doc.get("metadata", {})
        actual = {"source_name": meta.get("law_name") or meta.get("source_name") or meta.get("title"),
                  "article": meta.get("article_label") or meta.get("article_number"),
                  "article_title": meta.get("article_title"), "source_url": meta.get("source_url"),
                  "effective_date": meta.get("effective_date"), "version_id": meta.get("version_id")}
        mismatches = [field for field, value in actual.items() if source.get(field) != value]
        if mismatches:
            errors.append({"code": "metadata_mismatch", "source_id": number, "fields": mismatches})
            continue
        missing = [field for field in ["source_name", "article", "source_url"] if not actual[field]]
        if missing:
            warnings.append({"code": "missing_metadata", "source_id": number, "fields": missing})
        citations.append({**source, **actual, "text": text, "context_text": excerpt})
    # Generator의 주장 매핑을 Judge로 전달합니다. 주장 누락 여부나 정확성을 보증하지 않습니다.
    claims = state.get("claims", [])
    missing_claims = 0
    if not isinstance(claims, list) or not claims:
        errors.append({"code": "missing_claim_mapping"})
        claims = []
    for claim in claims:
        ids = claim.get("source_ids", []) if isinstance(claim, dict) else []
        if not isinstance(ids, list) or not ids:
            missing_claims += 1
            errors.append({"code": "claim_without_citation"})
        elif any(number not in references for number in ids):
            errors.append({"code": "claim_citation_mismatch"})
    valid = {c["source_id"] for c in citations}
    response = {"citations": citations, "validation_errors": errors, "validation_warnings": warnings,
                "citation_validity_rate": sum(n in valid for n in references) / len(references) if references else None,
                "invalid_citation_count": sum(n not in valid for n in references),
                "missing_citation_count": missing_claims if claims else None,
                "citation_coverage": None, "citation_support_rate": None}
    return response


# 구조 검증을 실행하고 실패 시 답변을 보류하며, 통과 시 Judge 입력을 반환합니다.
def citation(state: dict, *, verbose: bool = True) -> dict:
    """Generator State에서 답변·출처·구조 판정·Judge 입력을 반환합니다."""
    started = perf_counter()
    status = state.get("generation_status")
    if status != "generated_unverified":
        response = {"answer": state.get("answer", ""), "citations": [], "sources": [],
                    "used_document_ids": [], "citation_validation_status": "skipped",
                    "validation_errors": [{"code": status or "missing_generation_status"}],
                    "judge_status": "not_run", "judge_input": None}
        if status not in {"insufficient_evidence", "needs_clarification", "conflicting_evidence", "llm_error", "invalid_output"}:
            response.update(answer="생성 상태를 확인할 수 없어 답변을 보류합니다.", citation_validation_status="failed")
    else:
        response = validate_citations(state)
        passed = not response["validation_errors"]
        response.update(answer=state["answer"] if passed else "인용 출처를 확인할 수 없어 답변을 보류합니다.",
                        citation_validation_status=("passed_with_missing_metadata" if response["validation_warnings"] else "passed") if passed else "failed",
                        sources=response["citations"] if passed else [],
                        used_document_ids=list(dict.fromkeys(c["document_id"] for c in response["citations"])) if passed else [],
                        judge_status="not_run",
                        judge_input={"question": state.get("origin_query"), "answer": state["answer"],
                                     "claims": state.get("claims", []), "evidence": response["citations"]} if passed else None)
    response["citation_latency_ms"] = round((perf_counter() - started) * 1000, 1)
    if verbose:
        print({k: response[k] for k in ["answer", "citation_validation_status", "validation_errors", "citation_latency_ms"]})
    return response


if __name__ == "__main__":
    # 테스트 모듈을 실행합니다. --live를 지정하면 실제 Qdrant·LLM 통합 테스트를 수행합니다.
    runpy.run_module("tests.rag_citation", run_name="__main__")
