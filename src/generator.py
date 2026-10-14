"""재정렬된 실제 검색 근거로 답변을 생성합니다. 의미 검증은 별도 Judge의 역할입니다."""
import hashlib
import json
import os
import re
from time import perf_counter

import tiktoken
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from src.const.model import create_generator_llm, OPENAI_LLM_MODEL
from src.prompt.generator_prompt import GENERATOR_PROMPT
from src.query_rewriter import query_rewriter
from src.multi_query import multi_query
from src.rerank import rerank
from src.rrf import reciprocal_rank_fusion

import logging
logger = logging.getLogger(__name__)

def print_response(response, width=80):
    answer = " ".join(str(response["answer"]).split())
    print(f"  answer: {answer}")
    print("  sources:")
    for i, source in enumerate(response["sources"] or [], 1):
        text = " ".join(str(source).split())
        if len(text) > width:
            text = text[:width] + "…"
        print(f"    [{i}] {text}")
    print(f"  generation_status: {response['generation_status']}")
    print(f"  latency_ms: {response['latency_ms']}")
    print()

# 재정렬 순서를 유지하고, 출처·중복·토큰 한도·잘림 정보를 가진 컨텍스트를 만듭니다.
def build_context(documents: list[dict], model_name: str, token_budget: int) -> dict:
    """문서 목록을 받아 context, sources, included_documents, omissions를 반환합니다."""
    if not isinstance(documents, list) or type(token_budget) is not int or token_budget < 2:
        raise ValueError("문서 목록과 2 이상의 정수 토큰 예산이 필요합니다.")
    try:
        encoder = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoder = tiktoken.get_encoding("cl100k_base")
    sources, records, included, omissions, seen = [], [], [], [], set()
    for doc in documents:
        if not isinstance(doc, dict) or not isinstance(doc.get("text"), str) or not doc["text"].strip():
            raise ValueError("각 문서에는 비어 있지 않은 text가 필요합니다.")
        meta = doc.get("metadata", {})
        if not isinstance(meta, dict):
            raise ValueError("metadata는 딕셔너리여야 합니다.")
        identity = (meta.get("document_id"), meta.get("chunk_id"))
        key = identity if all(isinstance(x, str) and x for x in identity) else doc.get("point_id")
        key = key or hashlib.sha256(doc["text"].encode()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        source = {"source_id": f"S{len(sources) + 1}", "point_id": doc.get("point_id"),
                  "document_id": meta.get("document_id"), "chunk_id": meta.get("chunk_id"),
                  "source_name": meta.get("law_name") or meta.get("source_name") or meta.get("title"),
                  "article": meta.get("article_label") or meta.get("article_number"),
                  "article_title": meta.get("article_title"),
                  "version_id": meta.get("version_id"),
                  "rerank_score": doc.get("score"),
                  "before_rerank": meta.get("before_rerank"),
                  "source_url": meta.get("source_url"), "document_type": meta.get("document_type"),
                  "effective_date": meta.get("effective_date"), "warnings": meta.get("warnings"),
                  "required_context_unit_ids": meta.get("required_context_unit_ids"),
                  "required_context_chunk_ids": meta.get("required_context_chunk_ids"),
                  "truncated": False}
        source["missing_fields"] = [name for name in ["source_name", "article", "source_url"] if not source[name]]
        text = doc["text"]
        tokens = encoder.encode(text, disallowed_special=())
        record = {**source, "text": text}
        # 문서 헤더를 포함한 JSON 전체가 토큰 예산 안에 들어가도록 구성합니다.
        if len(encoder.encode(json.dumps(records + [record], ensure_ascii=False), disallowed_special=())) > token_budget:
            low, high = 0, len(tokens)
            while low < high:
                mid = (low + high + 1) // 2
                candidate = {**source, "truncated": True,
                             "text": encoder.decode_bytes(tokens[:mid]).decode("utf-8", errors="ignore")}
                size = len(encoder.encode(json.dumps(records + [candidate], ensure_ascii=False), disallowed_special=()))
                if size <= token_budget:
                    low = mid
                else:
                    high = mid - 1
            text = encoder.decode_bytes(tokens[:low]).decode("utf-8", errors="ignore")
            if not text.strip():
                omissions.append({"point_id": doc.get("point_id"), "reason": "token_budget"})
                continue
            source["truncated"] = True
            record = {**source, "text": text}
        sources.append(source)
        records.append(record)
        included.append({**doc, "text": text})
    context = json.dumps(records, ensure_ascii=False)
    response = {"context": context, "sources": sources, "included_documents": included,
                "omissions": omissions, "context_tokens": len(encoder.encode(context, disallowed_special=()))}
    return response


# 원본 질문과 검색 근거를 모델에 전달하고, 인용 출처의 존재를 확인해 결과를 반환합니다.
def generator(state: dict, *, llm=None, verbose: bool = True) -> dict:
    """origin_query/documents를 받아 답변·출처·상태·모델·지연 시간을 반환합니다."""
    started = perf_counter()
    question = state.get("origin_query")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("문자열 origin_query가 필요합니다.")
    # 질문이 컨텍스트 예산 밖에서 무제한으로 커지는 것을 방지합니다.
    encoder = tiktoken.encoding_for_model(OPENAI_LLM_MODEL)
    if len(encoder.encode(question, disallowed_special=())) > 2000:
        raise ValueError("원본 질문은 2000 토큰 이하로 입력하세요.")
    response = {"answer": "답변에 필요한 검색 근거를 확인하지 못했습니다.", "sources": [],
                "used_document_ids": [], "generation_status": "insufficient_evidence",
                "model_name": OPENAI_LLM_MODEL, "latency_ms": 0.0, "token_usage": {},
                "estimated_cost_usd": None,
                "error_type": None, "output_validation_error": None, "rejected_output": None,
                "claims": [], "source_validation_passed": False, "judge_status": "not_run"}
    documents = state.get("documents", [])
    if not isinstance(documents, list):
        raise ValueError("documents는 문서 목록이어야 합니다.")
    if documents:
        prepared = build_context(documents, OPENAI_LLM_MODEL, int(os.environ["GENERATOR_CONTEXT_TOKENS"]))
        response.update(context_tokens=prepared["context_tokens"], context_sources=prepared["sources"],
                        context_documents=prepared["included_documents"], context_omissions=prepared["omissions"])
        if prepared["sources"]:
            prompt = ChatPromptTemplate.from_messages([
                SystemMessage(content=GENERATOR_PROMPT),
                ("human", "사용자의 원본 질문: {question}\n검색 자료(JSON):\n{context}"),
            ])
            try:
                model = create_generator_llm() if llm is None else llm
                message = (prompt | model).invoke({"question": question, "context": prepared["context"]})
                response["token_usage"] = getattr(message, "usage_metadata", None) or {}

            except Exception as error:
                logger.exception("generator LLM 호출 실패")
                response.update(answer="답변 생성 서비스 호출에 실패했습니다.", generation_status="llm_error",
                    error_type=type(error).__name__, output_validation_error=str(error))
            else:
                output = None
                try:
                    output = json.loads(message.content)
                    status, answer, claims = output["status"], output["answer"], output["claims"]
                    if status not in {"answered", "insufficient_evidence", "needs_clarification", "conflicting_evidence"}:
                        raise ValueError("알 수 없는 생성 상태")
                    if not isinstance(answer, str) or not answer.strip() or not isinstance(claims, list):
                        raise ValueError("답변 형식이 올바르지 않습니다.")
                    claimed = set()
                    for claim in claims:
                        if (not isinstance(claim, dict) or not isinstance(claim.get("statement"), str)
                                or not claim["statement"].strip() or not isinstance(claim.get("source_ids"), list)
                                or not claim["source_ids"] or any(not isinstance(x, str) for x in claim["source_ids"])):
                            raise ValueError("주장별 근거 형식이 올바르지 않습니다.")
                        claimed.update(claim["source_ids"])
                    available = {s["source_id"] for s in prepared["sources"]}
                    # [1] 또는 [ S1 ]처럼 표기가 달라도 실제 출처와 주장 매핑이 있을 때만 통일합니다.
                    # 없는 번호에 새 출처를 만들거나, 임의로 인용을 추가하지 않습니다.
                    answer = re.sub(
                        r"\[\s*(S?\d+)\s*\]",
                        lambda match: f"[S{match.group(1).lstrip('S')}]"
                        if f"S{match.group(1).lstrip('S')}" in available & claimed else match.group(0),
                        answer,
                    )
                    markers = re.findall(r"\[\s*(S?\d+)\s*\]", answer)
                    if any(marker not in available for marker in markers):
                        raise ValueError("실제 출처와 매핑되지 않은 인용")
                    cited = set(re.findall(r"\[(S\d+)\]", answer))
                    if not (cited | claimed) <= available or (status == "answered" and (not cited or cited != claimed)):
                        raise ValueError("없는 출처 또는 누락된 인용")
                    used = cited | claimed
                    sources = [s for s in prepared["sources"] if s["source_id"] in used]
                    response.update(answer=answer, sources=sources, claims=claims, source_validation_passed=True,
                                    used_document_ids=list(dict.fromkeys(s["document_id"] or s["point_id"] for s in sources)),
                                    generation_status="generated_unverified" if status == "answered" else status)
                except (ValueError, KeyError, TypeError) as error:
                    response.update(answer="답변의 출처 형식을 확인할 수 없어 답변을 보류합니다.",
                                    generation_status="invalid_output", output_validation_error=str(error),
                                    rejected_output=output)
    response["latency_ms"] = round((perf_counter() - started) * 1000, 1)
    if verbose:
        print_response(response)
    return response


if __name__ == "__main__":
    # 기존 검색 함수를 재사용하고 실제 검색 근거로 생성을 실행합니다.
    question = input("질문을 입력하세요: ")
    rewritten = query_rewriter(question)
    result = multi_query(rewritten)
    hits = reciprocal_rank_fusion(result["rankings"], top_k=20)
    state = {"origin_query": question, "documents": [h.to_document() for h in hits]}
    state.update(rerank(state, verbose=False))
    generator(state)
