"""실제 수집 근거로 기능을 검사합니다. 실제 API 생성 평가는 별도입니다."""
import copy
import json

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from src.const.config import RAG_EVALUATION_CASES_PATH
from src.const.model import OPENAI_LLM_MODEL
from src.generator import build_context, generator


# 문서가 없을 때 생성 API 없이 근거 부족 상태를 반환하는지 확인합니다.
def test_empty_documents_skip_api(evidence):
    result = generator({"origin_query": "근거 확인", "documents": []}, verbose=False)
    assert result["generation_status"] == "insufficient_evidence"


# 동일한 실제 청크를 두 번 전달해도 출처와 본문이 한 번만 포함되는지 확인합니다.
def test_duplicate_real_chunk(evidence):
    doc = evidence[0]
    result = build_context([doc, doc], OPENAI_LLM_MODEL, 6000)
    assert len(result["sources"]) == 1
    assert result["included_documents"][0]["text"] == doc["text"]


# 긴 실제 본문이 토큰 한도를 지키고 잘림 기록을 남기는지 확인합니다.
def test_long_real_document_budget(evidence):
    doc = max(evidence, key=lambda d: len(d["text"]))
    result = build_context([doc], OPENAI_LLM_MODEL, 600)
    assert result["context_tokens"] <= 600
    assert result["sources"] and result["sources"][0]["truncated"]
    assert doc["text"].startswith(result["included_documents"][0]["text"])


# 출처 정보가 없을 때 법령명이나 URL을 임의로 만들지 않는지 확인합니다.
def test_missing_metadata_does_not_invent_source(evidence):
    doc = copy.deepcopy(evidence[0])
    doc["metadata"] = {}
    result = build_context([doc], OPENAI_LLM_MODEL, 1000)
    assert result["sources"][0]["source_name"] is None
    assert result["sources"][0]["source_url"] is None


# 실패한 API 호출의 상태를 근거 부족 상태와 구분하는지 확인합니다.
def test_call_failure_is_distinct(evidence):
    # 오류만 재현하며, 실제 모델 성공이나 검색 품질을 모의 결과로 주장하지 않습니다.
    def fail(_):
        raise ConnectionError("test")
    result = generator({"origin_query": "최저임금법의 목적", "documents": [evidence[0]]},
                       llm=RunnableLambda(fail), verbose=False)
    assert result["generation_status"] == "llm_error"
    assert result["error_type"] == "ConnectionError"


# 제공되지 않은 출처를 인용한 응답을 보류하는지 확인합니다.
def test_nonexistent_citation_is_withheld(evidence):
    output = {"status": "answered", "answer": "출처 검사 [S999]",
              "claims": [{"statement": "출처 검사", "source_ids": ["S999"]}]}
    invalid = RunnableLambda(lambda _: AIMessage(content=json.dumps(output)))
    result = generator({"origin_query": "최저임금법의 목적", "documents": [evidence[0]]},
                       llm=invalid, verbose=False)
    assert result["generation_status"] == "invalid_output"
    assert not result["source_validation_passed"]


# 실제 존재하는 출처의 숫자 표기만 정규화하고, 없는 숫자 인용은 거부합니다.
def test_numeric_citation_normalization(evidence):
    for marker, expected in [("[1]", "generated_unverified"), ("[ S1 ]", "generated_unverified"),
                             ("[ S1]", "generated_unverified"), ("[ 1 ]", "generated_unverified"),
                             ("[999]", "invalid_output"), ("[ S999 ]", "invalid_output")]:
        output = {"status": "answered", "answer": f"인용 형식 검사 {marker}",
                  "claims": [{"statement": "인용 형식 검사", "source_ids": ["S1"]}]}
        result = generator({"origin_query": "최저임금법의 목적", "documents": [evidence[0]]},
                           llm=RunnableLambda(lambda _, value=output: AIMessage(content=json.dumps(value))),
                           verbose=False)
        assert result["generation_status"] == expected
        if expected == "generated_unverified":
            assert result["answer"].endswith("[S1]")


# 공통 경로의 실제 수집 근거를 읽고 모든 검사 결과를 출력합니다.
def main():
    data = json.loads(RAG_EVALUATION_CASES_PATH.read_text(encoding="utf-8-sig"))
    evidence = [doc for case in data["cases"] for doc in case["evidence"]]
    if not evidence:
        raise ValueError("검사할 실제 수집 근거가 없습니다.")
    tests = [test_empty_documents_skip_api, test_duplicate_real_chunk,
             test_long_real_document_budget, test_missing_metadata_does_not_invent_source,
             test_call_failure_is_distinct, test_nonexistent_citation_is_withheld,
             test_numeric_citation_normalization]
    failures = []
    for test in tests:
        try:
            test(evidence)
            print(f"PASS: {test.__name__}")
        except Exception as error:
            failures.append(test.__name__)
            print(f"FAIL: {test.__name__} ({type(error).__name__}: {error})")
    print(f"\n총 {len(tests)}개: 통과 {len(tests) - len(failures)}, 실패 {len(failures)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
