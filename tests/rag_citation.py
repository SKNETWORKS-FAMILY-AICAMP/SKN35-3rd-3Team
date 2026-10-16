"""Citation 기능 검사와 실제 Qdrant 기반 통합 테스트를 함수로 실행합니다."""
import argparse
import copy
import json
from time import perf_counter
from urllib.parse import urlparse

from src.const.config import RAG_CITATION_REPORT_PATH, RAG_EVALUATION_CASES_PATH
from src.const.model import OPENAI_LLM_MODEL
from src.citation import citation
from src.rag.generator import build_context, generator
from src.rag.query_rewriter import query_rewriter
from src.rag.multi_query import multi_query
from src.rag.rrf import reciprocal_rank_fusion
from src.rag.rerank import rerank
from src.vector_db.bootstrap import create_team_retriever


# 실제 수집 근거로 인용 구조의 성공·실패 처리를 확인합니다. API 통합 성공으로 보고하지 않습니다.
def check_structure() -> list[dict]:
    data = json.loads(RAG_EVALUATION_CASES_PATH.read_text(encoding="utf-8-sig"))
    doc = data["cases"][0]["evidence"][0]
    context = build_context([doc], OPENAI_LLM_MODEL, 2000)
    base = {"origin_query": "구조 검사", "documents": [doc],
            "context_sources": context["sources"], "context_documents": context["included_documents"],
            "answer": "인용 구조 검사 [S1]", "claims": [{"statement": "인용 구조 검사", "source_ids": ["S1"]}],
            "generation_status": "generated_unverified"}
    cases = [("valid_mapping", copy.deepcopy(base), "passed")]
    bad = copy.deepcopy(base); bad["answer"] = "없는 번호 [S99]"
    cases.append(("unknown_citation", bad, "failed"))
    bad = copy.deepcopy(base); bad["context_sources"][0]["document_id"] = None
    cases.append(("missing_document_id", bad, "failed"))
    bad = copy.deepcopy(base); bad["context_sources"][0]["source_url"] = "https://example.invalid"
    cases.append(("metadata_mismatch", bad, "failed"))
    bad = copy.deepcopy(base); bad["documents"][0]["text"] = ""
    cases.append(("missing_text", bad, "failed"))
    bad = copy.deepcopy(base); bad["context_sources"].append(copy.deepcopy(bad["context_sources"][0]))
    cases.append(("duplicate_number", bad, "failed"))
    bad = copy.deepcopy(base); bad["answer"] = "번호 없음"
    cases.append(("no_citation", bad, "failed"))
    bad = copy.deepcopy(base); bad["context_sources"][0]["source_url"] = None
    bad["documents"][0]["metadata"]["source_url"] = None
    cases.append(("missing_url", bad, "passed_with_missing_metadata"))
    bad = copy.deepcopy(base); bad["generation_status"] = "llm_error"; bad["answer"] = "호출 실패 안내"
    cases.append(("generator_error", bad, "skipped"))
    results = []
    for name, state, expected in cases:
        result = citation(state, verbose=False)
        assert result["citation_validation_status"] == expected, (name, result)
        results.append({"case": name, "status": result["citation_validation_status"],
                        "errors": result["validation_errors"]})
        print(f"PASS: {name}")
    return results


# 기존 검색 함수를 실행하고 실제 공개 법령 검색 결과로 Generator·Citation을 검사합니다.
def main() -> None:
    parser = argparse.ArgumentParser(description="Citation 구조 검사 / 실제 공개 법령 통합 테스트")
    parser.add_argument("--live", action="store_true", help="Qdrant·OpenAI 실제 호출")
    args = parser.parse_args()
    checks = check_structure()
    report = {"structure_checks": checks, "integration": "not_run", "judge": "not_run"}
    if args.live:
        question = "최저임금법의 목적은 무엇인가요?"
        started = perf_counter()
        searcher = create_team_retriever()
        assert searcher.vector_store.settings.collection.search_mode == "hybrid"
        try:
            collection = searcher.get_collection_info().to_dict()
            rewritten = query_rewriter(question)
            result = multi_query(rewritten, searcher=searcher, candidate_k=5,
                                 filters={"document_id": "law:000129_218303_20200526"})
            hits = reciprocal_rank_fusion(result["rankings"], top_k=5)
        finally:
            searcher.vector_store.client.close()
        assert hits, "실제 Qdrant 검색 결과가 없습니다."
        # 공개 법령과 필요한 메타데이터만 외부 모델에 전달합니다.
        allowed = ["document_id", "chunk_id", "source_name", "law_name", "article_label", "article_number",
                   "article_title", "source_url", "document_type", "effective_date", "version_id", "before_rerank"]
        documents = []
        for hit in hits:
            assert hit.metadata.get("document_id") == "law:000129_218303_20200526"
            assert hit.metadata.get("document_type") == "law"
            assert urlparse(hit.metadata.get("source_url", "")).hostname in {"law.go.kr", "www.law.go.kr"}
            documents.append({**hit.to_document(), "metadata": {key: hit.metadata.get(key) for key in allowed}})
        state = {"origin_query": question, "documents": documents}
        state.update(rerank(state, top_k=3, verbose=False))
        generated = generator(state, verbose=False)
        state.update(generated)
        checked = citation(state)
        report.update(integration="completed" if checked["citation_validation_status"].startswith("passed") else "failed",
                      data_origin="live remote Qdrant hybrid search; no local files used as integration evidence",
                      collection=collection, question=question, queries=result["queries"],
                      retrieved_counts={name:len(docs) for name,docs in result["rankings"].items()},
                      generated_answer=generated["answer"], generation_status=generated["generation_status"],
                      model_name=generated["model_name"], token_usage=generated["token_usage"],
                      generator_latency_ms=generated["latency_ms"], citation_result=checked,
                      total_latency_ms=round((perf_counter()-started)*1000,1))
    # 결과 저장 경로는 config에서 가져옵니다.
    RAG_CITATION_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RAG_CITATION_REPORT_PATH.write_text(
        "# Citation 검증 보고서\n\n구조 검사만으로 법률적 정확성을 보증하지 않습니다.\n\n```json\n"
        + json.dumps(report, ensure_ascii=False, indent=2) + "\n```\n", encoding="utf-8")
    print(f"보고서: {RAG_CITATION_REPORT_PATH}")
    if report["integration"] == "failed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
