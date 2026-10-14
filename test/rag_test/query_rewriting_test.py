"""원본/재작성 질문의 동일 Retriever 검색 품질과 실행 시간 평가."""
from evaluation.standalone import run_stage_test


def evaluate_query_rewriting(context):
    original, original_time = context.search(context.question)
    rewritten, rewrite_ms, llm_ms = context.rewrite()
    hits, search_time = context.search(rewritten)
    return {
        "query_rewriting_original": context.entry(original, original_time["search_total_ms"],
                                                   timings=original_time, query=context.question),
        "query_rewriting": context.entry(hits, rewrite_ms + search_time["search_total_ms"], rewrite_ms,
                            rewrite_ms=rewrite_ms, llm_call_ms=llm_ms, timings=search_time, query=rewritten),
    }


def main(argv=None):
    return run_stage_test("query_rewriting", argv, evaluate_case=evaluate_query_rewriting)


if __name__ == "__main__":
    raise SystemExit(main())
