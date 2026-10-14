"""단일/확장 질문의 검색·RRF 통합 품질과 개별 실행 시간 평가."""
from evaluation.standalone import run_stage_test


def evaluate_multi_query(context):
    combined = context.multi()
    text, rewrite_ms, _ = context.rewrite()
    single, timings = context.search(text)
    return {
        "single_query": context.entry(single, rewrite_ms + timings["search_total_ms"],
                                       timings["search_total_ms"], timings=timings),
        "multi_query": combined,
    }


def main(argv=None):
    return run_stage_test("multi_query", argv, evaluate_case=evaluate_multi_query)


if __name__ == "__main__":
    raise SystemExit(main())
