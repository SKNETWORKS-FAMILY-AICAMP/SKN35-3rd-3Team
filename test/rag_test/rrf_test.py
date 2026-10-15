"""고정된 질문별 검색 후보의 RRF 전후 품질·순위·융합 시간 평가."""
from evaluation.standalone import run_stage_test


def evaluate_rrf(context):
    combined = context.multi()
    combined = {**combined, 'details': {**combined['details'],
        'effect_scope': 'expanded candidate retrieval plus RRF fusion vs query_0; not a pure rank-only ablation'}}
    text, rewrite_ms, _ = context.rewrite()
    single, timings = context.search(text)
    return {
        "single_query": context.entry(single, rewrite_ms + timings["search_total_ms"],
                                       timings["search_total_ms"], timings=timings),
        "rrf": {**combined, "independent_ms": combined["details"]["fusion_ms"]},
    }


def main(argv=None):
    return run_stage_test("rrf", argv, evaluate_case=evaluate_rrf)


if __name__ == "__main__":
    raise SystemExit(main())
