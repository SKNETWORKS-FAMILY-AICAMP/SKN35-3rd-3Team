"""같은 실제 컬렉션의 Dense/Sparse/Hybrid 검색 품질·시간 비교."""
from evaluation.standalone import run_stage_test


def evaluate_hybrid_search(context, modes=("dense", "sparse", "hybrid")):
    results = {}
    for mode in modes:
        hits, timings = context.search(context.question, mode)
        results[mode] = context.entry(hits, timings["search_total_ms"],
                                     timings=timings, query=context.question, search_mode=mode)
    return results


def main(argv=None):
    return run_stage_test("hybrid", argv, evaluate_case=evaluate_hybrid_search)


if __name__ == "__main__":
    raise SystemExit(main())
