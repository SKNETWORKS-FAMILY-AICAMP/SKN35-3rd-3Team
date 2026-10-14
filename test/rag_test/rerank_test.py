"""동일 실제 RRF 후보의 CPU Reranking 전후 품질·순위·추론 시간 평가."""
from evaluation.standalone import run_stage_test
from pathlib import Path
import sys


def evaluate_rerank(context):
    combined = context.multi()
    return {
        "reranking_before": combined,
        "reranking": context.reranked(output_k=context.config["candidate_k"]),
    }


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    return run_stage_test("reranking", ["--config", str(Path(__file__).with_name("rerank_config.json")), *arguments],
                          evaluate_case=evaluate_rerank)


if __name__ == "__main__":
    raise SystemExit(main())
