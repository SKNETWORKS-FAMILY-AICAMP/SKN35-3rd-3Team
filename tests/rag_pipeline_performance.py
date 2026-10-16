"""실제 RAG 파이프라인의 단계별 지연과 검색 품질을 전후 비교합니다.

외부 LLM, 임베딩 API, Qdrant와 로컬 리랭커를 모두 사용하므로 일반 단위
테스트에는 포함하지 않습니다. 같은 케이스 파일로 ``before``와 ``after``를
각각 실행해야 비교표가 유효합니다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
from collections import defaultdict
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from time import perf_counter
from typing import Callable, TypeVar

import torch

from src.const.model import OPENAI_LLM_MODEL, RERANK_MODEL
from src.rag.generator import generator
from src.graph.rag_graph.validation_node import RERANK_MIN_SCORE, validation_node
from src.graph.workflow import route_start
from src.rag.multi_query import create_queries, search_rankings
from src.rag.query_rewriter import query_rewriter
from src.rag.rerank import rerank
from src.rag.retrieval_evaluation import evaluate_ranking
from src.rag.rrf import reciprocal_rank_fusion
from src.vector_db.bootstrap import create_team_retriever


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = PROJECT_ROOT / "tests" / "data" / "rag_performance_cases.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "docs" / "rag_performance"
SOURCE_DATASET = PROJECT_ROOT / "rag_evaluation_cases.review.json"
T = TypeVar("T")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("백분위수를 계산할 값이 없습니다.")
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _timed(stage_ms: dict[str, float], name: str, call: Callable[[], T]) -> T:
    started = perf_counter()
    with redirect_stdout(StringIO()):
        value = call()
    stage_ms[name] += (perf_counter() - started) * 1000
    return value


def _document_key(document: dict) -> str:
    metadata = document.get("metadata") or {}
    document_id = metadata.get("document_id")
    chunk_id = metadata.get("chunk_id")
    if isinstance(document_id, str) and document_id and isinstance(chunk_id, str) and chunk_id:
        return f"{document_id}::{chunk_id}"
    return str(document.get("point_id") or "")


def _load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("cases"), list):
        raise ValueError("성능 케이스 파일 형식이 올바르지 않습니다.")

    source = json.loads(SOURCE_DATASET.read_text(encoding="utf-8"))
    reviewed = {case["case_id"]: case for case in source["cases"]}
    seen: set[str] = set()
    for case in payload["cases"]:
        case_id = case.get("case_id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError("case_id는 중복 없는 문자열이어야 합니다.")
        seen.add(case_id)
        original = reviewed.get(case_id)
        if original is None or original.get("review_status") != "reviewed":
            raise ValueError(f"{case_id}: 검수 완료 원본 케이스가 없습니다.")
        for field in ("query", "filters", "relevant_ids"):
            if case.get(field) != original.get(field):
                raise ValueError(f"{case_id}: 원본 평가 데이터의 {field}와 다릅니다.")
        evaluate_ranking([], case["relevant_ids"], k=5)
    return payload["cases"]


def _measure_case(case: dict) -> dict:
    stage_ms: dict[str, float] = defaultdict(float)
    attempts: list[dict] = []
    total_started = perf_counter()
    question = case["query"]

    route = _timed(
        stage_ms,
        "routing",
        lambda: route_start({"origin_query": question, "messages": []}),
    )

    retry_count = 0
    final_documents: list[dict] = []
    final_score: float | None = None
    while True:
        rewritten = _timed(
            stage_ms,
            "query_rewrite",
            lambda: query_rewriter(question),
        )
        retry_count += 1
        queries = _timed(
            stage_ms,
            "multi_query_generation",
            lambda: create_queries(rewritten, max_queries=3),
        )
        searcher = _timed(stage_ms, "retriever_init", create_team_retriever)
        search_calls_ms: list[float] = []
        try:
            started = perf_counter()
            rankings = search_rankings(
                searcher,
                queries,
                candidate_k=20,
                filters=case.get("filters"),
            )
            elapsed_ms = (perf_counter() - started) * 1000
            search_calls_ms.append(elapsed_ms)
            stage_ms["retrieval"] += elapsed_ms
        finally:
            searcher.vector_store.client.close()

        hits = _timed(
            stage_ms,
            "rrf",
            lambda: reciprocal_rank_fusion(rankings, top_k=20),
        )
        rerank_timings: dict[str, float] = {}
        reranked = _timed(
            stage_ms,
            "rerank",
            lambda: rerank(
                {
                    "origin_query": question,
                    "rewrite_query": rewritten,
                    "queries": queries,
                    "documents": [hit.to_document() for hit in hits],
                },
                top_k=5,
                timings=rerank_timings,
                verbose=False,
            ),
        )
        validation = _timed(
            stage_ms,
            "validation",
            lambda: validation_node(
                {
                    "documents": reranked["documents"],
                    "score": reranked["score"],
                    "retry_count": retry_count,
                }
            ),
        )
        final_documents = reranked["documents"]
        final_score = reranked["score"]
        attempts.append(
            {
                "attempt": retry_count,
                "queries": queries,
                "query_count": len(queries),
                "search_calls_ms": [round(value, 3) for value in search_calls_ms],
                "candidate_count": len(hits),
                "rerank_window_count": rerank_timings.get("window_count", 0),
                "rerank_model_load_ms": round(
                    rerank_timings.get("model_load_seconds", 0.0) * 1000, 3
                ),
                "rerank_inference_ms": round(
                    rerank_timings.get("inference_seconds", 0.0) * 1000, 3
                ),
                "score": final_score,
                "is_valid": validation["is_valid"],
                "validation_reason": validation.get("validation_reason"),
            }
        )
        if validation["is_valid"]:
            break
        if retry_count >= 3:
            raise RuntimeError("RAG 재시도 횟수가 예상 범위를 초과했습니다.")

    generated = _timed(
        stage_ms,
        "generation",
        lambda: generator(
            {"origin_query": question, "documents": final_documents},
            verbose=False,
        ),
    )
    retrieved_ids = [_document_key(document) for document in final_documents]
    quality = evaluate_ranking(retrieved_ids, case["relevant_ids"], k=5)

    return {
        "case_id": case["case_id"],
        "scenario": case["scenario"],
        "query": question,
        "relevant_ids": case["relevant_ids"],
        "route": route,
        "total_ms": round((perf_counter() - total_started) * 1000, 3),
        "stage_ms": {key: round(value, 3) for key, value in stage_ms.items()},
        "retry_count": retry_count,
        "attempts": attempts,
        "final_score": final_score,
        "retrieved_ids": retrieved_ids,
        "quality_at_5": quality,
        "generation_status": generated.get("generation_status"),
        "generation_error": generated.get("output_validation_error"),
        "generator_reported_latency_ms": generated.get("latency_ms"),
        "token_usage": generated.get("token_usage") or {},
    }


def _mean(rows: list[dict], getter: Callable[[dict], float]) -> float:
    return statistics.fmean(getter(row) for row in rows)


def _summary(rows: list[dict]) -> dict:
    totals = [row["total_ms"] for row in rows]
    stage_names = (
        "routing",
        "query_rewrite",
        "multi_query_generation",
        "retriever_init",
        "retrieval",
        "rrf",
        "rerank",
        "validation",
        "generation",
    )
    stage_mean_ms = {
        name: round(_mean(rows, lambda row, key=name: row["stage_ms"].get(key, 0.0)), 3)
        for name in stage_names
    }
    rerank_load = [
        sum(attempt["rerank_model_load_ms"] for attempt in row["attempts"])
        for row in rows
    ]
    rerank_inference = [
        sum(attempt["rerank_inference_ms"] for attempt in row["attempts"])
        for row in rows
    ]
    mean_total = statistics.fmean(totals)
    return {
        "case_count": len(rows),
        "mean_total_ms": round(mean_total, 3),
        "p50_total_ms": round(_percentile(totals, 0.50), 3),
        "p95_total_ms": round(_percentile(totals, 0.95), 3),
        "throughput_questions_per_minute": round(60000 / mean_total, 4),
        "mean_retry_count": round(_mean(rows, lambda row: row["retry_count"]), 4),
        "stage_mean_ms": stage_mean_ms,
        "mean_rerank_model_load_ms": round(statistics.fmean(rerank_load), 3),
        "mean_rerank_inference_ms": round(statistics.fmean(rerank_inference), 3),
        "quality_at_5": {
            key: round(_mean(rows, lambda row, metric=key: row["quality_at_5"][metric]), 6)
            for key in ("recall", "precision", "hit", "rr", "ndcg")
        },
        "generation_success_rate": round(
            _mean(
                rows,
                lambda row: float(row["generation_status"] == "generated_unverified"),
            ),
            6,
        ),
    }


def _value(report: dict, path: str) -> float:
    value: object = report["summary"]
    for key in path.split("."):
        value = value[key]  # type: ignore[index]
    return float(value)


def _comparison_markdown(before: dict, after: dict | None) -> str:
    specs = [
        ("전체 응답 평균", "mean_total_ms", "ms", "lower"),
        ("전체 응답 P50", "p50_total_ms", "ms", "lower"),
        ("전체 응답 P95", "p95_total_ms", "ms", "lower"),
        ("처리량", "throughput_questions_per_minute", "질문/분", "higher"),
        ("라우팅", "stage_mean_ms.routing", "ms", "lower"),
        ("질문 재작성", "stage_mean_ms.query_rewrite", "ms", "lower"),
        ("멀티쿼리 생성", "stage_mean_ms.multi_query_generation", "ms", "lower"),
        ("검색기 초기화", "stage_mean_ms.retriever_init", "ms", "lower"),
        ("검색", "stage_mean_ms.retrieval", "ms", "lower"),
        ("RRF", "stage_mean_ms.rrf", "ms", "lower"),
        ("리랭크 전체", "stage_mean_ms.rerank", "ms", "lower"),
        ("리랭크 추론", "mean_rerank_inference_ms", "ms", "lower"),
        ("검증", "stage_mean_ms.validation", "ms", "lower"),
        ("답변 생성", "stage_mean_ms.generation", "ms", "lower"),
        ("평균 실행 횟수", "mean_retry_count", "회", "lower"),
        ("Recall@5", "quality_at_5.recall", "score", "higher"),
        ("Precision@5", "quality_at_5.precision", "score", "higher"),
        ("Hit Rate@5", "quality_at_5.hit", "score", "higher"),
        ("MRR@5", "quality_at_5.rr", "score", "higher"),
        ("nDCG@5", "quality_at_5.ndcg", "score", "higher"),
        ("생성 성공률", "generation_success_rate", "score", "higher"),
    ]
    environment = (after or before)["environment"]
    lines = [
        "# RAG 성능 개선 전후 비교",
        "",
        "같은 테스트 케이스와 정답 ID를 사용합니다. 지연 시간은 낮을수록, 처리량과 품질 지표는 높을수록 좋습니다.",
        "상승치는 개선 방향을 양수로 표시합니다. 품질 지표는 퍼센트포인트 차이입니다.",
        "",
        "## 테스트 환경",
        "",
        "| 항목 | 값 |",
        "|---|---|",
        f"| 운영체제 | {environment['platform']} |",
        f"| CPU | {environment['processor']} · 논리 코어 {environment['logical_cpu_count']}개 |",
        f"| Python / PyTorch | {environment['python']} / {environment['torch']} |",
        f"| 리랭커 실행 | CPU · PyTorch 스레드 {environment['torch_threads']}개 |",
        f"| CUDA | {'사용' if environment['cuda_available'] else '미사용'} |",
        f"| LLM / 리랭커 | {before['settings']['llm_model']} / {before['settings']['rerank_model']} |",
        "| 측정 조건 | 케이스당 1회 · 실제 OpenAI/Qdrant 네트워크 호출 · 첫 모델 적재 포함 |",
        "",
        "## 성능 비교",
        "",
        "| 영역 | 개선 전 | 개선 후 | 상승치 |",
        "|---|---:|---:|---:|",
    ]
    for label, path, unit, direction in specs:
        before_value = _value(before, path)
        before_display = f"{before_value:.3f} {unit}" if unit != "score" else f"{before_value * 100:.2f}%"
        if after is None:
            after_display = "미측정"
            improvement = "미측정"
        else:
            after_value = _value(after, path)
            after_display = f"{after_value:.3f} {unit}" if unit != "score" else f"{after_value * 100:.2f}%"
            if unit == "score":
                improvement = f"{(after_value - before_value) * 100:+.2f}%p"
            elif before_value == 0:
                improvement = f"{after_value - before_value:+.3f} {unit}"
            else:
                signed = (before_value - after_value) / before_value if direction == "lower" else (after_value - before_value) / before_value
                improvement = f"{signed * 100:+.2f}%"
        lines.append(f"| {label} | {before_display} | {after_display} | {improvement} |")

    lines += [
        "",
        "## 케이스별 결과",
        "",
        "| 케이스 | 유형 | 개선 전 시간(ms) | 개선 후 시간(ms) | 전→후 Recall@5 | 최종 결과 | 비고 |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    after_by_id = {row["case_id"]: row for row in after["cases"]} if after else {}
    for row in before["cases"]:
        improved = after_by_id.get(row["case_id"])
        if improved is None:
            lines.append(
                f"| {row['case_id']} | {row['scenario']} | {row['total_ms']:.3f} | 미측정 | "
                f"{row['quality_at_5']['recall']:.3f}→미측정 | 미측정 | - |"
            )
            continue
        failures = []
        if improved["quality_at_5"]["hit"] == 0:
            failures.append("검색·재정렬 상위 5개에 검수 정답 ID 없음")
        if improved["generation_status"] != "generated_unverified":
            failures.append(
                f"답변 생성 상태 {improved['generation_status']}"
                + (f" ({improved.get('generation_error')})" if improved.get("generation_error") else "")
            )
        result = "FAIL" if failures else "PASS"
        lines.append(
            f"| {row['case_id']} | {row['scenario']} | {row['total_ms']:.3f} | {improved['total_ms']:.3f} | "
            f"{row['quality_at_5']['recall']:.3f}→{improved['quality_at_5']['recall']:.3f} | "
            f"{result} | {'; '.join(failures) if failures else '정답 근거 포함·답변 형식 통과'} |"
        )
    lines += [
        "",
        "## 특이사항",
        "",
        "- 실제 외부 API와 원격 벡터 DB를 사용한 1회 측정이므로 네트워크 상태에 따라 수치가 달라질 수 있습니다.",
        "- 최초 리랭커 모델 적재 시간을 포함했습니다. 이후 같은 프로세스에서는 캐시로 적재 비용이 줄어듭니다.",
        "- 라우터가 q001·q007을 범위 밖으로 오분류했지만, RAG 자체를 비교하기 위해 라우팅 시간만 기록하고 동일 RAG 파이프라인을 강제 실행했습니다.",
        "- 검수 정답은 대표 근거 ID이며 전체 말뭉치의 모든 관련 문서를 완전 판정한 목록은 아닙니다.",
        "- 생성 상태가 invalid_output이면 검색 근거와 별개로 JSON·인용 형식 검증을 통과하지 못한 경우입니다.",
    ]
    lines += [
        "",
        "## 재현 명령",
        "",
        "```powershell",
        "uv run python -B tests/rag_pipeline_performance.py --phase before",
        "uv run python -B tests/rag_pipeline_performance.py --phase after",
        "```",
        "",
        "`after`는 성능 개선 코드 적용 뒤 동일한 PC·네트워크 조건에서 실행합니다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 단계별 성능 전후 비교")
    parser.add_argument("--phase", required=True, choices=("before", "after"))
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case-id", action="append", dest="case_ids")
    args = parser.parse_args()

    cases = _load_cases(args.cases)
    if args.case_ids:
        selected = set(args.case_ids)
        cases = [case for case in cases if case["case_id"] in selected]
        missing = selected - {case["case_id"] for case in cases}
        if missing:
            parser.error(f"알 수 없는 case_id: {sorted(missing)}")
    if not cases:
        parser.error("실행할 케이스가 없습니다.")

    rows = []
    for index, case in enumerate(cases, 1):
        print(f"[{index}/{len(cases)}] {case['case_id']} {case['scenario']} 측정")
        row = _measure_case(case)
        rows.append(row)
        print(f"  total={row['total_ms'] / 1000:.2f}s, retry={row['retry_count']}, hit@5={row['quality_at_5']['hit']:.0f}")

    report = {
        "schema_version": 1,
        "phase": args.phase,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "case_file": str(args.cases.relative_to(PROJECT_ROOT)),
        "case_file_sha256": _digest(args.cases),
        "source_dataset_sha256": _digest(SOURCE_DATASET),
        "settings": {
            "candidate_k": 20,
            "final_k": 5,
            "max_queries": 3,
            "rerank_min_score": RERANK_MIN_SCORE,
            "llm_model": OPENAI_LLM_MODEL,
            "rerank_model": RERANK_MODEL,
        },
        "environment": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_threads": torch.get_num_threads(),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "notes": [
            "케이스당 1회 실측이며 OpenAI와 원격 Qdrant 네트워크 지연을 포함합니다.",
            "첫 리랭커 모델 적재 시간을 포함하고 이후 케이스는 프로세스 캐시를 사용합니다.",
            "라우팅 결과와 관계없이 RAG 영역 자체를 같은 조건으로 측정했습니다.",
            "검수 정답 ID는 대표 근거이며 전체 말뭉치의 완전한 관련성 판정이 아닙니다.",
        ],
        "summary": _summary(rows),
        "cases": rows,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / f"rag_performance_{args.phase}.json"
    result_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    before_path = args.output_dir / "rag_performance_before.json"
    after_path = args.output_dir / "rag_performance_after.json"
    if before_path.exists():
        before = json.loads(before_path.read_text(encoding="utf-8"))
        after = json.loads(after_path.read_text(encoding="utf-8")) if after_path.exists() else None
        comparison_path = args.output_dir / "rag_performance_comparison.md"
        comparison_path.write_text(_comparison_markdown(before, after), encoding="utf-8")
        print(f"비교표: {comparison_path}")
    print(f"측정값: {result_path}")


if __name__ == "__main__":
    main()
