"""같은 원격 Qdrant 검색기로 멀티쿼리 유무를 비교합니다."""
import argparse
import json
from pathlib import Path
from time import perf_counter

from src.const.model import create_llm
from src.multi_query import multi_query
from src.retrieval_evaluation import chunk_key, evaluate_ranking
from src.rrf import reciprocal_rank_fusion
from src.vector_db.bootstrap import create_team_retriever


def main():
    parser = argparse.ArgumentParser(description="멀티쿼리 없음 / 있음 검색 비교")
    parser.add_argument("--question", help="정답 파일 없이 비교할 질문")
    parser.add_argument("--evaluation-file", type=Path, help="query, relevant_ids를 가진 JSON 목록")
    parser.add_argument("--k", type=int, default=3, help="최종 문서 개수")
    parser.add_argument("--candidate-k", type=int, default=20, help="질문별 검색 후보 개수")
    args = parser.parse_args()
    if not 1 <= args.k <= args.candidate_k <= 100:
        parser.error("1 <= k <= candidate-k <= 100이어야 합니다.")
    if args.question and args.evaluation_file:
        parser.error("--question과 --evaluation-file 중 하나만 지정하세요.")

    # 정답 파일이 없으면 입력 질문으로 문서만 비교합니다.
    cases = (json.loads(args.evaluation_file.read_text(encoding="utf-8-sig"))
             if args.evaluation_file else [{"query": args.question or input("질문을 입력하세요: ")}])
    if isinstance(cases, dict):
        # 새 평가 JSON은 라벨 정책과 cases 목록을 함께 보관합니다.
        print("평가 라벨 정책:", cases.get("label_policy", "미기재"))
        cases = cases.get("cases")
    if not isinstance(cases, list) or not cases:
        parser.error("평가 파일은 비어 있지 않은 질문 목록이어야 합니다.")
    # 모든 질문과 정답을 검사한 뒤 실제 API를 호출합니다.
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("query"), str) or not case["query"].strip():
            parser.error("각 항목에는 비어 있지 않은 query가 필요합니다.")
        if args.evaluation_file:
            if not isinstance(case.get("relevant_ids"), list):
                parser.error("각 평가 항목에는 relevant_ids 목록이 필요합니다.")
            evaluate_ranking([], case["relevant_ids"], k=args.k)

    # 두 방식에 같은 Qdrant Dense/Sparse 검색기와 후보 개수를 사용합니다.
    searcher = create_team_retriever()
    scores = {"멀티쿼리 없음": [], "멀티쿼리 있음": []}
    try:
        llm = create_llm()
        for case in cases:
            question, filters = case["query"], case.get("filters")
            print(f"\n질문: {question}")
            start = perf_counter()
            original = list(searcher.search_documents(
                question, top_k=args.candidate_k, filters=filters,
            ))[:args.k]
            original_seconds = perf_counter() - start
            start = perf_counter()
            response = multi_query(question, llm, searcher,
                                   candidate_k=args.candidate_k, filters=filters)
            expanded = reciprocal_rank_fusion(response["rankings"], top_k=args.k)
            expanded_seconds = perf_counter() - start

            # ID와 본문 앞부분만 출력하며 전체 메타데이터는 출력하지 않습니다.
            for label, docs, seconds in [("멀티쿼리 없음", original, original_seconds),
                                         ("멀티쿼리 있음", expanded, expanded_seconds)]:
                print(f"\n=== {label} ({seconds:.2f}초) ===")
                if not docs:
                    print("검색 결과 없음")
                for index, doc in enumerate(docs, 1):
                    print(f"[{index}] id={chunk_key(doc)}\n{doc.text[:200]}\n")
                # 정답 근거가 있을 때만 검색 지표를 계산합니다.
                if args.evaluation_file:
                    metrics = evaluate_ranking([chunk_key(doc) for doc in docs],
                                               case["relevant_ids"], k=args.k)
                    scores[label].append(metrics)
                    print(f"지표@{args.k}: {metrics}")
        if args.evaluation_file:
            print("\n=== 전체 질문 평균 (rr 평균은 MRR) ===")
            for label, rows in scores.items():
                print(label, {key: round(sum(row[key] for row in rows) / len(rows), 4)
                              for key in rows[0]})
    finally:
        searcher.vector_store.client.close()


if __name__ == "__main__":
    main()
