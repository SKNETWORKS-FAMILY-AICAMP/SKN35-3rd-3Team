"""검색 연결 확인 및 정답 데이터 기반 비교 실행.

사용: uv run python -B -m src.Scripts.search_evaluate --check
평가: --evaluation-file <JSON 경로> --multi-query-model <사용 가능한 Gemini 모델>
정답 JSON: [{"query":"질문", "relevant_ids":["document_id::chunk_id"], "filters":{}}]
"""
import argparse
import json
import os
from pathlib import Path

from src.const.env_loader import load_project_env
from src.const.qdrant_config import load_qdrant_settings
from src.multi_query import LLMQueryExpander, MultiQueryRetriever
from src.retrieval_evaluation import EvaluationCase, compare_retrievers, evaluate_ranking
from src.vector_db.qdrant_store import QdrantVectorStore
from src.vector_db.search_bootstrap import create_team_hybrid_retriever
from src.vectorstore import VectorSearchRequest


def main():
    parser = argparse.ArgumentParser(description="Qdrant 연결 확인 / 검색 성능 비교")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--evaluation-file", type=Path)
    parser.add_argument("--multi-query-model", default=os.getenv("MULTI_QUERY_MODEL"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rerank", action="store_true", help="무료 CPU BGE 모델로 재정렬 적용 전후 비교")
    parser.add_argument("--rerank-candidates", type=int, default=10, help="재정렬할 후보 수(최종 K와 별개)")
    parser.add_argument("--rerank-max-length", type=int, default=512, help="질문+문서 토큰 상한; 짧으면 빨라지지만 근거가 잘릴 수 있음")
    args = parser.parse_args()
    load_project_env()
    if args.check:
        settings = load_qdrant_settings()
        settings.validate_for_vector_work()
        # 원격 서버에는 키가 필요합니다. 키/주소 전체를 출력하지 않습니다.
        if "localhost" not in settings.connection.url and not settings.connection.api_key:
            parser.error("원격 Qdrant 접속 키가 없습니다. .env의 QDRANT_API_KEY를 설정하세요.")
        store = QdrantVectorStore.from_settings(settings)
        try:
            info = store.validate_collection_contract()
        except Exception as error:
            parser.error(f"연결 또는 규격 확인 실패 ({type(error).__name__}). 주소/키/네트워크를 확인하세요.")
        finally:
            store.client.close()
        print(json.dumps(info.to_dict(), ensure_ascii=False, indent=2))
        return
    if not args.evaluation_file:
        parser.error("--check 또는 --evaluation-file을 지정하세요.")
    model = args.multi_query_model or os.getenv("MULTI_QUERY_MODEL")
    if not model:
        parser.error("Multi-Query에 사용할 실제 모델을 --multi-query-model 또는 MULTI_QUERY_MODEL로 지정하세요.")
    raw = json.loads(args.evaluation_file.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        parser.error("평가 파일은 질문 목록 JSON이어야 합니다.")
    cases = []
    if not raw:
        parser.error("평가 질문이 없습니다.")
    for item in raw:
        if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not item["query"].strip():
            parser.error("각 평가 항목에는 문자열 query가 필요합니다.")
        if not isinstance(item.get("relevant_ids"), list):
            parser.error("relevant_ids는 정답 청크 ID의 문자열 목록이어야 합니다.")
        evaluate_ranking([], item["relevant_ids"], k=args.k)
        cases.append(EvaluationCase(item["query"], tuple(item["relevant_ids"]), item.get("filters")))
    # 설정의 Gemini 모델 이름을 임의 추측하지 않고 호출자가 명시하도록 합니다.
    from langchain_google_genai import ChatGoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model=model, temperature=0, google_api_key=os.getenv("GOOGLE_API_KEY"))
    hybrid = create_team_hybrid_retriever()
    expanded = MultiQueryRetriever(hybrid, LLMQueryExpander(llm))
    try:
        searchers = {"dense": hybrid.dense, "bm25": hybrid.keyword, "hybrid": hybrid, "multi_query_hybrid": expanded}
        if args.rerank:
            from src.rerank import CrossEncoderScorer, RerankingRetriever
            VectorSearchRequest.create([0.0], top_k=args.rerank_candidates)
            scorer = CrossEncoderScorer.from_pretrained(max_length=args.rerank_max_length)
            searchers["hybrid_rerank"] = RerankingRetriever(hybrid, scorer, candidate_k=args.rerank_candidates)
            searchers["multi_query_hybrid_rerank"] = RerankingRetriever(expanded, scorer, candidate_k=args.rerank_candidates)
        report = compare_retrievers(
            searchers,
            cases, k=args.k,
        )
    finally:
        hybrid.dense.vector_store.client.close()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
