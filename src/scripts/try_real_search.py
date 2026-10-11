"""실제 청크를 읽고 BM25/Rerank를 한 줄 명령으로 시험합니다.

기본은 로컬 data/chunk, --branch를 주면 git show로 해당 브랜치 데이터를 읽습니다.
브랜치 데이터는 작업 폴더로 복사하지 않습니다. Qdrant/유료 API도 사용하지 않습니다.
정답 파일이 없으면 검색 결과만 보여 주고 평가 점수는 만들지 않습니다.
"""
import argparse
import json
from pathlib import Path
import subprocess
from time import perf_counter

from src.retriever import BM25Retriever
from src.retrieval_evaluation import EvaluationCase, chunk_key, compare_retrievers, evaluate_ranking
from src.vectorstore import VectorSearchHit


ROOT = Path(__file__).resolve().parents[2]


def load_documents(branch=None):
    """JSONL 3개와 안내서 보고서에 기록된 JSON을 읽습니다."""
    def read(relative):
        if branch:
            return subprocess.check_output(
                ["git", "show", f"{branch}:{relative}"], cwd=ROOT
            ).decode("utf-8")
        return (ROOT / relative).read_text(encoding="utf-8")

    rows = []
    for kind in ("laws", "precedents", "consumer_sources"):
        rows.extend(json.loads(line) for line in read(f"data/chunk/{kind}/chunks.jsonl").splitlines() if line.strip())
    report = json.loads(read("data/chunk/guides/guide_chunking_report.json"))
    for saved in report["saved"]:
        filename = saved["file"]
        # 보고서에는 파일명만 허용하여 data/chunk/guides 밖의 파일은 읽지 않습니다.
        if Path(filename).name != filename:
            raise ValueError("안내서 보고서 파일명 형식을 확인하세요.")
        rows.extend(json.loads(read(f"data/chunk/guides/{filename}"))["chunks"])
    documents = []
    for row in rows:
        if row.get("embedding_candidate") is False:
            continue
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        documents.append(VectorSearchHit(
            f"{row['document_id']}::{row['chunk_id']}", 0.0, text,
            {key: value for key, value in row.items() if key != "text"},
        ))
    return documents


def main():
    parser = argparse.ArgumentParser(description="실제 문서 BM25 검색 및 무료 CPU Rerank")
    parser.add_argument("--query", default="보증금 반환")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--document-type", help="law / precedent / official_guide 등")
    parser.add_argument("--branch", help="로컬 파일 대신 읽을 Git 기록. 예: origin/minha")
    parser.add_argument("--evaluation-file", type=Path, help="사람이 확인한 질문/정답 청크 ID 목록 JSON")
    args = parser.parse_args()
    if not 1 <= args.k <= 100 or not 1 <= args.candidates <= 100 or not args.query.strip():
        parser.error("K와 후보 수는 1~100, 질문은 비어 있지 않아야 합니다.")
    try:
        documents = load_documents(args.branch)
    except FileNotFoundError:
        parser.error("청크 파일이 없습니다. --branch origin/minha를 붙이면 가져온 브랜치 기록에서 읽습니다.")
    bm25 = BM25Retriever(documents)
    methods = {"BM25": bm25}
    cases = None
    if args.evaluation_file:
        raw = json.loads(args.evaluation_file.read_text(encoding="utf-8"))
        if not isinstance(raw, list) or not raw:
            parser.error("평가 파일은 질문 목록 JSON이어야 합니다.")
        cases = []
        valid_ids = {chunk_key(doc) for doc in documents}
        for item in raw:
            if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not item["query"].strip():
                parser.error("각 평가 항목에는 문자열 query가 필요합니다.")
            if not isinstance(item.get("relevant_ids"), list):
                parser.error("relevant_ids는 실제 정답 청크 ID 목록이어야 합니다.")
            evaluate_ranking([], item["relevant_ids"], k=args.k)
            if not set(item["relevant_ids"]) <= valid_ids:
                parser.error("평가 정답에 현재 데이터에 없는 청크 ID가 있습니다.")
            cases.append(EvaluationCase(item["query"], tuple(item["relevant_ids"]), item.get("filters")))

    print(f"청크 {len(documents):,}개 준비 완료", flush=True)
    if args.rerank:
        from src.rerank import CrossEncoderScorer, RerankingRetriever
        print("무료 CPU BGE 모델 로딩 중...", flush=True)
        methods["BM25_Rerank"] = RerankingRetriever(
            bm25, CrossEncoderScorer.from_pretrained(), candidate_k=args.candidates
        )
    if cases is not None:
        report = compare_retrievers(methods, cases, k=args.k)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    filters = {"document_type":args.document_type} if args.document_type else None
    for name, searcher in methods.items():
        start = perf_counter()
        hits = searcher.search_documents(args.query, top_k=args.k, filters=filters)
        print(f"\n[{name}] 결과 {len(hits)}개 / {perf_counter()-start:.3f}초")
        for rank, hit in enumerate(hits, 1):
            print(f"\n{rank}. {hit.metadata.get('title', '')} | 점수 {hit.score:.4f}")
            print(f"평가용 ID: {chunk_key(hit)}")
            print(f"출처: {hit.metadata.get('source_url') or hit.metadata.get('source_path')}")
            print(hit.text[:350])
            if hit.metadata.get("warnings"):
                print("주의:", hit.metadata["warnings"])
    print("\n정답 라벨을 지정하지 않았으므로 Recall/Precision/Hit/MRR은 계산하지 않았습니다.")


if __name__ == "__main__":
    main()
