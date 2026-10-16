from src.graph.rag_graph.rag_graph import build_rag_graph
import contextlib
import io

if __name__ == "__main__":

    question = input("질문을 입력하세요: ")
    graph = build_rag_graph()

    stream = graph.stream(
        {"origin_query": question, "retry_count": 0, "messages": []},
        stream_mode="updates",
    )

    while True:
        try:
            # 노드·구현 함수의 내부 print를 숨깁니다.
            with contextlib.redirect_stdout(io.StringIO()):
                update = next(stream)
        except StopIteration:
            break

        # 긴 메타데이터 대신 단계별 핵심 결과만 출력합니다.
        for node, result in update.items():
            print(f"\n=== {node} 완료 ===")
            if "rewrite_query" in result:
                print("원본:", question)
                print("재작성:", result["rewrite_query"])

            if "queries" in result:
                for index, query in enumerate(result["queries"], 1):
                    print(f"검색 질문 {index}: {query}")
            if "rankings" in result:
                print("질문별 검색 문서 수:",
                      {name: len(docs) for name, docs in result["rankings"].items()})

            if "documents" in result:
                documents = result["documents"]
                print(f"문서 {len(documents)}개 — 상위 3개 표시")
                for index, doc in enumerate(documents[:3], 1):
                    meta = doc.get("metadata", {})
                    title = meta.get("law_name") or meta.get("source_name") or "출처명 없음"
                    article = meta.get("article_label") or meta.get("article_number") or ""
                    print(f"[{index}] {title} {article} | 점수={doc.get('score', 0):.4f}")
                    previous = meta.get("before_rerank")
                    if previous:
                        print(f"    RRF {previous['rank']}위 → 재정렬 {index}위")
                    print("   ", " ".join(doc["text"].split())[:160])

            if "is_valid" in result:
                print("생성 진행:", result["is_valid"])
                print("검사 이유:", result.get("validation_reason", ""))
                print("재검색 횟수:", result.get("retry_count", 0))

            if "answer" in result:
                print("답변:")
                print(result["answer"])
            if "generation_status" in result:
                print("생성 상태:", result["generation_status"])
                print(f"모델: {result.get('model_name')} | 생성 시간: {result.get('latency_ms', 0):.0f}ms")
                usage = result.get("token_usage", {})
                print(f"토큰: 입력 {usage.get('input_tokens', 0)} / 출력 {usage.get('output_tokens', 0)}")
                if result.get("output_validation_error") or result.get("error_type"):
                    print("생성 오류:", result.get("output_validation_error") or result.get("error_type"))

            if "citation_validation_status" in result:
                print("인용 검사:", result["citation_validation_status"])
                for source in result.get("sources", []):
                    print(f"[{source['source_id']}] {source.get('source_name') or '출처명 없음'} "
                          f"{source.get('article') or ''}")
                    print("   ", source.get("source_url") or "URL 없음")
                if result.get("validation_errors"):
                    print("인용 검사 안내:", result["validation_errors"])
                print("Judge:", result.get("judge_status", "not_run"))
