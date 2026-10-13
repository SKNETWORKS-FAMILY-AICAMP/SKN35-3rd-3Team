# 현재 RAG 평가 입력 안내

평가 입력은 프로젝트 루트의 `rag_evaluation_cases.review.json`입니다.
이 파일은 `cases` 목록, 컬렉션·임베딩 규격, 실제 근거 및 `label_policy`를 포함합니다.
라벨은 AI가 검토한 참고 라벨이며 사람이 검수한 완전한 정답 집합으로 취급하지 않습니다.

멀티쿼리 없음/있음 비교:

```powershell
python -X utf8 -m tests.rag_multiquery_evaluation --evaluation-file rag_evaluation_cases.review.json
```

같은 실제 RRF 후보의 재정렬 전후 비교:

```powershell
python -X utf8 -m tests.rag_rerank_evaluation
```

자세한 검수·실행 조건은 [재정렬 평가 안내](rag_rerank_evaluation_guide.md)를 참고하세요.
정답 ID는 `document_id::chunk_id`이며, 데이터 변경 시 해당 ID의 존재와 관련성을 다시 확인해야 합니다.
Precision·Recall·Hit·MRR 등은 라벨 기준 검색 지표이며 답변 정확성이나 법률적 정확성을 보증하지 않습니다.

기존 초안 평가 입력과 `src/scripts/`의 이전 검색·재작성·속도 평가 스크립트는 삭제됐습니다.
과거 BM25 결과와 CPU 벤치마크 JSON은 과거 실험 기록이며 현재 품질이나 속도로 사용하지 않습니다.
