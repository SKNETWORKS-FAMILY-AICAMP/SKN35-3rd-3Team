# 실제 RAG 단계별 평가

입력은 루트의 `rag_evaluation_cases_list.json`입니다. 기존 `data/`는 과거 40개 평가 자료이며 사용하지 않습니다. 데이터 생성 스크립트도 필요하지 않습니다.
원본 JSON은 수정하지 않고 `relevance_labels`를 메모리에서 0/1/2 qrels로 변환합니다. 양수 라벨과 `reference_relevant_ids`의 일치를 검증합니다.
관련도 1/2는 Hit/Recall/MRR의 정답이고, nDCG는 `2^grade-1` 이득을 사용합니다.
AI provisional 라벨을 허용하며 결과에 잠정 평가임을 표시합니다. 사람 검수 완료로 취급하지 않습니다.
정답 없는 unanswerable/uncertain 질문은 품질 평균에서 제외합니다. 오류를 0점이나 더미 데이터로 대체하지 않습니다.

## 빠른 전체 실행

실제 src 구현, Qdrant, LLM, 리랭커를 사용합니다. 개별 파일의 단계 함수를 공유 실행하여 중복 검색을 줄입니다.

```powershell
.venv/Scripts/python.exe -m evaluation.run --stage all --stages query_rewriting multi_query hybrid rrf reranking --config test/rag_test/fast_config.json --allow-ai-reviewed --output reports/evaluation/direct_70_fast
.venv/Scripts/python.exe -m test.rag_test.report_results reports/evaluation/direct_70_fast/evaluation_results.json --output-root reports/evaluation/direct_70_fast/stages
```

전체 70개, 후보 5개, 워밍업 없음, 측정 1회, CPU 리랭커 배치 1입니다. K=1/3/5별 Hit@K, Recall@K, MRR@K, nDCG@K 및 Precision과 시간을 CSV/JSON/Markdown으로 저장합니다.
측정 1회는 반복 실행 변동성을 평가하지 않습니다. 실행 시간은 외부 서비스와 CPU에 좌우됩니다.

| 파일 | 비교 |
|---|---|
| query_rewriting_test.py | 원본 / 재작성 질문 |
| multi_query_test.py | 재작성 단일 / 확장 질문 검색 및 융합 |
| hybrid_search_test.py | Dense / Sparse / Hybrid |
| rrf_test.py | 재작성 단일 / 확장 후보 검색 및 RRF (순위만의 효과는 아님) |
| rerank_test.py | 동일 RRF 후보의 재정렬 전 / 후 |

각 파일은 `.venv/Scripts/python.exe -m test.rag_test.<파일명> --config test/rag_test/fast_config.json`으로 실행합니다.
기본 config.json / rerank_config.json은 후보 20/50, 3회 반복 설정입니다. --evaluation-file 및 --output으로 입력과 출력을 변경합니다.
--check는 입력 검사만 수행하며 실제 평가 성공을 뜻하지 않습니다. 미검수 결과는 unknown이고 판단 커버리지를 함께 보고합니다.

## 동일한 20개 질문으로 실행 및 단계별 보고서 자동 생성

```powershell
.venv/Scripts/python.exe -m test.rag_test.run_fast
```

JSON 순서대로 첫 20개 질문을 사용합니다. `--limit`으로 개수를 바꾸고 `--resume`으로 같은 조건의 완료 체크포인트를 재사용합니다.
20개 진단은 전체 70개 결과와 구분하여 표시합니다. 기본 출력은 reports/evaluation/direct_20_fast_c5이며 stages/에 다섯 단계 보고서를 저장합니다.
