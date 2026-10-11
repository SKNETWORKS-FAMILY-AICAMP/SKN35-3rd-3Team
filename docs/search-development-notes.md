# 검색 개발 기록

## 연결 기준
- 2번 담당자 전달서의 jb 커밋 68fe666에서 공용 설정, Qdrant 어댑터와 의존 모듈을 새 파일로 가져옴.
- 기존 src/embedder.py는 유지. 제공자 코드는 src/vector_db/embedding_provider.py로 별도 배치.
- 기존 Dense, BM25, Hybrid, rrf(state), multi_query(text) 함수를 유지.
- 의존성 추가는 uv add만 사용. qdrant-client 추가 및 uv.lock 갱신.
- .env 키값을 출력하거나 수정하지 않음. 원격 DB에 쓰기/적재/삭제 요청 없음.
- 팀장 State와 그래프는 이번 작업에서 변경하지 않음.

## 검색 흐름
원본 질문 → LLMQueryExpander(최대 2개 추가) → 원본 포함 최대 3개 표현
→ 표현별 Hybrid(Dense + BM25 + RRF) → 질문 표현 간 RRF → 최종 청크 목록.
Judge/재생성은 팀장 흐름에서 담당. Rerank는 아직 구현하지 않음.

## 실제 연결
원격 OpenAI 컬렉션 team_documents_openai_dev의 규격 확인 성공:
1536차원, cosine, green, Point 수 0.
실제 청크가 없으므로 실제 Dense/Hybrid 검색 품질과 실제 LLM 확장 효과는 미평가.
Multi-Query 모델은 MULTI_QUERY_MODEL 또는 CLI 인자로 실제 사용 가능한 모델을 지정.
팀장 config의 모델 이름을 검증 없이 사용하지 않음.

## 기능 검증용 비교 (실제 품질 아님)
실제 로컬 메모리 Qdrant + 예시 청크 3개 + 질문 2개, 고정 2차원 벡터,
고정 확장 표현으로 K=3 평가. API 임베딩/LLM 호출을 대체한 기능 검증임.

| 방법 | Recall@3 | Precision@3 | Hit Rate@3 | MRR@3 |
| --- | ---: | ---: | ---: | ---: |
| Dense | 1.0 | 0.3333 | 1.0 | 1.0 |
| BM25 | 1.0 | 0.3333 | 1.0 | 1.0 |
| Hybrid | 1.0 | 0.3333 | 1.0 | 1.0 |
| Multi-Query Hybrid | 1.0 | 0.3333 | 1.0 | 1.0 |

질문당 정답이 1개라 Precision@3의 최고값은 1/3.
이 표로 방법 간 우열이나 실제 법률 서비스 성능을 판단하면 안 됨.

## 재실행
`uv run python -B -m unittest discover -s tests -v`
`uv run python -B -m src.Scripts.search_evaluate --check`

실제 평가 데이터:
`[{"query":"실제 평가 질문", "relevant_ids":["document_id::chunk_id"], "filters":{}}]`
relevant_ids는 실제 corpus에서 사람이 관련성을 판단한 청크 ID를 사용.
정답 없는 질문은 별도 평가. 문서 ID와 청크 ID 수준을 섞지 않음.

실제 네 방식 비교:
`uv run python -B -m src.Scripts.search_evaluate --evaluation-file data/evaluation/questions.json --multi-query-model 실제_사용가능한_Gemini_모델 --k 5 --output data/evaluation/results.json`
결과에는 질문별 검색 ID, 지표, 호출 시간이 포함됨.
K=5/10/20은 같은 질문과 정답 데이터로 비교. 유사도 점수를 Precision으로 사용하지 않음.

## 남은 작업
1. 실제 데이터 적재 후 검색 검증 및 정답 라벨 데이터 준비.
2. 실제 API/LLM 기반 네 방식 비교.
3. 한국어 형태소 분석 tokenizer 비교(현재 기본은 문자열 단위 토큰화).
4. 구현한 Rerank를 실제 정답 데이터로 동일 조건 평가.
5. 팀장과 State.documents 타입 및 검색 호출 연결.

## 무료 CPU Rerank 구현 및 실측
- 기존 src/rerank.py의 rerank(state)를 유지하고 CrossEncoderScorer와 RerankingRetriever 추가.
- 모델: BAAI/bge-reranker-v2-m3. 유료 API 없이 CPU에서 실행.
- uv add --optional rerank sentence-transformers로 패키지 등록.
- 모델은 프로젝트 .cache/rerank에 캐시하고 .gitignore에 .cache/ 추가.
- 후보 검색 → 원본 질문/청크 쌍 점수 → 재정렬 → 상위 K개 반환.
- 기존 검색 점수/순위/RRF 증거/출처 보존. Judge/재생성 분기는 처리하지 않음.
- 평가 CLI의 --rerank로 Hybrid와 Multi-Query Hybrid 각각 적용 전후 비교 가능.
- --rerank-candidates 및 --rerank-max-length로 후보 수/입력 길이 조정 가능.
- tests/test_rerank.py는 새로운 검증 파일. src/Scripts/rerank_benchmark.py는 CPU 실측용 새 파일.

실측: Intel Core Ultra 7 155H, CPU 스레드 4, batch_size=1, max_length=512, 후보 10개.
초기 모델 다운로드/로딩 65.315초. 짧은 입력 3.285초 및 2.509초,
긴 입력(예시 문장 20회 반복) 16.923초. 프로세스 작업 메모리 약 2923MiB.
로컬에서 실행 가능하지만 긴 입력에는 지연이 큼. 실제 법률 청크의 품질/지연은 미평가.
점수는 법적 정확성이나 정답 확률을 의미하지 않음.
실측 원본: docs/rerank-cpu-benchmark.json. 총 13개 자동 검증 통과.
