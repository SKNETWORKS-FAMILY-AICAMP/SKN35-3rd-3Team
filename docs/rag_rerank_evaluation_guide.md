# 실제 리랭크 평가

평가 코드는 `tests/rag_rerank_evaluation.py` 한 파일입니다. 실제 검색·RRF에서 한 번 얻은 **동일 후보의 원래 순위 / 실제 rerank 후 순위**만 비교합니다.

## 프로젝트 규칙

- 입력·보고서·모델 캐시 경로: `src/const/config.py`
- 모델 생성: `src/const/model.py`의 기존 `create_llm()`과 `create_rerank_model()`
- 모델명: `.env`의 `RERANK_MODEL`, `OPENAI_LLM_MODEL` 또는 `LLM_MODEL`. 평가 코드에서 하드코딩하지 않습니다.
- 새 env_loader·클래스·프롬프트 파일은 만들지 않았습니다. 기존 프롬프트를 재사용합니다.
- 함수 설명 주석과 내부 주석을 작성하고, 평가 결과를 터미널에 출력한 뒤 `return response`로 반환합니다.
- 맨 아래 `if __name__ == '__main__':`로 실행합니다.

## 데이터 검수

공통 입력은 `rag_evaluation_cases.review.json`입니다. RAG 각 단계에서 같은 컬렉션·데이터 버전·필터 조건에 공통으로 사용할 수 있습니다. 질문 20개의 정답 후보·실제 본문·조문·출처·법령 버전·검수 상태를 보존했습니다.

사용자 요청에 따라 Codex(LLM)가 질문과 실제 근거의 관련성을 검수합니다. 완료한 라벨에는 `review_status=reviewed`, `review_type=ai`, `reviewed_by=Codex (LLM)`과 관련 이유를 기록합니다. 사람 검수로 표시하지 않습니다. AI 검수 라벨로 계산하는 지표는 AI가 확인한 정답 근거 기준의 결과이며, 전체 DB의 모든 관련 문서를 빠짐없이 라벨링한 결과로 간주하지 않습니다. 검수 방식과 범위는 데이터셋과 보고서에 함께 기록됩니다.

평가 스크립트는 데이터셋 생성 기능을 포함하지 않습니다. 공통 검수 데이터를 입력으로 사용합니다. 품질 평가 시 실제 Qdrant 본문·벡터 버전과 정답 근거·저장 임베딩 메타데이터를 다시 확인합니다. 데이터 변경·정답 누락·연결 실패는 오류로 중단하며 더미 데이터로 대체하지 않습니다.

## 실행

검수 완료 후 프로젝트 루트에서 실행합니다.

```powershell
uv run --extra rerank python -B tests/rag_rerank_evaluation.py
```

후보 20개, 최종 10개가 기본입니다. `--candidate-k`, `--final-k`, `--evaluation-file`로 변경할 수 있고 @10 계산을 위해 final-k는 10 이상입니다.

정답 라벨 없이 실제 연결·리랭크 시간만 확인하려면 다음을 실행합니다. 품질 점수는 계산하지 않습니다.

```powershell
uv run --extra rerank python -B tests/rag_rerank_evaluation.py --smoke-test
```

앞단은 기존 재작성 → 멀티쿼리 → 설정된 실제 검색 → RRF를 한 번 실행합니다. 그 기법들 자체의 우열은 평가하지 않습니다. 실제 `src/rerank.py`를 사용하고, 후보 ID·본문·입력 해시와 반환 개수를 검사합니다. DB에는 쓰지 않습니다.

## Markdown 출력과 재현

평가 실행 시 출력 파일은 **`docs/rag_rerank_evaluation/rag_rerank_report.md` 하나**입니다. CSV·별도 결과 JSON·캐시 JSON·중간 결과·실행 상태 파일은 생성하지 않습니다. 과거 실행 기록과 시간 확인용 보고서는 정리했으며, 다음 실행에서 보고서를 생성합니다.

보고서에는 Recall@5/@10, Precision@5, MRR@10, 이진 nDCG@10, Hit Rate@5, 평균·P95 지연, 전후 변화율, 질문별 개선/악화·실패 관찰을 담습니다. 초기 로딩·실제 후보 준비 실행·순수 리랭크 처리·predict 시간을 분리합니다. predict는 처리 시간의 일부이므로 합산하지 않습니다. P95는 질문별 1회 측정값의 선형 보간입니다.

같은 모델·temperature=0·프롬프트로 생성한 질의, 실제 후보 본문·메타데이터, 데이터 해시와 상세 결과도 보고서 안의 JSON 코드 블록에 기록합니다. 별도 JSON 파일은 없습니다. 이전 보고서의 질의를 재사용하려면:

```powershell
uv run --extra rerank python -B tests/rag_rerank_evaluation.py --query-report docs/rag_rerank_evaluation/rag_rerank_report.md --output docs/rag_rerank_evaluation/rag_rerank_replay.md
```

데이터·설정·라벨 버전이 다르면 질의 재사용을 거부합니다. 실제 후보는 다시 검색하되 한 실행의 전후 조건은 같은 후보를 공유합니다. 재사용 생성 시간은 최초 실제 API 호출 측정값입니다.

평가 함수는 요약을 터미널에 출력하고 결과 딕셔너리를 반환합니다. 실행 성공과 품질 개선은 별개이며, 미검수 라벨로 품질 보고서를 생성하지 않습니다. 실패 원인과 효과가 미미한 이유를 수치만으로 확정하지 않습니다.
