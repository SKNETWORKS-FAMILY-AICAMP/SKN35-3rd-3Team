# STEP 4·5 실행

STEP 4는 Development에서 임계값과 통과 방향을 선택하고 평가한다. STEP 5는 그 임계값과 방향을 그대로 Validation에 적용한다. Validation 결과를 보고 재선정하지 않으며 candidate_k별 결과를 모두 보고한다.

문서 단위 정답은 grade 1·2=양성, grade 0=음성이고 예측 양성은 임계값 통과다. TP=관련 문서 통과, TN=무관 문서 차단, FP=무관 문서 통과, FN=관련 문서 차단이다. 평가 단위는 점수 파일에 있는 질문-후보 문서 쌍이며 전체 코퍼스 문서가 아니다.

두 STEP에서 다음 지표를 모두 계산한다.

| 지표 | 계산식 |
|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) |
| Precision | TP/(TP+FP) |
| Recall | TP/(TP+FN) |
| F1 | 2TP/(2TP+FP+FN) |
| MCC | (TP×TN−FP×FN)/√((TP+FP)(TP+FN)(TN+FP)(TN+FN)) |
| FPR | FP/(FP+TN) |
| FNR | FN/(FN+TP) |

질문 단위 예측은 문서가 하나라도 통과하면 답변 진행, 모두 차단되면 근거 부족이다. 분기 정확도는 (질문 TP+TN)/평가 질문 수, 오분기율은 (질문 FP+FN)/평가 질문 수다. 문서 지표와 별도로 pool 기준 및 answerability 기준을 각각 계산한다. 질문 FP는 답변 진행으로 잘못 분기, 질문 FN은 근거 부족으로 잘못 분기한 경우다.

검색·리랭크 추론 없이 원점수 파일만 읽는다. STEP 2·3 코드는 수정하지 않는다.

```powershell
python -m unittest discover -s test/rerank -p test_threshold_evaluation.py -v
python test/rerank/threshold_evaluation.py
```

입력은 JSON 문서 배열 또는 `documents`, `scores`, `results`, `records` 중 하나가 문서 배열인 객체다. 각 행은 사용자 지정 8개 필드를 모두 가져야 한다. 중첩된 질문별 결과는 평탄화해서 전달한다. `evaluation_split`은 `development`/`validation`, rank는 1부터 시작한다. `candidate_k`별로 별도 평가하며 모든 데이터셋 질문의 점수가 있어야 한다. 후보가 0개인 질문은 최소 문서 필드만으로 누락과 구분할 수 없어 현재 명시적 입력 오류로 처리한다. 후보 수가 k보다 적은 것은 허용한다.

Development의 모든 고유 원점수와 전부 차단하는 경계를 탐색한다. 두 통과 방향을 Development MCC로 비교하며 동률이면 기본 `score >= threshold` 방향을 유지한다. 이는 경험적 방향 검사이고 모델 명세 자체의 확인은 아니다. 문서 MCC 최대 → 문서 F1 최대 → 후보 풀 기준 질문 오분기율 최소 → 문서 FPR 최소 순으로 선택한다. 최종 동률은 작은 임계값을 선택한다. MCC는 두 클래스 불균형을 고려하려는 사전 지정 기준이며 서비스별 비용 최적화는 별도 정책이 필요하다. Validation으로 방향·임계값·candidate_k를 고르지 않는다.

기본 입력 점수는 `test/rerank/docs/rerank_scores.json`이며, 출력은 모두 `test/rerank/docs/`에 저장된다. 폴더가 없으면 자동 생성한다. `--scores`, `--output`으로 경로를 변경할 수 있다.

출력:

- `threshold_comparison.csv`: Development 임계값·방향별 문서/질문 지표
- `split_metrics.csv`: STEP 4/5별 고정 임계값, 문서 Confusion Matrix와 전체 지표, 질문 분기 정확도(`accuracy`)와 오분기율(`error_rate`), 평가 분모(`n`)
- `evaluation_results.json`: 고정 임계값, 입력 해시, 두 split 지표, 문서 FP/FN 및 질문 오류 전체 사례
- `report.md`: 선정 기준, 점수 범위, Dev/Validation 비교, 누수 검사 범위와 해석 한계

질문 분기는 후보 중 하나라도 통과하면 답변 진행이다. `question_pool`은 후보에 관련 문서가 있는지를 정답으로 삼으며 전체 질문을 분모로 사용한다. `question_gold_known_only`는 데이터셋 answerability 기준이고 uncertain은 제외하므로 `n`을 반드시 함께 읽는다. `question_strata`는 정답 없음/정답 후보 존재/정답 후보 누락/정답 불명으로 구분한다. 정답 후보 누락 상태에서 전부 차단한 것은 pool 기준 TN, gold 기준 FN이다. 무관 문서만 통과하면 근거가 부족해도 gold 기준 TP일 수 있다.

0인 분모의 지표는 JSON null/CSV 빈 셀이다. 작은 수작업 fixture는 임시 디렉터리에서만 평가되며 실제 산출물에 섞이지 않는다. 점수 파일 부재 시 종료 코드 2와 WAITING을 출력하고 실제 결과를 생성하지 않는다. 파일에 없는 모델 학습 데이터·사전 Validation 노출은 누수 없음을 보증할 수 없다.
