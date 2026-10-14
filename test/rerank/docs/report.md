# RAG STEP 4·5 평가

Development document MCC max; ties: document F1 max, question_pool error min, document FPR min. Exact ties prefer higher direction then smallest threshold. Separate threshold per candidate_k.

Checked case splits, source/group/exact-query separation; validation never used in selection. Prior model-training/manual tuning exposure cannot be verified from these files.

미정의 지표는 null/빈 CSV 셀. Grade 0=무관, 1·2=관련. 점수는 원본 그대로 사용.
question_pool은 후보 풀 관련 문서 존재 여부가 정답이며 전체 질문을 분모로 사용한다. question_gold_known_only는 데이터셋 answerability 기준이며 uncertain을 제외한 분모를 명시한다.
정답은 있으나 후보 풀에 없는 질문은 별도 strata로 보고한다. 관련 문서 없이 무관 문서만 통과한 답변도 gold 기준 TP가 될 수 있으므로 두 질문 지표와 문서 오류를 함께 읽어야 한다.

## candidate_k=20
선택 임계값: 0.014364708214998245; 통과 방향: higher; Dev 원점수 범위: [-11.044299125671387, 9.843378067016602]

### STEP 4: development

| 평가 단위 | n | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | MCC | FPR | FNR | 오분기율/오류율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| document | 1120 | 87 | 859 | 23 | 151 | 0.844643 | 0.790909 | 0.365546 | 0.500000 | 0.466623 | 0.026077 | 0.634454 | 0.155357 |
| question_pool | 56 | 38 | 10 | 0 | 8 | 0.857143 | 1.000000 | 0.826087 | 0.904762 | 0.677449 | 0.000000 | 0.173913 | 0.142857 |
| question_gold_known_only | 55 | 38 | 8 | 0 | 9 | 0.836364 | 1.000000 | 0.808511 | 0.894118 | 0.616827 | 0.000000 | 0.191489 | 0.163636 |

- document: {"TP": 87, "TN": 859, "FP": 23, "FN": 151, "n": 1120, "precision": 0.7909090909090909, "recall": 0.36554621848739494, "f1": 0.5, "accuracy": 0.8446428571428571, "mcc": 0.46662286671400305, "fpr": 0.026077097505668934, "fnr": 0.634453781512605, "error_rate": 0.15535714285714286}
- question_pool: {"TP": 38, "TN": 10, "FP": 0, "FN": 8, "n": 56, "precision": 1.0, "recall": 0.8260869565217391, "f1": 0.9047619047619048, "accuracy": 0.8571428571428571, "mcc": 0.6774490372475503, "fpr": 0.0, "fnr": 0.17391304347826086, "error_rate": 0.14285714285714285}
- question_gold_known_only: {"TP": 38, "TN": 8, "FP": 0, "FN": 9, "n": 55, "precision": 1.0, "recall": 0.8085106382978723, "f1": 0.8941176470588236, "accuracy": 0.8363636363636363, "mcc": 0.6168270377448741, "fpr": 0.0, "fnr": 0.19148936170212766, "error_rate": 0.16363636363636364}
- question_strata: {"gold_in_pool": {"TP": 38, "TN": 0, "FP": 0, "FN": 8, "n": 46, "precision": 1.0, "recall": 0.8260869565217391, "f1": 0.9047619047619048, "accuracy": 0.8260869565217391, "mcc": null, "fpr": null, "fnr": 0.17391304347826086, "error_rate": 0.17391304347826086}, "gold_missing_from_pool": {"TP": 0, "TN": 1, "FP": 0, "FN": 0, "n": 1, "precision": null, "recall": null, "f1": null, "accuracy": 1.0, "mcc": null, "fpr": 0.0, "fnr": null, "error_rate": 0.0}, "no_gold_answer": {"TP": 0, "TN": 8, "FP": 0, "FN": 0, "n": 8, "precision": null, "recall": null, "f1": null, "accuracy": 1.0, "mcc": null, "fpr": 0.0, "fnr": null, "error_rate": 0.0}, "unknown_answerability": {"TP": 0, "TN": 1, "FP": 0, "FN": 0, "n": 1, "precision": null, "recall": null, "f1": null, "accuracy": 1.0, "mcc": null, "fpr": 0.0, "fnr": null, "error_rate": 0.0}}
문서 오류 174건, 질문 오류 9건. 전체 사례 및 통과/관련 문서 ID는 evaluation_results.json 참조.
- 질문 실패 유형 gold_missing_from_pool: 1건; 예시 case_id: q045
- 질문 실패 유형 gold_in_pool: 8건; 예시 case_id: q022, q024, q013, q015, q016
- 문서 FP 23건; 예시 q004 / official_guide:kca_damage_redress_procedure::kca_damage_redress_procedure_redress_exclusions_0000 / grade=0 / score=1.1818583011627197. 무관 문서가 임계값을 통과함.
- 문서 FN 151건; 예시 q022 / official_guide:guide_23ffe4c8fd5715cb::guide_23ffe4c8fd5715cb_p0060_body_0000 / grade=2 / score=-6.271462917327881. 관련 문서가 임계값에서 차단됨.
### STEP 5: validation

| 평가 단위 | n | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | MCC | FPR | FNR | 오분기율/오류율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| document | 280 | 32 | 208 | 15 | 25 | 0.857143 | 0.680851 | 0.561404 | 0.615385 | 0.532368 | 0.067265 | 0.438596 | 0.142857 |
| question_pool | 14 | 9 | 3 | 0 | 2 | 0.857143 | 1.000000 | 0.818182 | 0.900000 | 0.700649 | 0.000000 | 0.181818 | 0.142857 |
| question_gold_known_only | 13 | 9 | 2 | 0 | 2 | 0.846154 | 1.000000 | 0.818182 | 0.900000 | 0.639602 | 0.000000 | 0.181818 | 0.153846 |

- document: {"TP": 32, "TN": 208, "FP": 15, "FN": 25, "n": 280, "precision": 0.6808510638297872, "recall": 0.5614035087719298, "f1": 0.6153846153846154, "accuracy": 0.8571428571428571, "mcc": 0.532368124904625, "fpr": 0.06726457399103139, "fnr": 0.43859649122807015, "error_rate": 0.14285714285714285}
- question_pool: {"TP": 9, "TN": 3, "FP": 0, "FN": 2, "n": 14, "precision": 1.0, "recall": 0.8181818181818182, "f1": 0.9, "accuracy": 0.8571428571428571, "mcc": 0.7006490497453707, "fpr": 0.0, "fnr": 0.18181818181818182, "error_rate": 0.14285714285714285}
- question_gold_known_only: {"TP": 9, "TN": 2, "FP": 0, "FN": 2, "n": 13, "precision": 1.0, "recall": 0.8181818181818182, "f1": 0.9, "accuracy": 0.8461538461538461, "mcc": 0.6396021490668313, "fpr": 0.0, "fnr": 0.18181818181818182, "error_rate": 0.15384615384615385}
- question_strata: {"gold_in_pool": {"TP": 9, "TN": 0, "FP": 0, "FN": 2, "n": 11, "precision": 1.0, "recall": 0.8181818181818182, "f1": 0.9, "accuracy": 0.8181818181818182, "mcc": null, "fpr": null, "fnr": 0.18181818181818182, "error_rate": 0.18181818181818182}, "no_gold_answer": {"TP": 0, "TN": 2, "FP": 0, "FN": 0, "n": 2, "precision": null, "recall": null, "f1": null, "accuracy": 1.0, "mcc": null, "fpr": 0.0, "fnr": null, "error_rate": 0.0}, "unknown_answerability": {"TP": 0, "TN": 1, "FP": 0, "FN": 0, "n": 1, "precision": null, "recall": null, "f1": null, "accuracy": 1.0, "mcc": null, "fpr": 0.0, "fnr": null, "error_rate": 0.0}}
문서 오류 40건, 질문 오류 2건. 전체 사례 및 통과/관련 문서 ID는 evaluation_results.json 참조.
- 질문 실패 유형 gold_in_pool: 2건; 예시 case_id: q021, q018
- 문서 FP 15건; 예시 q002 / law:000129_218303_20200526::000129_218303_20200526_art_6_0_chunk_0000 / grade=0 / score=1.4058966636657715. 무관 문서가 임계값을 통과함.
- 문서 FN 25건; 예시 q021 / official_guide:guide_23ffe4c8fd5715cb::guide_23ffe4c8fd5715cb_p0041_body_0000 / grade=1 / score=-6.225472927093506. 관련 문서가 임계값에서 차단됨.
Dev → Validation 지표 차이(Validation - Dev):
- document: {"f1": 0.11538461538461542, "mcc": 0.06574525819062199, "fpr": 0.04118747648536246, "fnr": -0.19585729028453486, "error_rate": -0.012500000000000011}
- question_pool: {"f1": -0.004761904761904745, "mcc": 0.023200012497820333, "fpr": 0.0, "fnr": 0.007905138339920958, "error_rate": 0.0}
- question_gold_known_only: {"f1": 0.00588235294117645, "mcc": 0.022775111321957198, "fpr": 0.0, "fnr": -0.009671179883945835, "error_rate": -0.00979020979020978}
Validation 질문 14개: 질문 한 개가 오분기율을 7.1%p 바꾼다. 소표본 결과를 일반화하지 않는다.

## candidate_k=50
선택 임계값: -2.607175350189209; 통과 방향: higher; Dev 원점수 범위: [-11.044299125671387, 9.843378067016602]

### STEP 4: development

| 평가 단위 | n | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | MCC | FPR | FNR | 오분기율/오류율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| document | 2800 | 198 | 2150 | 185 | 267 | 0.838571 | 0.516971 | 0.425806 | 0.466981 | 0.375346 | 0.079229 | 0.574194 | 0.161429 |
| question_pool | 56 | 43 | 6 | 2 | 5 | 0.875000 | 0.955556 | 0.895833 | 0.924731 | 0.568831 | 0.250000 | 0.104167 | 0.125000 |
| question_gold_known_only | 55 | 43 | 6 | 2 | 4 | 0.890909 | 0.955556 | 0.914894 | 0.934783 | 0.607771 | 0.250000 | 0.085106 | 0.109091 |

- document: {"TP": 198, "TN": 2150, "FP": 185, "FN": 267, "n": 2800, "precision": 0.5169712793733682, "recall": 0.4258064516129032, "f1": 0.4669811320754717, "accuracy": 0.8385714285714285, "mcc": 0.37534648896997486, "fpr": 0.07922912205567452, "fnr": 0.5741935483870968, "error_rate": 0.16142857142857142}
- question_pool: {"TP": 43, "TN": 6, "FP": 2, "FN": 5, "n": 56, "precision": 0.9555555555555556, "recall": 0.8958333333333334, "f1": 0.9247311827956989, "accuracy": 0.875, "mcc": 0.5688312786486196, "fpr": 0.25, "fnr": 0.10416666666666667, "error_rate": 0.125}
- question_gold_known_only: {"TP": 43, "TN": 6, "FP": 2, "FN": 4, "n": 55, "precision": 0.9555555555555556, "recall": 0.9148936170212766, "f1": 0.9347826086956522, "accuracy": 0.8909090909090909, "mcc": 0.607770797907894, "fpr": 0.25, "fnr": 0.0851063829787234, "error_rate": 0.10909090909090909}
- question_strata: {"gold_in_pool": {"TP": 43, "TN": 0, "FP": 0, "FN": 4, "n": 47, "precision": 1.0, "recall": 0.9148936170212766, "f1": 0.9555555555555556, "accuracy": 0.9148936170212766, "mcc": null, "fpr": null, "fnr": 0.0851063829787234, "error_rate": 0.0851063829787234}, "no_gold_answer": {"TP": 0, "TN": 6, "FP": 2, "FN": 0, "n": 8, "precision": 0.0, "recall": null, "f1": 0.0, "accuracy": 0.75, "mcc": null, "fpr": 0.25, "fnr": null, "error_rate": 0.25}, "unknown_answerability": {"TP": 0, "TN": 0, "FP": 0, "FN": 1, "n": 1, "precision": null, "recall": 0.0, "f1": 0.0, "accuracy": 0.0, "mcc": null, "fpr": null, "fnr": 1.0, "error_rate": 1.0}}
문서 오류 452건, 질문 오류 7건. 전체 사례 및 통과/관련 문서 ID는 evaluation_results.json 참조.
- 질문 실패 유형 no_gold_answer: 2건; 예시 case_id: q063, q064
- 질문 실패 유형 gold_in_pool: 4건; 예시 case_id: q022, q024, q013, q016
- 질문 실패 유형 unknown_answerability: 1건; 예시 case_id: q069
- 문서 FP 185건; 예시 q001 / law:008480_282351_20260102::008480_282351_20260102_art_2_0_chunk_0000 / grade=0 / score=-1.7969818115234375. 무관 문서가 임계값을 통과함.
- 문서 FN 267건; 예시 q022 / official_guide:guide_23ffe4c8fd5715cb::guide_23ffe4c8fd5715cb_p0060_body_0000 / grade=2 / score=-6.271462917327881. 관련 문서가 임계값에서 차단됨.
### STEP 5: validation

| 평가 단위 | n | TP | TN | FP | FN | Accuracy | Precision | Recall | F1 | MCC | FPR | FNR | 오분기율/오류율 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| document | 700 | 61 | 527 | 63 | 49 | 0.840000 | 0.491935 | 0.554545 | 0.521368 | 0.426826 | 0.106780 | 0.445455 | 0.160000 |
| question_pool | 14 | 9 | 2 | 1 | 2 | 0.785714 | 0.900000 | 0.818182 | 0.857143 | 0.440386 | 0.333333 | 0.181818 | 0.214286 |
| question_gold_known_only | 13 | 9 | 1 | 1 | 2 | 0.769231 | 0.900000 | 0.818182 | 0.857143 | 0.272475 | 0.500000 | 0.181818 | 0.230769 |

- document: {"TP": 61, "TN": 527, "FP": 63, "FN": 49, "n": 700, "precision": 0.49193548387096775, "recall": 0.5545454545454546, "f1": 0.5213675213675214, "accuracy": 0.84, "mcc": 0.42682630542600974, "fpr": 0.10677966101694915, "fnr": 0.44545454545454544, "error_rate": 0.16}
- question_pool: {"TP": 9, "TN": 2, "FP": 1, "FN": 2, "n": 14, "precision": 0.9, "recall": 0.8181818181818182, "f1": 0.8571428571428571, "accuracy": 0.7857142857142857, "mcc": 0.4403855060505443, "fpr": 0.3333333333333333, "fnr": 0.18181818181818182, "error_rate": 0.21428571428571427}
- question_gold_known_only: {"TP": 9, "TN": 1, "FP": 1, "FN": 2, "n": 13, "precision": 0.9, "recall": 0.8181818181818182, "f1": 0.8571428571428571, "accuracy": 0.7692307692307693, "mcc": 0.27247463045653303, "fpr": 0.5, "fnr": 0.18181818181818182, "error_rate": 0.23076923076923078}
- question_strata: {"gold_in_pool": {"TP": 9, "TN": 0, "FP": 0, "FN": 2, "n": 11, "precision": 1.0, "recall": 0.8181818181818182, "f1": 0.9, "accuracy": 0.8181818181818182, "mcc": null, "fpr": null, "fnr": 0.18181818181818182, "error_rate": 0.18181818181818182}, "no_gold_answer": {"TP": 0, "TN": 1, "FP": 1, "FN": 0, "n": 2, "precision": 0.0, "recall": null, "f1": 0.0, "accuracy": 0.5, "mcc": null, "fpr": 0.5, "fnr": null, "error_rate": 0.5}, "unknown_answerability": {"TP": 0, "TN": 1, "FP": 0, "FN": 0, "n": 1, "precision": null, "recall": null, "f1": null, "accuracy": 1.0, "mcc": null, "fpr": 0.0, "fnr": null, "error_rate": 0.0}}
문서 오류 112건, 질문 오류 3건. 전체 사례 및 통과/관련 문서 ID는 evaluation_results.json 참조.
- 질문 실패 유형 no_gold_answer: 1건; 예시 case_id: q060
- 질문 실패 유형 gold_in_pool: 2건; 예시 case_id: q021, q018
- 문서 FP 63건; 예시 q002 / law:005247_206564_20190101::005247_206564_20190101_art_11_0_chunk_0000 / grade=0 / score=-1.4705958366394043. 무관 문서가 임계값을 통과함.
- 문서 FN 49건; 예시 q021 / official_guide:guide_23ffe4c8fd5715cb::guide_23ffe4c8fd5715cb_p0047_body_0000 / grade=2 / score=-6.665560245513916. 관련 문서가 임계값에서 차단됨.
Dev → Validation 지표 차이(Validation - Dev):
- document: {"f1": 0.0543863892920497, "mcc": 0.05147981645603489, "fpr": 0.02755053896127463, "fnr": -0.12873900293255136, "error_rate": -0.001428571428571418}
- question_pool: {"f1": -0.06758832565284179, "mcc": -0.12844577259807533, "fpr": 0.08333333333333331, "fnr": 0.07765151515151515, "error_rate": 0.08928571428571427}
- question_gold_known_only: {"f1": -0.07763975155279512, "mcc": -0.335296167451361, "fpr": 0.25, "fnr": 0.09671179883945842, "error_rate": 0.1216783216783217}
Validation 질문 14개: 질문 한 개가 오분기율을 7.1%p 바꾼다. 소표본 결과를 일반화하지 않는다.

라벨은 데이터셋의 검토 수준에 종속된다. 문서 공유 자체는 누수로 단정하지 않으며, 의미적 중복 및 이전 Validation 노출 여부는 별도 검토가 필요하다.