# SKN35-3rd-3Team

## 데이터 스크립트와 검증

`src/Scripts/`에는 수집·전처리·청크 실행 스크립트를, `src/Scripts/tests/`에는 `test_*_pmh.py` 검증 파일을 보관합니다. 아래 명령은 프로젝트 루트에서 실행합니다.

전체 테스트:

```bash
uv run python -m unittest discover -s src/Scripts/tests -t . -p "test*_pmh.py"
```

개별 테스트 예시:

```bash
uv run python -m unittest src.Scripts.tests.test_preprocess_guides_pmh
```

검색 평가 보고서 생성·갱신(BM25 기반이며 임베딩·LLM 평가가 아닙니다):

```bash
uv run python -m src.Scripts.tests.test_precedent_retrieval_pmh
uv run python -m src.Scripts.tests.test_consumer_coverage_pmh
```

검색 평가 모듈은 직접 실행하면 보고서를 갱신하고, 전체 테스트에서는 저장된 데이터와 결과를 검증합니다. 소비자 검색 평가는 같은 테스트 패키지의 판례 평가 모듈에서 BM25 코드를 가져옵니다.