# jem 검색 변경을 dev에 통합할 때

검증 기준: dev e1fe7ca + 현재 jem 검색 수정본.
다른 팀원의 코드/데이터와 옛 폴더 연결을 jem에 다시 복사하지 않았습니다.

## 이번 변경

- DocumentRetriever가 공용 bootstrap의 expected_dimension, sparse_embedder 인자를 수용합니다.
- Sparse 요청과 문서 ID 조회 인터페이스를 추가했습니다. 기존 Dense, BM25, Hybrid 기능은 유지합니다.
- vectorstore의 요청, 조회 문서, 컬렉션 정보에 dev 호환 필드를 추가했습니다.
- 검색 평가 bootstrap은 dev 공용 src.embedder를 사용합니다. 평가의 Dense 경로는 서버 Hybrid와 중복되지 않도록 객체 설정만 Dense로 복사합니다. 서버 설정/.env는 바꾸지 않습니다.
- 리라이팅 평가의 상대 import는 Scripts/scripts 두 패키지명에서 모두 동작합니다.
- jem에는 공용 DB 코드가 없으므로 연동 테스트 3개만 명시적으로 건너뜁니다. dev에서는 17개 모두 실행합니다.

## 실제 검증 결과

- jem: 17개 중 14개 통과, 공용 DB 연동 3개 skipped.
- 별도 dev 검증 폴더: 검색 테스트 17개 모두 OK, 공용 bootstrap/검색 평가/리라이팅 평가 import OK.
- 통합 코드의 실제 로컬 청크 4,932개 BM25 검색 실행 확인.
- 실제 임베딩/LLM API를 호출하는 전체 검색 품질 및 LangGraph 앱 실행은 검증하지 않았습니다.
- 별도 dev 환경 설치 중 Windows MCP 실행파일 설치가 거부됐습니다. 검색 검증 환경에서는 MCP 설치만 제외했으며 공용 pyproject의 MCP 요구사항은 삭제하지 않았습니다.

## 머지 충돌 처리

Git 충돌이 전혀 없다는 보장은 아닙니다. 아래 파일은 양쪽 변경이 겹칩니다.

- src/retriever.py: 새 jem의 expected_dimension, sparse_embedder, lookup_document 지원과 HybridRetriever를 유지해야 합니다. 최신 dev의 추가 변경도 비교하세요.
- src/vectorstore.py: VectorDocument, query_text, sparse_indices/values, search_mode, sparse_vector_name, lookup이 반드시 남아야 합니다. 예전 placeholder로 되돌리지 않습니다.
- src/vector_db/search_bootstrap.py: 이번 jem 수정의 src.embedder 경로와 Dense-only 평가 객체 설정을 유지합니다.
- src/Scripts의 변경은 dev에서 src/scripts 경로에 반영합니다. 같은 파일을 두 폴더에 중복 생성하지 않습니다.
- pyproject.toml: dev의 기존 의존성/빌드 설정/파이썬 버전 범위를 유지하고 rerank 선택 의존성을 추가합니다. jem 전체 파일로 덮어쓰지 않습니다.
- uv.lock: 문자열로 충돌 부분을 합치지 말고, 합쳐진 pyproject 기준으로 uv add로 재생성합니다.

정상적인 lock이라면 아래로 rerank를 등록합니다.

```powershell
uv add --optional rerank "sentence-transformers>=6.1.0"
```

lock에 충돌 표시가 있거나 파싱 불가능한 경우, dev 프로젝트 루트에서 원본을 보관한 뒤 실행합니다. 백업 파일이 이미 있다면 다른 이름을 사용하세요.

```powershell
Move-Item -LiteralPath .\uv.lock -Destination .\uv.lock.before-search-merge
uv add --optional rerank "sentence-transformers>=6.1.0"
```

정리한 뒤 통합 dev에서 확인:

```powershell
uv sync --locked --extra rerank
uv run --extra rerank python -B -X utf8 -m unittest discover -s tests -p "test_*.py"
uv run --extra rerank python -B -X utf8 -m src.scripts.try_real_search --query "보증금 반환" --k 5
```

테스트는 skipped 없이 17개 OK여야 합니다. API 키는 각자 프로젝트 .env에서 준비하며 Git에 올리지 않습니다.
아직 커밋이나 푸시는 하지 않았습니다. .cache의 검증 폴더는 Git ignore 대상입니다.
