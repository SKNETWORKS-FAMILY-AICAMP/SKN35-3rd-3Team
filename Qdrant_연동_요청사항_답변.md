# Qdrant 연동 요청사항 답변

- 작성 기준일: 2026-10-11
- 기준 브랜치: `jb`
- Vector DB 기준 커밋: `6b04953f96b4445e8a9a9d9450eddde53d4c9732`
- 현재 사용 프로필: NVIDIA Embedding

## 현재 상태

원격 Qdrant 연결과 NVIDIA 전용 컬렉션 생성은 완료됐습니다. NVIDIA
`nvidia/nemotron-3-embed-1b` 모델의 실제 2048차원 임베딩 호출도
검증했습니다. 다만 전처리·청킹 데이터는 아직 적재되지 않아 실제 검색 품질은
검증하지 못한 상태입니다.

| 컬렉션 | 임베딩 모델 | 차원 | Point 수 | 용도 |
| --- | --- | ---: | ---: | --- |
| `team_documents_nvidia_dev` | `nvidia/nemotron-3-embed-1b` | 2048 | 0 | 현재 사용 |
| `team_documents_openai_dev` | `text-embedding-3-small` | 1536 | 0 | 미사용 대체 프로필 |

두 컬렉션은 실제 원격 Qdrant에서 존재 여부와 Point 수를 확인했습니다.

## 1. 연결 코드·브랜치·컬렉션·접속 설정

### 브랜치와 주요 파일

- 브랜치: `jb`
- Qdrant Client와 적재·검색: `src/vector_db/qdrant_store.py`
- 환경변수와 프로필 선택: `src/const/qdrant_config.py`
- 모델·차원·컬렉션 기본값: `src/const/vector_db_defaults.py`
- 검색 계층 조립: `src/vector_db/bootstrap.py`
- 질문 검색 서비스: `src/retriever.py`
- 문서·Payload 계약: `src/const/vector_data_contract.py`
- 문서 벡터 생성: `src/vector_db/ingestion.py`
- 임베딩 제공자: `src/embedder.py`

현재 NVIDIA 컬렉션 규격:

- 컬렉션: `team_documents_nvidia_dev`
- 벡터 차원: 2048
- 거리 함수: Cosine
- Named Vector: 사용하지 않음

### 원격 접속 설정

```text
https://31ff3cbb-4b6c-42a7-b0c1-09a7f3d59d65.us-west-1-0.aws.cloud.qdrant.io
```

각 사용자는 프로젝트 루트의 개인 `.env`에 다음 값을 설정합니다. 실제 API
키는 Git이나 문서에 기록하지 않습니다.

```dotenv
QDRANT_URL=https://31ff3cbb-4b6c-42a7-b0c1-09a7f3d59d65.us-west-1-0.aws.cloud.qdrant.io
QDRANT_API_KEY=개인_Database_API_Key
NVIDIA_API_KEY=개인_NVIDIA_API_Key
EMBEDDING_PROVIDER=nvidia
```

- 적재 담당자는 Qdrant `Manage/Write` 권한이 필요합니다.
- 검색만 수행하는 앱·MCP는 가능하면 `Read-Only` 키를 사용합니다.
- `.env`는 `.gitignore`에 등록되어 Git에 올라가지 않습니다.

연결과 컬렉션 규격 확인:

```powershell
uv run python -m src.Scripts.qdrant_setup check
```

## 2. 임베딩 모델과 질문 임베딩 함수

| 항목 | 값 |
| --- | --- |
| 제공자 | NVIDIA API |
| 모델 | `nvidia/nemotron-3-embed-1b` |
| 벡터 차원 | 2048 |
| 문서 입력 유형 | `passage` |
| 질문 입력 유형 | `query` |

문서 벡터는 `generate_document_vectors()`가 내부적으로
`embed_documents()`를 호출해 생성합니다. 질문 벡터는
`NvidiaEmbedder.embed_query()`를 사용합니다.

```python
from src.const.env_loader import load_project_env
from src.const.qdrant_config import load_qdrant_settings
from src.embedder import create_embedding_provider

load_project_env()
settings = load_qdrant_settings()
embedding_provider = create_embedding_provider(settings.embedding)
query_vector = embedding_provider.embed_query("질문 내용")
```

일반적인 검색에서는 직접 벡터를 만들기보다 검색기를 사용합니다.

```python
from src.vector_db import create_team_retriever

retriever = create_team_retriever()
results = retriever.search_documents(
    "계약갱신요구권은 언제 행사할 수 있나요?",
    top_k=5,
    filters={"document_type": "law"},
)
```

문서 적재와 질문 검색에는 반드시 같은 NVIDIA 모델과 NVIDIA 컬렉션을
사용해야 합니다.

## 3. 문서 Payload 구조와 예시

```json
{
  "id": "document_id와 chunk_id로 만든 안정적 UUID",
  "vector": [0.0123, -0.0456],
  "payload": {
    "text": "검색 대상이 되는 청크 본문",
    "document_id": "law-001",
    "chunk_id": "law-001-0001",
    "content_hash": "청크 본문 SHA-256",
    "indexed_at": "2026-10-11T00:00:00+00:00",
    "domain": "주택임대차",
    "document_type": "law",
    "source_name": "주택임대차보호법.pdf",
    "source_url": "https://example.go.kr/law/001",
    "source_path": "https://example.go.kr/law/001",
    "page": 12,
    "section": "계약갱신 요구",
    "article_label": "제6조의3",
    "case_number": "2024다12345",
    "effective_date": "2024-01-01",
    "embedding_provider": "nvidia",
    "embedding_model": "nvidia/nemotron-3-embed-1b",
    "embedding_dimension": 2048,
    "embedding_normalized": true
  }
}
```

필수 입력:

- `document_id`: 원본 문서 식별자
- `chunk_id`: 문서 내부 청크 식별자
- `text`: 검색·임베딩 대상 본문

표준 메타데이터에는 `domain`, `document_type`, `source_name`,
`source_url`, `page`, `section`, `article_label`, `effective_date` 등이
포함됩니다. `case_number`는 현재 전용 필드는 아니지만 추가 메타데이터로
입력하면 Payload에 보존되고 필터에도 사용할 수 있습니다.

## 4. 문서 종류별 컬렉션과 필터 검색

법령·안내서·판례는 임베딩 모델이 같다면 같은 NVIDIA 컬렉션에 저장합니다.
자료 종류별 컬렉션을 나누지 않고 `document_type`으로 구분합니다.

- 법령: `law`
- 안내서·공식 지침: `official_guide`
- 판례: `precedent`

현재 검색 코드는 문자열·정수·불리언 Payload의 정확 일치 필터를 지원합니다.

```python
results = retriever.search_documents(
    "임차인의 계약갱신 요구 조건",
    top_k=5,
    filters={
        "document_type": "law",
        "article_label": "제6조의3",
    },
)
```

`case_number`가 Payload에 저장돼 있다면 같은 방식으로 필터를 전달할 수
있습니다. 대량 데이터에서는 자주 쓰는 필터 필드에 Qdrant Payload Index를
추가하는 작업이 추후 필요할 수 있습니다.

## 5. Hybrid 키워드 검색 제공 여부

현재 구현은 NVIDIA Dense Vector 의미 검색만 제공합니다.

아직 구현되지 않은 기능:

- BM25 또는 Sparse Vector 키워드 검색
- Dense와 키워드 결과를 합치는 RRF(Reciprocal Rank Fusion)
- Hybrid 검색용 Sparse Vector 컬렉션 규격

따라서 Hybrid 검색이 프로젝트 필수라면 별도 구현이 필요합니다. 구현 전
Sparse Vector 규격과 RRF 결합 방식을 협의해야 하며, 기존 Dense 컬렉션을
삭제하거나 임의로 재생성해서는 안 됩니다.

## 6. 실제 테스트 데이터와 질문 예시

2026-10-11 확인 결과:

- `team_documents_nvidia_dev`: 0 Point
- `team_documents_openai_dev`: 0 Point

현재 실제 검색 가능한 문서는 없습니다. 연결·컬렉션·NVIDIA 임베딩 출력은
검증했지만 검색 정확도는 아직 검증하지 못했습니다.

전처리 데이터 적재 후 확인 질문 예시:

1. `계약갱신요구권은 언제 행사할 수 있나요?`
2. `임대인이 계약갱신을 거절할 수 있는 사유는 무엇인가요?`
3. `임금 체불을 신고하려면 어떤 절차를 따라야 하나요?`
4. `특정 사건번호 판결의 주요 판단 근거는 무엇인가요?`
5. `공식 안내서에서 보증금 반환 절차를 어떻게 설명하나요?`

질문별 기대 문서·조문·사건번호를 정한 평가 세트가 있어야 검색 정확도를
수치로 확인할 수 있습니다.

## 7. 정확 조회와 벡터 검색의 MCP 역할 분리

현재 MCP 서비스의 `search_documents`는 질문을 벡터로 변환하고 Payload
필터를 함께 적용하는 의미 검색입니다. 정확한 조문·사건번호만으로 문서를
가져오는 전용 MCP Tool은 아직 구현되지 않았습니다.

### 벡터 검색 담당

- 자연어 질문과 유사 의미 검색
- 관련 법령·안내서·판례 후보 탐색
- `top_k` 결과와 유사도 점수 반환

### MCP 정확 조회 담당

- `article_label="제6조의3"`처럼 조문 번호가 명확한 요청
- `case_number="사건번호"`처럼 식별자가 명확한 요청
- `document_id`, 원문 URL 또는 특정 자료의 직접 조회
- 결과를 RAG·LangGraph가 사용할 공통 JSON으로 변환

현재 `search_documents`에서도 필터와 벡터 검색을 함께 사용할 수 있지만,
정확 조회 전용 요청까지 의미 검색을 거치는 것은 비효율적입니다. 이후 MCP에
`get_document_by_reference`와 같은 정확 조회 Tool을 추가하는 구조가
적절합니다.

## 8. 실제 검색 가능한 Qdrant 주소

원격 Qdrant가 준비됐으므로 각 PC에서 별도의 로컬 Qdrant Docker를 실행할
필요가 없습니다.

```text
https://31ff3cbb-4b6c-42a7-b0c1-09a7f3d59d65.us-west-1-0.aws.cloud.qdrant.io
```

각 팀원이 자신의 Database API Key를 `.env`에 입력해야 합니다. 주소만 알고
API 키가 없으면 접근할 수 없습니다. 현재 컬렉션은 비어 있으므로 접속은
가능하지만 실제 문서 검색 결과는 나오지 않습니다.

## 9. `team_documents_openai_dev` 테스트 문서 확인

`team_documents_openai_dev` 컬렉션은 존재하지만 Point 수는 0개입니다.
테스트 문서는 들어 있지 않습니다.

현재 사용 모델은 NVIDIA이므로 앞으로 문서는
`team_documents_nvidia_dev`에 적재합니다. OpenAI 컬렉션은 비어 있는
상태로 유지하며 자동 삭제하지 않습니다.

## 전처리 데이터 구조와 다음 작업

전처리 담당자가 전달한 청크 구조의 예상 건수는 다음과 같습니다.

| 경로 | 자료 | 예상 청크 수 |
| --- | --- | ---: |
| `data/chunk/laws` | 법령 | 2,231 |
| `data/chunk/guides` | 안내서 | 435 |
| `data/chunk/precedents` | 판례 | 349 |
| `data/chunk/consumer_sources` | 소비자 자료 | 169 |
| 합계 |  | 3,184 |

실제 파일을 받은 뒤 다음 순서로 진행합니다.

1. 입력 계약과 메타데이터 검증
2. NVIDIA 문서 임베딩 생성
3. `team_documents_nvidia_dev` 적재
4. 중복·누락·잔여 Point 및 `content_hash` 검증 보고서 생성
5. 평가 질문으로 Dense 검색 검증
6. 검증된 검색기를 MCP Tool에 연결
