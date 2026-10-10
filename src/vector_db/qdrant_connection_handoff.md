# Qdrant 연결 및 데이터 규격 전달서

## 1. Qdrant 연결

프로젝트에서는 `.env`의 접속 정보를 읽어 Qdrant Client를 생성합니다.

```python
from src.const.env_loader import load_project_env
from src.const.qdrant_config import load_qdrant_settings
from src.vector_db.qdrant_store import QdrantVectorStore

load_project_env()
settings = load_qdrant_settings()

vector_store = QdrantVectorStore.from_settings(settings)
```

내부적으로는 다음 설정으로 `QdrantClient`를 생성합니다.

```python
QdrantClient(
    url=settings.connection.url,
    api_key=settings.connection.api_key,
    timeout=settings.connection.timeout_seconds,
    prefer_grpc=settings.connection.prefer_grpc,
)
```

개인 `.env`에는 실제 환경에 맞는 값을 입력합니다. 실제 키는 Git에 올리지
않습니다.

```dotenv
QDRANT_URL=원격_Qdrant_주소
QDRANT_API_KEY=개인별_접속_키
```

현재 로컬 설정은 다음과 같습니다.

| 항목 | 현재 값 |
| --- | --- |
| Qdrant URL | `http://localhost:6333` |
| API Key | 사용하지 않음 |
| Timeout | 10초 |
| gRPC | 사용하지 않음 |

`localhost`는 해당 PC 안에서만 접근할 수 있습니다. 다른 팀원이 동일한 서버를
사용하려면 접근 가능한 원격 Qdrant 주소와 개인 접속 키가 필요합니다.

현재 로컬 Qdrant 연결 확인은 실패한 상태이므로, 아래 컬렉션이 실제로 생성되고
데이터가 적재됐는지는 아직 확인되지 않았습니다.

## 2. 검색 컬렉션

### OpenAI 기본 프로필

| 항목 | 값 |
| --- | --- |
| 컬렉션 | `team_documents_openai_dev` |
| 벡터 차원 | 1536 |
| 거리 계산 | Cosine |
| Named Vector | 사용하지 않음 |

### NVIDIA 대체 프로필

| 항목 | 값 |
| --- | --- |
| 컬렉션 | `team_documents_nvidia_dev` |
| 벡터 차원 | 2048 |
| 거리 계산 | Cosine |
| Named Vector | 사용하지 않음 |

서로 다른 임베딩 모델이 만든 벡터는 같은 컬렉션에 혼합하지 않습니다.

## 3. Qdrant Point 및 Payload 구조

Qdrant에는 ID, 벡터, Payload를 한 Point로 저장합니다.

```json
{
  "id": "UUID 형식의 Point ID",
  "vector": [0.0123, -0.0456],
  "payload": {
    "text": "검색할 문서 청크 본문",
    "document_id": "law-001",
    "chunk_id": "law-001-0001",
    "content_hash": "본문 해시값",
    "indexed_at": "2026-10-11T00:00:00+00:00",
    "source_name": "주택임대차보호법.pdf",
    "source_path": "원본 경로 또는 URL",
    "source_url": "https://example.com/document",
    "page": 12,
    "section": "계약갱신 요구",
    "domain": "주택임대차",
    "document_type": "law",
    "effective_date": "2024-01-01",
    "article_label": "제6조의3",
    "document_updated_at": "2026-10-01",
    "published_at": "2024-01-01",
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "embedding_dimension": 1536,
    "embedding_normalized": true
  }
}
```

### 필수 Payload

- `text`: 검색 결과로 반환할 청크 본문
- `document_id`: 원본 문서 식별자
- `chunk_id`: 문서 안의 청크 식별자
- `content_hash`: 청크 본문의 해시값
- `indexed_at`: Qdrant 적재 시각

### 선택 Payload

- `source_name`, `source_path`, `source_url`
- `page`, `section`
- `domain`, `document_type`
- `effective_date`, `article_label`
- `document_updated_at`, `published_at`
- `embedding_provider`, `embedding_model`
- `embedding_dimension`, `embedding_normalized`

전처리 결과에 다른 메타데이터가 있으면 추가 Payload로 함께 보존합니다.
`source_url`은 기존 코드 호환을 위해 `source_path`에도 같은 값을 저장할 수
있습니다.

## 4. 저장 시 사용하는 임베딩 모델

### OpenAI 기본 프로필

| 항목 | 값 |
| --- | --- |
| 제공자 | OpenAI |
| 모델 | `text-embedding-3-small` |
| 차원 | 1536 |
| 정규화 | `true` |
| 컬렉션 | `team_documents_openai_dev` |

```dotenv
OPENAI_API_KEY=개인_OpenAI_API_키
```

### NVIDIA 대체 프로필

| 항목 | 값 |
| --- | --- |
| 제공자 | NVIDIA |
| 모델 | `nvidia/nemotron-3-embed-1b` |
| 차원 | 2048 |
| 정규화 | `true` |
| 컬렉션 | `team_documents_nvidia_dev` |

```dotenv
NVIDIA_API_KEY=개인_NVIDIA_API_키
```

OpenAI 키와 NVIDIA 키가 모두 설정된 경우에는 사용할 제공자를 명시합니다.

```dotenv
EMBEDDING_PROVIDER=openai
```

문서 적재와 질문 검색에는 반드시 같은 임베딩 모델과 같은 전용 컬렉션을
사용해야 합니다.

## 5. 관련 코드 위치

- Qdrant Client 생성: `src/vector_db/qdrant_store.py`
- 환경 설정 변환: `src/const/qdrant_config.py`
- 모델·차원·컬렉션 고정값: `src/const/vector_db_defaults.py`
- Payload 생성: `src/const/vector_data_contract.py`
- 문서 임베딩 및 Point 준비: `src/vector_db/ingestion.py`
- 검색기 조립: `src/vector_db/bootstrap.py`
