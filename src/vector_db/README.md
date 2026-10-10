# 팀 Qdrant 사용 방법

팀 공용 값은 `src/const/vector_db_defaults.py`에 있습니다. 개인 PC의
`.env`에는 비밀 키만 입력합니다.

```dotenv
QDRANT_API_KEY=개인별_Qdrant_키_또는_manage_JWT
EMBEDDING_API_KEY=개인별_임베딩_API_키
```

기본 환경에서는 `OPENAI_API_KEY`를 사용합니다. NVIDIA로 전환할 때는
`NVIDIA_API_KEY`를 사용해도 됩니다.
키 문자열은 사용자마다 달라도 같은 제공자·모델이면 동일한 벡터 공간을
사용하므로 문제가 없습니다.

로컬 Docker Qdrant를 인증 없이 사용할 때는 `QDRANT_API_KEY`를 비워둘 수
있습니다. 실제 `.env`는 Git에 올리지 않습니다.

## 최초 준비

```powershell
uv sync
Copy-Item .env.example .env
uv run python -m src.Scripts.qdrant_setup check
```

컬렉션이 없고 생성 권한이 있는 사용자만 아래 명령을 한 번 실행합니다.

```powershell
uv run python -m src.Scripts.qdrant_setup init
```

`init`은 컬렉션이 없을 때만 생성합니다. 기존 컬렉션 삭제·강제 재생성은
하지 않으며, 이미 존재하면 1536차원·Cosine 규격이 맞는지만 확인합니다.

## 전처리 결과 입력 계약

전처리 담당자가 전달하는 `chunks.jsonl`은 한 줄에 청크 한 건을 저장합니다.
Vector DB 계층은 전처리 결과를 수정하지 않고 아래 필드를 검증한 뒤 임베딩과
Qdrant Payload를 생성합니다.

```json
{
  "document_id": "law-001",
  "chunk_id": "law-001-0001",
  "text": "전처리와 청킹이 끝난 본문",
  "metadata": {
    "source_name": "주택임대차보호법.pdf",
    "domain": "주택임대차",
    "document_type": "law",
    "effective_date": "2024-01-01",
    "source_url": "https://example.com/law/001",
    "article_label": "제6조의3",
    "page": 12,
    "section": "계약갱신 요구"
  }
}
```

검색과 출처 표시에 사용하는 권장 Payload는 다음과 같습니다.

- `domain`: 주택임대차·근로·중고거래 등 문서 업무 영역
- `document_type`: `law`·`precedent`·`official_guide` 등 문서 종류
- `effective_date`: 법령·지침의 시행일 또는 기준일 (`YYYY-MM-DD` 권장)
- `source_url`: 원문 출처 URL
- `article_label`: 조문·항목 번호 또는 문서 내부 표제

이 값들은 선택 필드이므로 기존 전처리 결과도 계속 사용할 수 있습니다.
`source_url`은 Payload에 그대로 보존하며, 기존 코드 호환성을 위해
`source_path` 별칭에도 같은 값을 넣습니다.

## 임베딩 제공자 변경

현재 기본값은 OpenAI `text-embedding-3-small`, 1536차원입니다.
`EMBEDDING_PROVIDER`는 `openai`, `nvidia`, `openai_compatible`,
`local_huggingface`를 지원합니다.

다른 키를 쓰는 것과 다른 모델을 쓰는 것은 구분해야 합니다.

- 키만 다름: 같은 컬렉션 사용 가능
- 모델 또는 제공자가 다름: 문서를 다시 임베딩하고 새 컬렉션 사용

NVIDIA `nvidia/nemotron-3-embed-1b`로 전환할 때는 2048차원과 NVIDIA 전용
컬렉션을 함께 설정합니다.

```dotenv
EMBEDDING_PROVIDER=nvidia
NVIDIA_API_KEY=개인키
EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
EMBEDDING_DIMENSION=2048
QDRANT_COLLECTION=team_documents_nvidia_dev
```

개인 GPU에서는 Sentence Transformers 모델을 직접 실행할 수 있습니다. 공개
모델은 일반적으로 Hugging Face 토큰 없이 다운로드할 수 있고, 비공개 또는
승인형 모델을 사용할 때만 `HF_TOKEN`이 필요합니다.

```powershell
uv pip install sentence-transformers
```

```dotenv
EMBEDDING_PROVIDER=local_huggingface
EMBEDDING_MODEL=팀에서_정한_Hugging_Face_모델_ID
EMBEDDING_DIMENSION=모델의_실제_차원
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=32
QDRANT_COLLECTION=team_documents_local_model_dev

# 비공개·승인형 모델일 때만 입력
HF_TOKEN=개인_Hugging_Face_토큰
```

예를 들어 다른 OpenAI 호환 API를 쓸 때는 개인 `.env`에 모델 규격과 전용
컬렉션을 함께 지정합니다.

```dotenv
EMBEDDING_PROVIDER=openai_compatible
EMBEDDING_API_KEY=개인키
EMBEDDING_BASE_URL=https://provider.example.com/v1
EMBEDDING_MODEL=provider/model-name
EMBEDDING_DIMENSION=모델의_실제_차원
QDRANT_COLLECTION=team_documents_provider_model_v1
```

## 코드에서 검색기 사용

```python
from src.vector_db import create_team_retriever

retriever = create_team_retriever()
results = retriever.search_documents(
    "질문 내용",
    top_k=5,
    score_threshold=None,
    filters={
        "domain": "주택임대차",
        "document_type": "law",
    },
)
```

현재 필터는 문자열·정수·불리언의 정확한 일치 조건을 지원합니다.
`effective_date`의 범위 검색은 별도 확장 대상입니다.

## Hybrid 검색 확장 경계

현재 구현은 OpenAI Embedding 기반 Dense Vector 검색입니다. BM25 키워드
검색과 Dense 검색을 RRF(Reciprocal Rank Fusion)로 결합하는 Hybrid 검색은
아직 구현하지 않았습니다.

Hybrid 검색을 추가할 때는 기존 Dense 컬렉션을 삭제하거나 강제로 재생성하지
않습니다. Sparse Vector 또는 키워드 인덱스 규격을 먼저 확정하고 별도
컬렉션에서 검증한 뒤 전환합니다. 전처리 결과의 `article_label`, `domain`,
`document_type`은 Hybrid 검색의 필터와 키워드 정확도를 높이는 데 재사용할 수
있습니다.

원격 Qdrant 주소가 확정되면 관리자가
`src/const/vector_db_defaults.py`의 `TEAM_QDRANT_URL`을 한 번 변경해
공유합니다. 이후 팀원은 URL·컬렉션명·모델·차원을 각자 입력하지 않고 개인
키만 사용합니다.

팀원의 manage JWT는 컬렉션 변경 권한이 있으므로 개인별로 발급하고, Qdrant
루트 관리자 키는 공유하지 않습니다. 일반 앱 실행에는 가능하면 read-only
키를 사용합니다.
