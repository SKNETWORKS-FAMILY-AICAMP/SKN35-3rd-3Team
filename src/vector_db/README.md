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

## 임베딩 제공자 변경

현재 기본값은 OpenAI `text-embedding-3-small`, 1536차원입니다.
`EMBEDDING_PROVIDER`는 `nvidia`, `openai`, `openai_compatible`을 지원합니다.

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
    filters={},
)
```

원격 Qdrant 주소가 확정되면 관리자가
`src/const/vector_db_defaults.py`의 `TEAM_QDRANT_URL`을 한 번 변경해
공유합니다. 이후 팀원은 URL·컬렉션명·모델·차원을 각자 입력하지 않고 개인
키만 사용합니다.

팀원의 manage JWT는 컬렉션 변경 권한이 있으므로 개인별로 발급하고, Qdrant
루트 관리자 키는 공유하지 않습니다. 일반 앱 실행에는 가능하면 read-only
키를 사용합니다.
