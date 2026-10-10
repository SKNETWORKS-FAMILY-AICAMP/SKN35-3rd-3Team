# SKN35-3rd-3Team

## 기본 임베딩 설정

기본 임베딩 모델은 OpenAI `text-embedding-3-small`이며 출력 벡터는
1536차원입니다. 개인 `.env`에는 API 키를 입력하고, 실제 키가 들어 있는
`.env`는 Git에 올리지 않습니다.

```env
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=개인_OpenAI_API_키
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSION=1536
EMBEDDING_NORMALIZED=true
QDRANT_COLLECTION=team_documents_openai_dev
```

- 다른 OpenAI 임베딩 모델을 선택하면 `EMBEDDING_MODEL`과
  `EMBEDDING_DIMENSION`도 해당 모델의 실제 규격으로 변경합니다.
- `EMBEDDING_BASE_URL`에 NVIDIA 주소가 설정되어 있다면 삭제하거나 비워야
  합니다. OpenAI 호환 서버를 직접 사용할 때만 해당 주소를 설정합니다.
- NVIDIA와 OpenAI가 만든 벡터는 서로 다른 벡터 공간이므로 같은 Qdrant
  컬렉션에 섞지 않습니다. 모델을 변경하면 문서를 다시 임베딩하고 별도
  컬렉션에 적재합니다.
- 질문 검색에도 문서를 적재할 때 사용한 것과 동일한 임베딩 모델과 차원을
  사용해야 합니다.

## NVIDIA API로 전환하는 경우

NVIDIA는 현재 기본값이 아니지만 기존 코드에서 계속 지원합니다. 전환할 때는
개인 `.env`의 제공자·모델·차원·컬렉션을 함께 변경합니다.

```env
EMBEDDING_PROVIDER=nvidia
NVIDIA_API_KEY=개인_NVIDIA_API_키
EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
EMBEDDING_DIMENSION=2048
EMBEDDING_NORMALIZED=true
QDRANT_COLLECTION=team_documents_nvidia_dev
```

### HTTP 429 오류가 발생하는 경우

HTTP 429는 일반적으로 Qdrant 오류가 아니라 임베딩 API의 호출 제한 또는
사용 한도 문제입니다.

- `rate_limit_exceeded`: 잠시 기다린 후 재시도하거나 배치 크기와 호출 빈도를
  줄입니다.
- `insufficient_quota`: OpenAI 계정의 크레딧, 결제 설정과 프로젝트 사용 한도를
  확인합니다.
- 사용 중인 API 제공자는 오류 응답의 요청 주소와 메시지로 확인합니다.
- 모델명이나 엔드포인트가 잘못된 경우에는 일반적으로 400 또는 404 오류가
  발생하므로 429와 구분합니다.
