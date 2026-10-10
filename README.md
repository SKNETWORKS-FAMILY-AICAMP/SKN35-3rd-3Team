# SKN35-3rd-3Team

## 임베딩 방식 빠른 선택

팀 기본값은 **OpenAI API**입니다. 사용할 방식에 따라 프로젝트 루트의
`.env`만 변경합니다.

| 구분 | OpenAI API | NVIDIA API | 개인 그래픽 카드 |
| --- | --- | --- | --- |
| `EMBEDDING_PROVIDER` | `openai` | `nvidia` | `local_huggingface` |
| 모델 | `text-embedding-3-small` | `nvidia/nemotron-3-embed-1b` | 팀에서 정한 Hugging Face 모델 |
| 벡터 차원 | 1536 | 2048 | 선택한 모델의 실제 차원 |
| 필요한 키 | `OPENAI_API_KEY` | `NVIDIA_API_KEY` | 없음 |
| 추가 설치 | 없음 | 없음 | `sentence-transformers` |
| 컬렉션 | OpenAI 전용 | NVIDIA 전용 | 로컬 모델 전용 |

세 방식의 벡터는 같은 Qdrant 컬렉션에 섞지 않습니다. 방식을 변경하면 문서를
다시 임베딩하고 해당 방식의 전용 컬렉션에 적재합니다.

### 공통 준비

```powershell
uv sync
Copy-Item .env.example .env
```

실제 API 키가 들어 있는 `.env`는 Git에 올리지 않습니다. 로컬 Qdrant를
인증 없이 사용할 때는 `QDRANT_API_KEY`를 비워둘 수 있습니다.

## 1. OpenAI API 사용

팀 기본 임베딩 모델은 OpenAI `text-embedding-3-small`이며 출력 벡터는
1536차원입니다. 개인 `.env`에는 API 키를 입력하고, 실제 키가 들어 있는
`.env`는 Git에 올리지 않습니다.

### `.env`에서 직접 변경할 값

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

## 2. NVIDIA API 사용

NVIDIA는 현재 기본값이 아니지만 기존 코드에서 계속 지원합니다. 전환할 때는
개인 `.env`의 제공자·모델·차원·컬렉션을 함께 변경합니다.

### `.env`에서 직접 변경할 값

```env
EMBEDDING_PROVIDER=nvidia
NVIDIA_API_KEY=개인_NVIDIA_API_키
EMBEDDING_MODEL=nvidia/nemotron-3-embed-1b
EMBEDDING_DIMENSION=2048
EMBEDDING_NORMALIZED=true
QDRANT_COLLECTION=team_documents_nvidia_dev
```

## 3. 개인 그래픽 카드 사용

기본 OpenAI 설정과 별도로 `local_huggingface` 제공자를 지원합니다. 이 방식은
개인 PC의 GPU 또는 CPU에서 Sentence Transformers 모델을 실행하므로 API 호출
한도와 429 오류의 영향을 받지 않습니다.

로컬 GPU를 사용할 사람만 선택 패키지를 자신의 가상환경에 설치합니다. 기본
프로젝트 의존성에는 포함하지 않으므로 다른 팀원의 환경에는 영향을 주지
않습니다.

### 개인 환경에만 추가 설치

```powershell
uv pip install sentence-transformers
```

### `.env`에서 직접 변경할 값

```env
EMBEDDING_PROVIDER=local_huggingface
EMBEDDING_MODEL=팀에서_정한_Hugging_Face_모델_ID
EMBEDDING_DIMENSION=모델의_실제_출력_차원
EMBEDDING_NORMALIZED=true
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=32
QDRANT_COLLECTION=team_documents_local_model_dev
```

- CUDA가 없는 PC에서는 `EMBEDDING_DEVICE=cpu`를 사용합니다.
- 팀이 사용할 모델 ID·버전·차원·정규화 방식은 동일하게 고정합니다.
- 모델이 query/document 접두사를 요구하면 `EMBEDDING_QUERY_PREFIX`와
  `EMBEDDING_DOCUMENT_PREFIX`를 모델 설명에 맞게 설정합니다.
- OpenAI, NVIDIA 및 로컬 모델의 벡터는 각각 별도 Qdrant 컬렉션에 저장합니다.
- 개인 GPU는 벡터 생성에 사용하며 Qdrant 검색 서버를 GPU로 바꾸는 설정은
  아닙니다.

## HTTP 429 오류가 발생하는 경우

HTTP 429는 일반적으로 Qdrant 오류가 아니라 임베딩 API의 호출 제한 또는
사용 한도 문제입니다.

- `rate_limit_exceeded`: 잠시 기다린 후 재시도하거나 배치 크기와 호출 빈도를
  줄입니다.
- `insufficient_quota`: OpenAI 계정의 크레딧, 결제 설정과 프로젝트 사용 한도를
  확인합니다.
- 사용 중인 API 제공자는 오류 응답의 요청 주소와 메시지로 확인합니다.
- 모델명이나 엔드포인트가 잘못된 경우에는 일반적으로 400 또는 404 오류가
  발생하므로 429와 구분합니다.
