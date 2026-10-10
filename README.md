# SKN35-3rd-3Team

## 임베딩 방식 빠른 선택

팀 기본값은 **OpenAI API**입니다. API 방식은 개인 키만 입력하면 고정된
모델·차원·컬렉션을 자동으로 선택합니다.

| 구분 | OpenAI API | NVIDIA API | 개인 그래픽 카드 |
| --- | --- | --- | --- |
| 선택 기준 | `OPENAI_API_KEY` | `NVIDIA_API_KEY` | `local_huggingface` 직접 설정 |
| 모델 | `text-embedding-3-small` | `nvidia/nemotron-3-embed-1b` | 팀에서 정한 Hugging Face 모델 |
| 벡터 차원 | 1536 | 2048 | 선택한 모델의 실제 차원 |
| 필요한 키 | `OPENAI_API_KEY` | `NVIDIA_API_KEY` | 공개 모델은 없음 |
| 추가 설치 | 없음 | 없음 | `sentence-transformers` |
| 컬렉션 | OpenAI 전용 | NVIDIA 전용 | 로컬 모델 전용 |

세 방식의 벡터는 같은 Qdrant 컬렉션에 섞지 않습니다. 방식을 변경하면 문서를
다시 임베딩하고 해당 방식의 전용 컬렉션에 적재합니다.

### 공통 준비

```powershell
uv sync
Copy-Item .env.example .env
uv run python -m src.Scripts.qdrant_setup check
```

실제 API 키가 들어 있는 `.env`는 Git에 올리지 않습니다. 로컬 Qdrant를
인증 없이 사용할 때는 `QDRANT_API_KEY`를 비워둘 수 있습니다.

`check` 결과 대상 컬렉션이 없고 생성 권한이 있는 경우에만 다음 명령을 한 번
실행합니다. 기존 컬렉션을 삭제하거나 강제로 재생성하지 않습니다.

```powershell
uv run python -m src.Scripts.qdrant_setup init
```

## 1. OpenAI API 사용

팀 기본 임베딩 모델은 OpenAI `text-embedding-3-small`이며 출력 벡터는
1536차원입니다. 개인 `.env`에는 API 키를 입력하고, 실제 키가 들어 있는
`.env`는 Git에 올리지 않습니다.

### `.env`에서 직접 변경할 값

```env
OPENAI_API_KEY=개인_OpenAI_API_키
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

NVIDIA는 현재 기본값이 아니지만 기존 코드에서 계속 지원합니다. OpenAI 키를
비우고 NVIDIA 개인 키만 입력하면 NVIDIA 고정 프로필을 사용합니다.

### `.env`에서 직접 변경할 값

```env
NVIDIA_API_KEY=개인_NVIDIA_API_키
```

두 API 키가 모두 설정된 경우에만
`EMBEDDING_PROVIDER=openai` 또는 `nvidia`를 추가해 하나를 선택합니다.

## 3. 개인 그래픽 카드 사용

Hugging Face에서 임베딩 모델을 내려받고, PyTorch·CUDA를 통해 개인 PC의
NVIDIA GPU에서 벡터를 생성하는 방식입니다. Qdrant 자체의 설정이 아니라
Qdrant에 저장할 벡터를 만드는 단계에 적용됩니다.

### Hugging Face 토큰이 필요한 경우

- 공개 모델: 일반적으로 토큰 없이 다운로드할 수 있습니다.
- 비공개 모델: 개인 `HF_TOKEN`이 필요합니다.
- 사용 승인이 필요한 모델: 모델 사용 동의 후 개인 `HF_TOKEN`이 필요합니다.
- 다운로드가 완료된 모델: 로컬 캐시를 사용하면 이후 실행에는 일반적으로
  토큰이 필요하지 않습니다.

### 개인 환경에만 추가 설치

```powershell
uv pip install sentence-transformers
```

CUDA를 지원하는 PyTorch와 NVIDIA 그래픽 드라이버도 준비되어 있어야 합니다.
이 선택 패키지는 팀 기본 의존성에 포함하지 않습니다.

### `.env`에서 직접 변경할 값

```env
EMBEDDING_PROVIDER=local_huggingface
EMBEDDING_MODEL=팀에서_정한_Hugging_Face_모델_ID
EMBEDDING_DIMENSION=모델의_실제_출력_차원
EMBEDDING_NORMALIZED=true
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=32
EMBEDDING_QUERY_PREFIX=
EMBEDDING_DOCUMENT_PREFIX=
QDRANT_COLLECTION=team_documents_local_model_dev

# 비공개·승인형 모델을 사용할 때만 입력
HF_TOKEN=개인_Hugging_Face_토큰
```

### 확인할 사항

- GPU 메모리가 부족하면 `EMBEDDING_BATCH_SIZE`를 줄입니다.
- CUDA를 사용할 수 없는 환경에서는 `EMBEDDING_DEVICE=cpu`를 사용합니다.
- 모델 ID·버전·차원·정규화 방식은 팀원 모두 동일하게 맞춥니다.
- 모델이 질문과 문서에 별도 접두사를 요구하면
  `EMBEDDING_QUERY_PREFIX`와 `EMBEDDING_DOCUMENT_PREFIX`를 설정합니다.
- OpenAI·NVIDIA·개인 GPU에서 만든 벡터는 각각 별도 컬렉션에 저장합니다.

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
