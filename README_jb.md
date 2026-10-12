# SKN35-3rd-3Team

내·외부 문서를 임베딩하고 Qdrant에서 검색한 근거를 MCP와 RAG에 전달하는
질의응답 프로젝트입니다.

## 현재 Vector DB 상태

2026-10-12 기준 확인 결과입니다.

| 항목 | 상태 |
| --- | --- |
| Qdrant | Cloud 연결 정상 |
| 활성 개발 컬렉션 | `team_documents_openai_dev` |
| 활성 임베딩 | OpenAI `text-embedding-3-small`, 1536차원 |
| 검색 방식 | Dense + BM25 Sparse + RRF Hybrid |
| Sparse 모델 | FastEmbed `Qdrant/bm25` |
| 거리 방식 | Cosine |
| 컬렉션 상태 | green |
| 입력 청크 | 4,932건 |
| Point 수 | 4,934 (토큰 제한 초과 청크 1건을 3개 Point로 분할) |
| Payload 인덱스 | Cloud 적용 완료 (`document_id`, `document_type`, `source_name`) |
| 적재 검증 | 누락·잔여·불일치·중복 0건 |

실제 Qdrant URL과 API 키는 `.env`에서 관리하며 Git에 올리지 않습니다.

## 전체 실행 순서

```text
전처리 청크 확인
    ↓
Qdrant 연결·컬렉션·Payload 인덱스 준비
    ↓
문서 Dense 임베딩·BM25 Sparse 임베딩
    ↓
Qdrant Batch Upsert
    ↓
적재 결과 검증
    ↓
질문 Dense·BM25 검색 후 RRF 결합
    ↓
MCP/RAG 전달
```

### 1. 환경 준비

```powershell
uv sync --locked
uv run python -m src.scripts.environment_setup
```

`uv sync --locked`는 `.python-version`의 Python 3.12와 `uv.lock`에 고정된
패키지만 설치합니다. `.env`와 API 키는 동기화하지 않습니다. 위 초기화 명령은
`.env`가 없을 때만 `.env.example`을 복사하며 기존 파일은 덮어쓰지 않습니다.
명령이 표시한 누락 키를 `.env`에 입력한 뒤 아래 상태 확인을 실행합니다.

```powershell
uv run python -m src.scripts.environment_setup --check
uv run python -m src.scripts.qdrant_setup check
```

Qdrant Cloud 주소는 코드에 팀 공용값으로 설정되어 있습니다. 팀원은 개인 키만
입력하면 됩니다.

```dotenv
QDRANT_API_KEY=개인별_Qdrant_Cloud_API_Key
```

로컬 Docker Qdrant를 사용할 때는 `.env`에서 주소를 명시적으로 덮어씁니다.
인증을 사용하지 않으면 `QDRANT_API_KEY`를 비워둘 수 있습니다.

```dotenv
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

### 2. 전처리 입력만 점검

```powershell
uv run python -m src.scripts.qdrant_ingest check-input
```

이 명령은 임베딩 API와 Qdrant를 호출하지 않습니다.

### 3. Qdrant 상태 확인

```powershell
uv run python -m src.scripts.qdrant_setup check
```

### 4. 컬렉션과 Payload 인덱스 준비

관리 권한이 있는 사용자가 적재 전에 실행합니다.

```powershell
uv run python -m src.scripts.qdrant_setup init
```

기존 컬렉션은 삭제하거나 강제로 재생성하지 않습니다. 누락된 Payload
인덱스만 생성하며 형식이 다르면 오류로 알려 줍니다.

### 5. 실제 임베딩·적재·검증

```powershell
uv run python -m src.scripts.qdrant_ingest ingest
```

적재 검증은 데이터를 자동 삭제하지 않습니다. 문제가 있으면 보고서에서
중복·누락·잔여 Point를 확인하고 입력을 수정한 뒤 재적재합니다.

## 전처리 입력 구조

```text
data/chunk/
├─ laws/              # JSONL
├─ guides/            # JSON
├─ precedents/        # JSONL
└─ consumer_sources/  # JSONL
```

- 전체 예상 레코드: **4,932건**
- 인코딩: UTF-8, BOM 없음 확인
- JSONL: 한 줄에 청크 한 건
- 안내서 JSON: `chunks` 배열 등 여러 청크 포함 가능
- `metadata` 중첩 객체 없이 필드를 최상위에 저장

현재 Vector DB 입력 필수값:

```json
{
  "document_id": "precedent:240967",
  "chunk_id": "240967_0000",
  "text": "정제와 청킹이 끝난 본문",
  "title": "사건명 또는 문서명",
  "source_url": "공식 원문 주소"
}
```

최상위 `title`은 Qdrant Payload의 `source_name`으로 변환됩니다. 입력 폴더에
따라 `document_type`도 자동으로 보완됩니다. 원본 파일은 수정하지 않습니다.

## Payload 인덱스

현재 실제 필터에 사용할 세 필드만 인덱싱합니다.

| 입력 기준 | Payload 키 | 형식 | 용도 |
| --- | --- | --- | --- |
| `document_id` | `document_id` | Keyword | 문서별 청크 조회·검증 |
| 폴더명 | `document_type` | Keyword | 자료 종류 필터 |
| `title` | `source_name` | Keyword | 문서명·사건명 정확 일치 |

조문·사건번호·시행일·선고일은 실제 전처리 JSON 키와 값 형식을 확인한 뒤
필터 요구가 있는 항목만 추가합니다.

## 임베딩 방식 선택

API 방식은 개인 키를 입력하면 팀 고정 모델·차원·컬렉션을 자동 선택합니다.

| 구분 | OpenAI API | NVIDIA API | 개인 그래픽 카드 |
| --- | --- | --- | --- |
| 선택 기준 | `OPENAI_API_KEY` | `NVIDIA_API_KEY` | `local_huggingface` 직접 설정 |
| 모델 | `text-embedding-3-small` | `nvidia/nemotron-3-embed-1b` | 팀에서 정한 모델 |
| 차원 | 1536 | 2048 | 모델의 실제 차원 |
| 컬렉션 | `team_documents_openai_dev` | `team_documents_nvidia_dev` | 별도 컬렉션 |
| 추가 설치 | 없음 | 없음 | `sentence-transformers` |

세 방식의 벡터는 같은 컬렉션에 섞지 않습니다. 모델을 바꾸면 문서를 다시
임베딩하고 해당 모델 전용 컬렉션에 적재해야 합니다.

현재 검증 상태:

| 프로필 | 상태 | 확인 내용 |
| --- | --- | --- |
| NVIDIA | 실제 검증 완료 | 2048차원, 정규화 벡터 길이 1.0 |
| OpenAI | 실제 검증 완료 | 1536차원 전체 4,932건 임베딩, 4,934 Point 적재·Hybrid 검색 확인 |
| 개인 GPU | 실제 호출 대기 | 고정 모델·차원 및 `torch`, `sentence-transformers` 필요 |

### OpenAI API

```dotenv
OPENAI_API_KEY=개인_OpenAI_API_키
```

### NVIDIA API

```dotenv
NVIDIA_API_KEY=개인_NVIDIA_API_키
```

두 키가 모두 설정된 경우에만 아래 값을 추가해 하나를 선택합니다.

```dotenv
EMBEDDING_PROVIDER=nvidia
```

### 개인 GPU

```powershell
uv pip install sentence-transformers
```

```dotenv
EMBEDDING_PROVIDER=local_huggingface
EMBEDDING_MODEL=팀에서_정한_Hugging_Face_모델_ID
EMBEDDING_DIMENSION=모델의_실제_차원
EMBEDDING_NORMALIZED=true
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=32
QDRANT_COLLECTION=team_documents_local_model_dev

# 비공개·승인형 모델일 때만 필요
HF_TOKEN=개인_Hugging_Face_토큰
```

공개 모델은 일반적으로 Hugging Face 토큰 없이 내려받을 수 있습니다. CUDA를
사용할 수 없거나 GPU 메모리가 부족하면 `EMBEDDING_DEVICE=cpu` 또는 더 작은
`EMBEDDING_BATCH_SIZE`를 사용합니다.

## 검색 범위

현재 구현:

- Dense Vector와 BM25 Sparse Vector의 RRF Hybrid 검색
- Hybrid 적재 완료 후 `SEARCH_MODE=dense`를 통한 Dense 단독 비교·복귀
- 선택적 점수 임계값
- 문자열·정수·불리언 정확 일치 필터
- `document_id` 기반 임베딩 없는 정확 조회
- 검색 결과의 본문·점수·Payload 반환
- 공식 MCP Python SDK v2 기반 `MCPServer`
- `search_documents`, `lookup_document`, `health_check`, `get_collection_info` Tool
- 로컬 `stdio` Transport와 선택적 Streamable HTTP 실행
- 인증 없는 Streamable HTTP의 비로컬 주소 바인딩 차단
- 실제 MCP Client의 Tool 조회 및 원격 Qdrant 상태 확인
- 실제 Cloud Point를 대상으로 MCP 네 Tool 연결 확인
- 토큰 제한으로 분할된 Point를 검색 결과에서 원본 논리 청크로 병합

남은 평가·선택 확장 항목:

- 법령·판례·안내서·소비자 자료의 대표 질문 검색 품질 평가
- Datetime 범위 검색
- 별도 `article_label`·`case_number` 필드 기반 정확 조회
- RAG·LangGraph 담당자에게 MCP Tool 입력·출력 계약 인계

MCP 로컬 서버는 다음 명령으로 실행합니다. MCP Host에서는 같은 명령을
`stdio` 서버로 등록하고 `search_documents`, `lookup_document`,
`health_check`, `get_collection_info` Tool을 호출합니다.

```powershell
uv run python -m src.scripts.mcp_server
```

질문 검색 명령은 선택한 provider의 Dense 질문 임베딩과 로컬 BM25 Sparse
임베딩을 만든 뒤 Qdrant RRF Hybrid 검색을 호출합니다.
`document_id` 조회는 임베딩 API를 호출하지 않고 Payload 인덱스를 사용합니다.
Point가 없는 상태에서는 `empty_result`가 정상입니다.

OpenAI 입력 제한을 초과한 청크는 적재 시 페이지 단위의 여러 물리 Point로
나눕니다. 검색된 Point에 `split_from_chunk_id`가 있으면 같은 부모의 조각을
순서대로 합쳐 원래 `chunk_id` 한 건으로 반환합니다. 반환 Metadata의
`merged_split_chunks`, `child_chunk_ids`, `pages`로 병합 여부와 범위를 확인할
수 있습니다.

팀 적재 명령은 Dense와 Sparse Vector 누락을 방지하기 위해
`SEARCH_MODE=hybrid`에서만 실행됩니다. `SEARCH_MODE=dense`는 Hybrid 적재가
완료된 컬렉션의 검색 결과 비교에만 사용합니다.

## 안전 원칙

- `.env`와 API 키는 Git에 올리지 않습니다.
- 로그에 키를 출력하지 않습니다.
- 기존 컬렉션을 자동 삭제·재생성하지 않습니다.
- 같은 `document_id/chunk_id`는 안정적인 Point ID로 Upsert합니다.
- 적재 검증 실패 시 데이터를 자동 삭제하지 않고 보고서로 확인합니다.
- 임베딩 모델·차원·청킹 구조가 바뀌면 새 컬렉션에 재색인합니다.

## HTTP 429 오류

HTTP 429는 일반적으로 Qdrant 문제가 아니라 임베딩 API의 호출 제한 또는
사용 한도 문제입니다.

- `rate_limit_exceeded`: 잠시 기다리거나 배치 크기와 호출 빈도를 줄입니다.
- `insufficient_quota`: 계정 크레딧·결제 설정·사용 한도를 확인합니다.
- 모델명이나 엔드포인트 오류는 일반적으로 400 또는 404와 구분합니다.

Vector DB의 공유 실행 기준과 구현 현황은 추적되는
`src/vector_db/README.md` 및 실제 코드를 참고합니다.
`qdrant_design_plan.md`는 Git 미추적 로컬 작업 메모이므로 팀 공통 기준으로
사용하지 않습니다.
