# MCP 연결

이 패키지는 기존 Retriever를 공식 MCP Python SDK v2의 조회 Tool로
노출합니다. 서버는 데이터 적재·삭제·컬렉션 재생성을 제공하지 않습니다.

## 제공 Tool

- `search_documents`: Dense + BM25 결과를 RRF로 결합한 Hybrid 검색
- `lookup_document`: 임베딩 없이 `document_id`가 같은 청크 정확 조회
- `health_check`: Qdrant 연결 확인
- `get_collection_info`: 민감정보를 제외한 컬렉션 상태 확인

`search_documents`의 `filters`는 모든 팀원에게 같은 규칙을 적용합니다. 현재
Qdrant Keyword Payload 인덱스로 확정된 다음 필드만 사용할 수 있습니다.

- `document_id`
- `document_type`
- `source_name`

미지원 필드나 문자열·정수·불리언 이외의 값은 Qdrant 또는 임베딩 API를
호출하기 전에 `invalid_request`로 반환합니다.

전처리 데이터에서 4,932개 청크 모두 `document_id`가 확인되었으므로 정확
조회는 이 값을 기준으로 제공합니다. `lookup_document`는 임베딩 API를 호출하지
않고 Qdrant의 `document_id` Keyword 인덱스를 직접 사용합니다.

## 실행

로컬 MCP Host가 하위 프로세스로 실행할 때는 기본 `stdio`를 사용합니다.

```powershell
uv run python -m src.scripts.mcp_server
```

Streamable HTTP가 필요한 팀 통합 환경에서는 다음처럼 실행합니다.

```powershell
uv run python -m src.scripts.mcp_server `
  --transport streamable-http `
  --host 127.0.0.1 `
  --port 8000
```

클라이언트 URL은 `http://127.0.0.1:8000/mcp`입니다. 원격에 공개할 때는
인증과 TLS를 별도 구성해야 하므로 현재 실행기는 `127.0.0.1`, `localhost`,
`::1` 이외의 주소 바인딩을 거부합니다. 팀 공유는 기본 `stdio`를 사용하고,
원격 HTTP가 필요하면 인증·TLS 게이트웨이를 먼저 구성합니다.

MCP Host에는 위 `stdio` 실행 명령을 서버로 등록합니다. 연결 후에는
`health_check`로 Qdrant 상태를 확인하고, `search_documents`로 Hybrid 검색,
`lookup_document`로 `document_id` 정확 조회를 수행합니다. 이 조회 Tool들은
Point를 생성·수정·삭제하지 않습니다.

서버 시작 시 `.env`의 Qdrant·임베딩 설정을 읽되 Qdrant 원격 검증은
지연합니다. 따라서 Qdrant가 일시적으로 응답하지 않아도 MCP 서버와 Tool은
시작되며, 실제 연결 상태는 `health_check`로 확인합니다. 실제 청크가 아직
적재되지 않았다면 `search_documents` 결과는 `empty_result`가 됩니다.

MCP Tool의 입력·출력 형식은 Dense에서 Hybrid로 전환해도 바뀌지 않습니다.
검색 구현은 `src.vector_db`와 `src.retriever`에 있으며, MCP는 같은
`search_documents` Tool로 결과를 전달합니다.
