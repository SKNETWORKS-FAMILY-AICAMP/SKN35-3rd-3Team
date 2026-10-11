"""MCP Tool 입력 및 문자열 JSON 반환 계약."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from src.vector_db.payload_indexes import CORE_PAYLOAD_INDEXES


MCP_EXTERNAL_STATUSES = frozenset(
    {
        "success",
        "empty_result",
        "connection_error",
        "invalid_request",
        "internal_error",
        "insufficient_evidence",
    }
)
MCP_FILTER_FIELDS = frozenset(
    index.field_name
    for index in CORE_PAYLOAD_INDEXES
    if index.schema == "keyword"
)


def _validated_filters(
    filters: Mapping[str, object] | None,
) -> dict[str, object]:
    """팀 공용 Keyword Payload 인덱스에 해당하는 필터만 허용합니다."""

    if filters is None:
        return {}
    if not isinstance(filters, Mapping):
        raise TypeError("filters는 객체 또는 null이어야 합니다.")

    unsupported_fields = sorted(
        str(key)
        for key in filters
        if not isinstance(key, str) or key not in MCP_FILTER_FIELDS
    )
    if unsupported_fields:
        allowed_fields = ", ".join(sorted(MCP_FILTER_FIELDS))
        unsupported_text = ", ".join(unsupported_fields)
        raise ValueError(
            "지원하지 않는 filters 필드입니다: "
            f"{unsupported_text}. 허용 필드: {allowed_fields}"
        )

    for key, value in filters.items():
        if isinstance(value, float) or not isinstance(value, (str, int, bool)):
            raise TypeError(
                "filters 값은 문자열·정수·불리언만 지원합니다: "
                f"{key}"
            )

    return dict(filters)


@dataclass(frozen=True)
class SearchDocumentsRequest:
    """`search_documents` Tool의 MCP 경계 입력."""

    query: str
    top_k: int = 5
    score_threshold: float | None = None
    filters: Mapping[str, object] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        query: str,
        *,
        top_k: int = 5,
        score_threshold: float | None = None,
        filters: Mapping[str, object] | None = None,
    ) -> SearchDocumentsRequest:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query는 비어 있지 않은 문자열이어야 합니다.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError("top_k는 정수여야 합니다.")
        if not 1 <= top_k <= 100:
            raise ValueError("top_k는 1 이상 100 이하여야 합니다.")

        if score_threshold is not None:
            if isinstance(score_threshold, bool) or not isinstance(
                score_threshold,
                (int, float),
            ):
                raise TypeError("score_threshold는 숫자 또는 null이어야 합니다.")
            score_threshold = float(score_threshold)
            if not math.isfinite(score_threshold):
                raise ValueError("score_threshold는 유한한 숫자여야 합니다.")

        return cls(
            query=query.strip(),
            top_k=top_k,
            score_threshold=score_threshold,
            filters=_validated_filters(filters),
        )


@dataclass(frozen=True)
class LookupDocumentRequest:
    """`lookup_document` Tool의 document_id 정확 조회 입력."""

    document_id: str
    limit: int = 20

    @classmethod
    def create(
        cls,
        document_id: str,
        *,
        limit: int = 20,
    ) -> LookupDocumentRequest:
        if not isinstance(document_id, str) or not document_id.strip():
            raise ValueError("document_id는 비어 있지 않은 문자열이어야 합니다.")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit은 정수여야 합니다.")
        if not 1 <= limit <= 100:
            raise ValueError("limit은 1 이상 100 이하여야 합니다.")
        return cls(document_id=document_id.strip(), limit=limit)


def serialize_tool_response(
    status: str,
    **fields: object,
) -> str:
    """MCP 외부 반환을 UTF-8 JSON 문자열로 직렬화합니다."""

    if status not in MCP_EXTERNAL_STATUSES:
        raise ValueError(f"지원하지 않는 MCP 상태값입니다: {status}")

    response = {"status": status, **fields}
    return json.dumps(response, ensure_ascii=False, separators=(",", ":"))


def parse_tool_response(response: str) -> dict[str, object]:
    """MCP 문자열 JSON을 검증된 사전으로 복원합니다."""

    if not isinstance(response, str) or not response.strip():
        raise ValueError("MCP 응답은 비어 있지 않은 문자열이어야 합니다.")

    try:
        parsed_response = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("MCP 응답이 올바른 JSON 문자열이 아닙니다.") from error

    if not isinstance(parsed_response, dict):
        raise ValueError("MCP 응답의 최상위 값은 객체여야 합니다.")

    status = parsed_response.get("status")
    if status not in MCP_EXTERNAL_STATUSES:
        raise ValueError("MCP 응답 status가 없거나 지원되지 않습니다.")

    return parsed_response
