"""Qdrant Payload 필터 인덱스의 팀 공용 규격.

전처리 결과에서 실제로 확인된 키와 Vector DB 계층이 확정적으로 만드는
Payload만 등록합니다. 실제 키·값 형식이 확인되지 않은 후보 필드는 이 목록에
추가하지 않습니다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PayloadIndexSpec:
    """입력 필드와 Qdrant Payload 인덱스의 대응 관계."""

    field_name: str
    schema: str
    input_key: str
    purpose: str


CORE_PAYLOAD_INDEXES = (
    PayloadIndexSpec(
        field_name="document_id",
        schema="keyword",
        input_key="document_id",
        purpose="문서 한 건의 모든 청크를 정확히 조회·검증",
    ),
    PayloadIndexSpec(
        field_name="document_type",
        schema="keyword",
        input_key="폴더명(laws/guides/precedents/consumer_sources)",
        purpose="법령·안내서·판례·소비자 자료 종류별 검색",
    ),
    PayloadIndexSpec(
        field_name="source_name",
        schema="keyword",
        input_key="title",
        purpose="문서명·사건명과 정확히 일치하는 자료로 검색 범위 제한",
    ),
)


def payload_index_map() -> dict[str, str]:
    """필드명과 Qdrant 인덱스 형식의 읽기 전용 복사본을 반환합니다."""

    return {spec.field_name: spec.schema for spec in CORE_PAYLOAD_INDEXES}
