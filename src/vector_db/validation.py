"""Qdrant 적재 결과를 삭제 없이 재확인합니다.

이 모듈은 입력 Point와 Qdrant에 저장된 Point의 식별자·Payload만 비교합니다.
문제를 발견해도 Qdrant 데이터를 삭제하거나 수정하지 않습니다.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from src.vector_db.ingestion import PreparedPoint


@dataclass(frozen=True)
class StoredPointSnapshot:
    """검증에 필요한 Qdrant Point의 최소 정보."""

    point_id: str
    payload: Mapping[str, object]


@dataclass(frozen=True)
class DuplicatePointGroup:
    """같은 문서·청크 키를 공유하는 저장 Point 묶음."""

    document_id: str
    chunk_id: str
    point_ids: tuple[str, ...]


@dataclass(frozen=True)
class IngestionValidationReport:
    """입력과 Qdrant 저장 결과의 읽기 전용 비교 결과."""

    expected_count: int
    stored_count: int
    matched_count: int
    missing_point_ids: tuple[str, ...]
    residual_point_ids: tuple[str, ...]
    content_mismatch_point_ids: tuple[str, ...]
    duplicate_groups: tuple[DuplicatePointGroup, ...]
    invalid_stored_point_ids: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """중복·누락·잔여·내용 불일치가 없으면 True입니다."""

        return not any(
            (
                self.missing_point_ids,
                self.residual_point_ids,
                self.content_mismatch_point_ids,
                self.duplicate_groups,
                self.invalid_stored_point_ids,
            )
        )

    def to_markdown(self) -> str:
        """사람이 재확인할 수 있는 Markdown 보고서를 반환합니다."""

        status = "통과" if self.passed else "재확인 필요"
        lines = [
            "# Qdrant 적재 검증 보고서",
            "",
            f"- 검증 결과: **{status}**",
            f"- 입력 Point: {self.expected_count}",
            f"- 조회된 Point: {self.stored_count}",
            f"- 정상 일치 Point: {self.matched_count}",
            f"- 누락 Point: {len(self.missing_point_ids)}",
            f"- 잔여 Point: {len(self.residual_point_ids)}",
            f"- 내용 불일치 Point: {len(self.content_mismatch_point_ids)}",
            f"- 중복 논리 청크: {len(self.duplicate_groups)}",
            f"- Payload 식별자 오류: {len(self.invalid_stored_point_ids)}",
            "",
            "> 이 보고서는 읽기 전용입니다. 자동 삭제를 수행하지 않습니다.",
        ]
        _append_id_section(lines, "누락 Point", self.missing_point_ids)
        _append_id_section(lines, "잔여 Point", self.residual_point_ids)
        _append_id_section(
            lines,
            "내용 불일치 Point",
            self.content_mismatch_point_ids,
        )
        _append_id_section(
            lines,
            "Payload 식별자 오류",
            self.invalid_stored_point_ids,
        )

        if self.duplicate_groups:
            lines.extend(("", "## 중복 논리 청크", ""))
            for group in self.duplicate_groups:
                point_ids = ", ".join(group.point_ids)
                lines.append(
                    f"- `{group.document_id}/{group.chunk_id}`: {point_ids}"
                )

        lines.extend(
            (
                "",
                "## 재확인 순서",
                "",
                "1. 입력의 `document_id`, `chunk_id`, `content_hash`를 확인합니다.",
                "2. 잘못된 입력을 수정한 뒤 같은 안정적 Point ID로 재적재합니다.",
                "3. 이 보고서를 다시 생성해 누락과 불일치가 해소됐는지 확인합니다.",
                "4. 잔여·중복 Point는 자동 삭제하지 않고 별도 정리 대상으로 남깁니다.",
                "",
            )
        )
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        """검증 결과를 UTF-8 Markdown 파일로 저장합니다."""

        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path


class StoredPointReader(Protocol):
    """문서 범위의 저장 Point를 읽는 최소 인터페이스."""

    def read_document_points(
        self,
        document_ids: Sequence[str],
        *,
        page_size: int = 256,
    ) -> Sequence[StoredPointSnapshot]: ...


def _payload_text(payload: Mapping[str, object], key: str) -> str | None:
    value = payload.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _append_id_section(
    lines: list[str],
    title: str,
    point_ids: Sequence[str],
) -> None:
    if not point_ids:
        return
    lines.extend(("", f"## {title}", ""))
    lines.extend(f"- `{point_id}`" for point_id in point_ids)


def build_ingestion_validation_report(
    expected_points: Sequence[PreparedPoint],
    stored_points: Sequence[StoredPointSnapshot],
) -> IngestionValidationReport:
    """입력 후보와 저장 결과에서 중복·누락·잔여를 계산합니다."""

    expected_by_id: dict[str, PreparedPoint] = {}
    for point in expected_points:
        if point.point_id in expected_by_id:
            raise ValueError(f"입력 Point ID가 중복되었습니다: {point.point_id}")
        expected_by_id[point.point_id] = point

    stored_by_id: dict[str, StoredPointSnapshot] = {}
    logical_groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    invalid_stored_ids: list[str] = []
    for point in stored_points:
        stored_by_id[point.point_id] = point
        document_id = _payload_text(point.payload, "document_id")
        chunk_id = _payload_text(point.payload, "chunk_id")
        if document_id is None or chunk_id is None:
            invalid_stored_ids.append(point.point_id)
            continue
        logical_groups[(document_id, chunk_id)].append(point.point_id)

    expected_ids = set(expected_by_id)
    stored_ids = set(stored_by_id)
    missing_ids = sorted(expected_ids - stored_ids)
    residual_ids = sorted(stored_ids - expected_ids)

    content_mismatch_ids: list[str] = []
    matched_count = 0
    for point_id in sorted(expected_ids & stored_ids):
        expected_hash = _payload_text(
            expected_by_id[point_id].payload,
            "content_hash",
        )
        stored_hash = _payload_text(
            stored_by_id[point_id].payload,
            "content_hash",
        )
        if expected_hash != stored_hash:
            content_mismatch_ids.append(point_id)
        else:
            matched_count += 1

    duplicate_groups = tuple(
        DuplicatePointGroup(
            document_id=document_id,
            chunk_id=chunk_id,
            point_ids=tuple(sorted(point_ids)),
        )
        for (document_id, chunk_id), point_ids in sorted(logical_groups.items())
        if len(point_ids) > 1
    )

    return IngestionValidationReport(
        expected_count=len(expected_points),
        stored_count=len(stored_points),
        matched_count=matched_count,
        missing_point_ids=tuple(missing_ids),
        residual_point_ids=tuple(residual_ids),
        content_mismatch_point_ids=tuple(content_mismatch_ids),
        duplicate_groups=duplicate_groups,
        invalid_stored_point_ids=tuple(sorted(invalid_stored_ids)),
    )


def validate_qdrant_ingestion(
    reader: StoredPointReader,
    expected_points: Sequence[PreparedPoint],
    *,
    page_size: int = 256,
) -> IngestionValidationReport:
    """입력 문서 범위만 Qdrant에서 읽어 적재 결과를 비교합니다."""

    document_ids = sorted(
        {
            document_id
            for point in expected_points
            if (
                document_id := _payload_text(
                    point.payload,
                    "document_id",
                )
            )
        }
    )
    if not document_ids:
        raise ValueError("검증할 입력에 document_id가 없습니다.")

    stored_points = reader.read_document_points(
        document_ids,
        page_size=page_size,
    )
    return build_ingestion_validation_report(expected_points, stored_points)
