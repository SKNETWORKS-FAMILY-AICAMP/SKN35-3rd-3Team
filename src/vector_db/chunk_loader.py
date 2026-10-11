"""전처리 팀의 ``data/chunk`` 폴더를 읽고 적재 전에 검증합니다.

원본 파일은 수정하지 않습니다. 실제 임베딩 API나 Qdrant에도 연결하지 않고,
폴더·파일·필수 필드·중복·예상 수량만 먼저 확인할 수 있습니다.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from src.const.vector_data_contract import PreparedChunk


@dataclass(frozen=True)
class ChunkDirectorySpec:
    """청크 하위 폴더의 입력 규격."""

    folder_name: str
    document_type: str


CHUNK_DIRECTORY_SPECS = (
    ChunkDirectorySpec("laws", "law"),
    ChunkDirectorySpec("guides", "official_guide"),
    ChunkDirectorySpec("precedents", "precedent"),
    ChunkDirectorySpec("consumer_sources", "consumer_source"),
)
EXPECTED_TOTAL_COUNT = 3_793

_SUPPORTED_SUFFIXES = frozenset({".json", ".jsonl"})
_CONTAINER_KEYS = ("chunks", "records", "items", "data")
_CORE_INPUT_KEYS = frozenset(
    {"document_id", "chunk_id", "text", "vector", "metadata"}
)


@dataclass(frozen=True)
class ChunkInputIssue:
    """파일 또는 개별 레코드에서 확인한 입력 문제."""

    path: str
    message: str
    record_number: int | None = None

    def to_text(self) -> str:
        location = self.path
        if self.record_number is not None:
            location = f"{location} (record {self.record_number})"
        return f"{location}: {self.message}"


@dataclass(frozen=True)
class ChunkInputReport:
    """Qdrant 연결 전에 생성하는 전처리 청크 점검 결과."""

    root: Path
    file_count: int
    discovered_count: int
    valid_count: int
    counts_by_folder: Mapping[str, int]
    expected_total_count: int
    duplicate_keys: tuple[str, ...]
    issues: tuple[ChunkInputIssue, ...]

    @property
    def passed(self) -> bool:
        return not self.issues and not self.duplicate_keys

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "passed": self.passed,
            "file_count": self.file_count,
            "discovered_count": self.discovered_count,
            "valid_count": self.valid_count,
            "counts_by_folder": dict(self.counts_by_folder),
            "expected_total_count": self.expected_total_count,
            "duplicate_keys": list(self.duplicate_keys),
            "issues": [issue.to_text() for issue in self.issues],
        }

    def to_markdown(self) -> str:
        status = "통과" if self.passed else "재확인 필요"
        lines = [
            "# Qdrant 입력 청크 사전 점검 보고서",
            "",
            f"- 점검 결과: **{status}**",
            f"- 입력 경로: `{self.root}`",
            f"- 발견 파일: {self.file_count}",
            f"- 발견 레코드: {self.discovered_count}",
            f"- 유효 레코드: {self.valid_count}",
            f"- 전체 예상 레코드: {self.expected_total_count}",
            "",
            "## 폴더별 수량",
            "",
            "| 폴더 | 실제 |",
            "| --- | ---: |",
        ]
        for spec in CHUNK_DIRECTORY_SPECS:
            actual = self.counts_by_folder.get(spec.folder_name, 0)
            lines.append(f"| `{spec.folder_name}` | {actual} |")

        if self.duplicate_keys:
            lines.extend(("", "## 중복 청크 키", ""))
            lines.extend(f"- `{key}`" for key in self.duplicate_keys)

        if self.issues:
            lines.extend(("", "## 입력 문제", ""))
            lines.extend(f"- {issue.to_text()}" for issue in self.issues)

        lines.extend(
            (
                "",
                "> 이 단계는 파일을 읽고 검증만 수행합니다. 임베딩 API와 "
                "Qdrant를 호출하지 않습니다.",
                "",
            )
        )
        return "\n".join(lines)

    def write_markdown(self, path: str | Path) -> Path:
        report_path = Path(path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(self.to_markdown(), encoding="utf-8")
        return report_path


@dataclass(frozen=True)
class LoadedChunkSet:
    """검증을 통과한 청크와 전체 점검 결과."""

    records: tuple[PreparedChunk, ...]
    report: ChunkInputReport


def _json_records(path: Path) -> Iterator[tuple[int, object]]:
    """JSON·JSONL 파일에서 레코드 번호와 값을 순서대로 반환합니다."""

    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8-sig") as input_file:
            record_number = 0
            for line_number, line in enumerate(input_file, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record_number += 1
                try:
                    yield record_number, json.loads(stripped)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"JSONL {line_number}행을 읽을 수 없습니다: {error.msg}"
                    ) from error
        return

    with path.open("r", encoding="utf-8-sig") as input_file:
        try:
            payload = json.load(input_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"JSON을 읽을 수 없습니다: {error.msg}") from error

    if isinstance(payload, list):
        for record_number, record in enumerate(payload, start=1):
            yield record_number, record
        return

    if isinstance(payload, Mapping):
        for key in _CONTAINER_KEYS:
            records = payload.get(key)
            if isinstance(records, list):
                for record_number, record in enumerate(records, start=1):
                    yield record_number, record
                return
        yield 1, payload
        return

    raise ValueError("JSON 최상위 값은 객체 또는 배열이어야 합니다.")


def _normalize_flattened_record(
    record: Mapping[str, object],
    document_type: str,
) -> Mapping[str, object]:
    """최상위 출처 필드를 표준 metadata 구조로 옮긴 복사본을 만듭니다."""

    metadata = record.get("metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        return record

    normalized_record = dict(record)
    normalized_metadata = dict(metadata or {})

    # 전처리 결과는 metadata 중첩 객체 없이 출처 필드를 최상위에 둡니다.
    # 핵심 입력값 외 필드는 Qdrant Payload에서 잃지 않도록 복사합니다.
    for key, value in record.items():
        if key not in _CORE_INPUT_KEYS:
            normalized_metadata.setdefault(str(key), value)

    # 화면과 답변 출처에 사용하는 표준 이름은 전처리의 title입니다.
    if not normalized_metadata.get("source_name"):
        title = normalized_metadata.get("title")
        if isinstance(title, str) and title.strip():
            normalized_metadata["source_name"] = title.strip()

    if not normalized_metadata.get("document_type"):
        normalized_metadata["document_type"] = document_type
    normalized_record["metadata"] = normalized_metadata
    return normalized_record


def _relative_text(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def load_chunk_tree(root: str | Path) -> LoadedChunkSet:
    """네 청크 폴더를 읽어 검증된 ``PreparedChunk``를 반환합니다."""

    chunk_root = Path(root).resolve()
    issues: list[ChunkInputIssue] = []
    records: list[PreparedChunk] = []
    counts: Counter[str] = Counter()
    discovered_count = 0
    file_count = 0

    if not chunk_root.is_dir():
        issues.append(
            ChunkInputIssue(str(chunk_root), "청크 루트 폴더가 없습니다.")
        )

    for spec in CHUNK_DIRECTORY_SPECS:
        folder = chunk_root / spec.folder_name
        if not folder.is_dir():
            issues.append(
                ChunkInputIssue(
                    _relative_text(folder, chunk_root),
                    "필수 하위 폴더가 없습니다.",
                )
            )
            continue

        input_files = sorted(
            path
            for path in folder.rglob("*")
            if path.is_file() and path.suffix.lower() in _SUPPORTED_SUFFIXES
        )
        file_count += len(input_files)
        if not input_files:
            issues.append(
                ChunkInputIssue(
                    _relative_text(folder, chunk_root),
                    "JSON 또는 JSONL 입력 파일이 없습니다.",
                )
            )
            continue

        for input_path in input_files:
            relative_path = _relative_text(input_path, chunk_root)
            try:
                file_records = list(_json_records(input_path))
            except (OSError, UnicodeError, ValueError) as error:
                issues.append(ChunkInputIssue(relative_path, str(error)))
                continue

            for record_number, raw_record in file_records:
                discovered_count += 1
                counts[spec.folder_name] += 1
                if not isinstance(raw_record, Mapping):
                    issues.append(
                        ChunkInputIssue(
                            relative_path,
                            "청크 한 건은 JSON 객체여야 합니다.",
                            record_number,
                        )
                    )
                    continue
                try:
                    record = PreparedChunk.from_mapping(
                        _normalize_flattened_record(
                            raw_record,
                            spec.document_type,
                        )
                    )
                    record.validate(require_vector=False)
                except (TypeError, ValueError) as error:
                    issues.append(
                        ChunkInputIssue(
                            relative_path,
                            str(error),
                            record_number,
                        )
                    )
                    continue
                records.append(record)

    if discovered_count != EXPECTED_TOTAL_COUNT:
        issues.append(
            ChunkInputIssue(
                str(chunk_root),
                "전체 예상 청크 수와 다릅니다: "
                f"expected={EXPECTED_TOTAL_COUNT}, actual={discovered_count}",
            )
        )

    key_counts = Counter(
        f"{record.document_id}/{record.chunk_id}" for record in records
    )
    duplicate_keys = tuple(
        sorted(key for key, count in key_counts.items() if count > 1)
    )

    report = ChunkInputReport(
        root=chunk_root,
        file_count=file_count,
        discovered_count=discovered_count,
        valid_count=len(records),
        counts_by_folder={
            spec.folder_name: counts[spec.folder_name]
            for spec in CHUNK_DIRECTORY_SPECS
        },
        expected_total_count=EXPECTED_TOTAL_COUNT,
        duplicate_keys=duplicate_keys,
        issues=tuple(issues),
    )
    return LoadedChunkSet(records=tuple(records), report=report)


def expected_chunk_count() -> int:
    """전처리 담당자가 확인한 전체 예상 청크 수를 반환합니다."""

    return EXPECTED_TOTAL_COUNT

