"""안내서 본문·표·검토된 이미지 요약을 페이지 출처와 함께 청크로 저장한다.

uv run python src/scripts/chunk_guides.py
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    CHUNK_GUIDES_DATA_PATH,
    PROCESSED_GUIDES_DATA_PATH,
    PROJECT_PATH,
    RAW_DATA_PATH,
    RAW_GUIDES_DATA_PATH,
)

ROOT = PROJECT_PATH

if __package__:
    from .chunk_metadata import normalize_chunk, METADATA_VERSION
    from .preprocess_guides import sha256, write_json, validate_source_coverage
else:
    from chunk_metadata import normalize_chunk, METADATA_VERSION
    from preprocess_guides import sha256, write_json, validate_source_coverage

INPUT_DIR = PROCESSED_GUIDES_DATA_PATH
OUTPUT_DIR = CHUNK_GUIDES_DATA_PATH


def make_chunks(document, size=1200, overlap=120):
    if (type(size) is not int or type(overlap) is not int or
            size <= 0 or not 0 <= overlap < size):
        raise ValueError("size는 양의 정수, overlap은 size보다 작은 음이 아닌 정수")
    chunks = []
    for page in document["pages"]:
        if page["page_kind"] != "content" or not page["embedding_candidate"] or page["needs_manual_review"]:
            continue
        number = page["page_number"]
        common = {"schema_version": 1, "document_type": "official_guide",
                  "guide_id": document["guide_id"], "title": document["title"],
                  "page_number": number, "source_file": document["source_file"],
                  "source_path": document["source_path"], "raw_sha256": document["raw_sha256"],
                  "source_url": document["source_url"], "source_site_url": document["source_site_url"],
                  "publisher": document["publisher"], "information_as_of": document["information_as_of"],
                  "page_headings": [h["text"] for h in page["headings"]],
                  "citation": f"{document['title']} PDF {number}페이지",
                  "embedding_candidate": True, "warnings": page["warnings"]}
        header = f"[{document['title']} / PDF {number}페이지 / 자료 기준일 {document['information_as_of']}]"

        def append(kind, index, text, **extra):
            chunk_id = f"{document['guide_id']}_p{number:04d}_{kind}_{index:04d}"
            origin = "수동 이미지 요약·전문 전사 아님" if kind == "image_summary" else kind
            chunks.append({**common, "chunk_id": chunk_id, "content_type": kind,
                           "text": text, "context_header": header,
                           "embedding_text": header + f"\n[{origin}]\n" + text,
                           "text_sha256": hashlib.sha256(text.encode()).hexdigest(), **extra})

        # 페이지 경계를 넘지 않으며 모든 문자를 구간 정보로 추적한다.
        text = page["body_text"]
        start, index = 0, 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                boundary = text.rfind("\n", start + size // 2, end)
                if boundary >= 0:
                    end = boundary + 1
            append("body", index, text[start:end], start=start, end=end,
                   offset_basis="page.body_text", extraction_method="pdf_text_layer_outside_tables")
            index += 1
            if end == len(text):
                break
            start = max(start + 1, end - overlap)

        # 표는 셀 순서·빈 셀·병합 셀(None)을 그대로 보존하여 전체 표 단위로 저장한다.
        for index, table in enumerate(page["tables"]):
            serialized = "\n".join(json.dumps(row, ensure_ascii=False) for row in table["rows"])
            append("table", index, serialized, table_rows=table["rows"], bbox=table["bbox"],
                   extraction_method="pdfplumber_lines_table", atomic=True,
                   oversize_atomic_table=len(serialized) > size,
                   serialization_note="행 배열 순서 유지. null은 추출된 빈/병합 셀이며 값을 추정하지 않음")
        for summary in page.get("visual_summaries", []):
            append("image_summary", summary["image_index"], summary["summary"],
                   extraction_method="manual_visual_summary", visual_review=summary,
                   document_snapshot=True, not_full_transcription=True)
    return [normalize_chunk(c, document) for c in chunks]


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, size=1200, overlap=120, *,
        project_root=ROOT, raw_input_dir=None, report_path=None):
    """정제 문서와 원본 지문을 검증해 청크를 저장하고 보고서를 반환한다.

    project_root는 기록된 출처 상대 경로의 기준이다. 원본 폴더는 보고서의
    raw_input_dir 또는 명시한 raw_input_dir로 지정한다. API 호출은 없다.
    """
    if type(size) is not int or type(overlap) is not int or size <= 0 or not 0 <= overlap < size:
        raise ValueError("size는 양의 정수, overlap은 0 이상 size 미만 정수여야 합니다.")
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    project_root = Path(project_root).resolve()
    raw_root = (project_root / RAW_DATA_PATH.relative_to(ROOT)).resolve()
    report_path = Path(report_path).resolve() if report_path is not None else output_dir / "guide_chunking_report.json"
    if report_path.is_relative_to(input_dir) or report_path.is_relative_to(raw_root):
        raise ValueError("청크 보고서가 원본 또는 정제 입력을 덮어쓸 수 없습니다.")
    if (output_dir.is_relative_to(raw_root) or output_dir.is_relative_to(input_dir)
            or input_dir.is_relative_to(output_dir)):
        raise ValueError("출력은 원본·정제 입력과 분리해야 합니다.")
    manifest = json.loads((input_dir / "guide_preprocessing_report.json").read_text(encoding="utf-8"))
    recorded_dir = manifest.get("raw_input_dir") or RAW_GUIDES_DATA_PATH.relative_to(ROOT)
    source_dir = (Path(raw_input_dir).resolve() if raw_input_dir is not None else
                  (project_root / recorded_dir).resolve())
    if (output_dir.is_relative_to(source_dir) or source_dir.is_relative_to(output_dir)
            or report_path.is_relative_to(source_dir)):
        raise ValueError("출력은 지정한 원본 PDF 폴더와 분리해야 합니다.")
    validate_source_coverage(manifest, source_dir)
    if manifest["failures"]:
        raise ValueError("전처리 실패를 먼저 확인하세요.")
    # 쓰기 전에 모든 입력과 원본 지문을 검증한다.
    prepared = []
    for entry in manifest["saved"]:
        path = (input_dir / entry["file"]).resolve()
        if not path.is_relative_to(input_dir):
            raise ValueError("정제 보고서의 파일 경로가 입력 폴더를 벗어났습니다.")
        doc = json.loads(path.read_text(encoding="utf-8"))
        source_path = (project_root / doc["source_path"]).resolve()
        if not source_path.is_relative_to(source_dir):
            raise ValueError("문서의 원본 출처 경로가 지정한 PDF 폴더를 벗어났습니다.")
        if doc["raw_sha256"] != sha256(source_path):
            raise ValueError("PDF 원본이 정제 당시와 다릅니다.")
        if (entry['source_file'] != doc['source_file'] or entry['raw_sha256'] != doc['raw_sha256']):
            raise ValueError('정제 보고서와 문서 출처 불일치')
        prepared.append((path, make_chunks(doc, size, overlap), doc))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"metadata_version": METADATA_VERSION, "schema_version": 1, "size": size, "overlap": overlap, "saved": [],
              "usage": "saved 파일만 로드. embedding_text 사용. 이미지 요약은 수동 설명이며 기준일 당시 자료임. 임베딩은 아직 수행하지 않음."}
    for path, chunks, doc in prepared:
        filename = doc["guide_id"] + "_chunks.json"
        write_json(output_dir / filename, {"guide_id": doc["guide_id"], "chunks": chunks})
        report["saved"].append({"file": filename, "source_file": doc["source_file"],
                                "processed_sha256": sha256(path), "raw_sha256": doc["raw_sha256"],
                                "chunk_count": len(chunks), "counts": dict(Counter(c["content_type"] for c in chunks)),
                                "represented_pages": sorted({c["page_number"] for c in chunks}),
                                "excluded_pages": [p["page_number"] for p in doc["pages"] if not p["embedding_candidate"]],
                                "exclusion_details": [{"page_number":p['page_number'],"page_kind":p['page_kind'],
                                    "needs_manual_review":p['needs_manual_review'],"warnings":p['warnings']}
                                    for p in doc['pages'] if not p['embedding_candidate']]})
    report["chunk_count"] = sum(e["chunk_count"] for e in report["saved"])
    report["counts"] = dict(sum((Counter(e["counts"]) for e in report["saved"]), Counter()))
    write_json(report_path, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--raw-input-dir", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, default=None)
    args = parser.parse_args(argv)
    try:
        report = run(args.input_dir, args.output_dir, args.size, args.overlap,
                     project_root=args.project_root, raw_input_dir=args.raw_input_dir,
                     report_path=args.report_file)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"청크 생성 실패: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"chunk_count": report["chunk_count"], "counts": report["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
