"""안내서 본문·표·검토된 이미지 요약을 페이지 출처와 함께 청크로 저장한다.

uv run python src/Scripts/chunk_guides.py
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

if __package__:
    from .preprocess_guides import ROOT, OUTPUT_DIR as INPUT_DIR, sha256, write_json
else:
    from preprocess_guides import ROOT, OUTPUT_DIR as INPUT_DIR, sha256, write_json

OUTPUT_DIR = ROOT / "data/chunk/guides"


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
    return chunks


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, size=1200, overlap=120):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if (output_dir.is_relative_to(ROOT / "data/raw") or output_dir.is_relative_to(input_dir)
            or input_dir.is_relative_to(output_dir)):
        raise ValueError("출력은 원본·정제 입력과 분리해야 합니다.")
    manifest = json.loads((input_dir / "guide_preprocessing_report.json").read_text(encoding="utf-8"))
    if manifest["failures"]:
        raise ValueError("전처리 실패를 먼저 확인하세요.")
    # 쓰기 전에 모든 입력과 원본 지문을 검증한다.
    prepared = []
    for entry in manifest["saved"]:
        path = input_dir / entry["file"]
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["raw_sha256"] != sha256(ROOT / doc["source_path"]):
            raise ValueError("PDF 원본이 정제 당시와 다릅니다.")
        prepared.append((path, make_chunks(doc, size, overlap), doc))
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"schema_version": 1, "size": size, "overlap": overlap, "saved": [],
              "usage": "saved 파일만 로드. embedding_text 사용. 이미지 요약은 수동 설명이며 기준일 당시 자료임. 임베딩은 아직 수행하지 않음."}
    for path, chunks, doc in prepared:
        filename = doc["guide_id"] + "_chunks.json"
        write_json(output_dir / filename, {"guide_id": doc["guide_id"], "chunks": chunks})
        report["saved"].append({"file": filename, "source_file": doc["source_file"],
                                "processed_sha256": sha256(path), "raw_sha256": doc["raw_sha256"],
                                "chunk_count": len(chunks), "counts": dict(Counter(c["content_type"] for c in chunks)),
                                "represented_pages": sorted({c["page_number"] for c in chunks}),
                                "excluded_pages": [p["page_number"] for p in doc["pages"] if not p["embedding_candidate"]]})
    report["chunk_count"] = sum(e["chunk_count"] for e in report["saved"])
    report["counts"] = dict(sum((Counter(e["counts"]) for e in report["saved"]), Counter()))
    write_json(output_dir / "guide_chunking_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args()
    report = run(args.input_dir, args.output_dir, args.size, args.overlap)
    print(json.dumps({"chunk_count": report["chunk_count"], "counts": report["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
