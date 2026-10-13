"""정제 법령의 본칙·부칙·별표 검색 후보를 청크로 분할한다.

uv run python src/scripts/chunk_laws.py
항·호·목 경계를 우선하며 긴 단일 본문만 문자 기준 겹침을 사용한다.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    CHUNK_LAWS_DATA_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_LAWS_DATA_PATH,
    PROJECT_PATH,
)

ROOT = PROJECT_PATH

if __package__:
    from .chunk_metadata import normalize_chunk, METADATA_VERSION
    from .preprocess_laws import REPORT_NAME, ensure_output, sha256, write_json
else:
    from chunk_metadata import normalize_chunk, METADATA_VERSION
    from preprocess_laws import REPORT_NAME, ensure_output, sha256, write_json

INPUT_DIR = PROCESSED_LAWS_DATA_PATH
OUTPUT_DIR = CHUNK_LAWS_DATA_PATH


def make_chunks(document, article, size=1200, overlap=120):
    if (isinstance(size, bool) or isinstance(overlap, bool) or not isinstance(size, int)
            or not isinstance(overlap, int) or size <= 0 or not 0 <= overlap < size):
        raise ValueError("size는 양의 정수, overlap은 size보다 작은 0 이상의 정수여야 합니다.")
    if not article["embedding_candidate"] or article["needs_manual_review"] or article["is_deleted"]:
        return []
    text, segments = article["body_text"], article["body_segments"]
    if not text or not segments:
        raise ValueError("검색 후보 조문의 본문이 비어 있습니다.")
    expected_start = 0
    for segment in segments:
        if segment["start"] != expected_start or text[segment["start"]:segment["end"]] != segment["text"]:
            raise ValueError("조문 본문의 구간 정보가 일치하지 않습니다.")
        expected_start = segment["end"] + 1
    if expected_start != len(text) + 1:
        raise ValueError("조문 본문에 누락된 구간이 있습니다.")
    boundaries = sorted({s["start"] for s in segments[1:]} | {len(text)})
    header = f"[{document['law_name']} / {article['article_label']}"
    if article["article_title"]:
        header += f"({article['article_title']})"
    header += f" / 법령 시행일 {document['effective_date']} / 조문 시행일 {article['effective_date']}]"
    if article["hierarchy"]:
        header += "\n[" + " > ".join(article["hierarchy"]) + "]"
    chunks, start = [], 0
    while start < len(text):
        limit = min(start + size, len(text))
        options = [b for b in boundaries if start < b <= limit]
        structural_end = bool(options)
        end = max(options) if options else limit
        fragment = text[start:end]
        intersecting = [s for s in segments if s["start"] < end and s["end"] > start]
        parents = [s for s in segments
                   if ((s["level"] == "조" and any(child["level"] != "조" for child in intersecting))
                       or any(child["path"] != s["path"]
                              and child["path"].startswith(s["path"].rsplit(".", 1)[0] + ".")
                              for child in intersecting))
                   and not (start <= s["start"] and s["end"] <= end)]
        parent_text = "\n".join(s["text"] for s in parents)
        embedding_text = header + ("\n[상위 조문 문맥]\n" + parent_text if parent_text else "") + "\n" + fragment
        chunks.append({"schema_version": 1, "document_type": "law",
                       "content_type": "main_article",
                       "chunk_id": f"{article['article_id']}_chunk_{len(chunks):04d}",
                       "version_id": document["version_id"], "article_id": article["article_id"],
                       "law_id": document["law_id"], "master_id": document["master_id"],
                       "law_name": document["law_name"], "article_number": article["article_number"],
                       "branch_number": article["branch_number"], "article_label": article["article_label"],
                       "article_title": article["article_title"], "hierarchy": article["hierarchy"],
                       "effective_date": document["effective_date"],
                       "article_effective_date": article["effective_date"],
                       "as_of": document["as_of"], "version_scope": document["version_scope"],
                       "source": document["source"], "source_url": document["source_url"],
                       "source_file": document["source_file"], "raw_sha256": document["raw_sha256"],
                       "source_unit_index": article["source_unit_index"],
                       "source_article_key": article["source_article_key"],
                       "text": fragment, "context_header": header,
                       "embedding_text": embedding_text,
                       "parent_context": [{"path": s["path"], "text": s["text"],
                                           "start": s["start"], "end": s["end"]} for s in parents],
                       "start": start, "end": end, "offset_basis": "article.body_text",
                       "segment_paths": [s["path"] for s in intersecting],
                       "segment_context": [{"path": s["path"], "level": s["level"], "label": s["label"]}
                                           for s in intersecting],
                       "split_basis": "structural_boundary" if structural_end else "long_segment_char_window",
                       "warnings": document["warnings"] + article["warnings"]})
        if end == len(text):
            break
        # 항·호 경계에서는 겹치지 않고 긴 단일 본문 내부에서만 겹친다.
        start = end if structural_end else max(start + 1, end - overlap)
    return [normalize_chunk(c, document) for c in chunks]


def make_aux_chunks(document, unit, size=1200, overlap=120):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("청크 크기·겹침이 유효하지 않습니다.")
    if not unit["embedding_candidate"] or unit["needs_manual_review"]:
        return []
    text = unit["body_text"]
    segments = unit["body_segments"]
    cursor = 0
    for segment in segments:
        if segment["start"] != cursor or text[segment["start"]:segment["end"]] != segment["text"]:
            raise ValueError("부칙·별표의 본문 구간이 일치하지 않습니다.")
        cursor = segment["end"]
    if not text or cursor != len(text):
        raise ValueError("부칙·별표 본문에 누락된 구간이 있습니다.")
    tables = unit.get("tables", [])
    # PDF 줄바꿈을 목 경계로 오인하면 구매일 불명 조건과 단서가 분리된다.
    # 일반적 해결기준의 비표형 본문은 기준·예외 전체를 원자 단위로 보존한다.
    atomic_criteria = (document["law_name"] == "소비자기본법 시행령"
                       and unit["content_type"] == "appendix" and not tables
                       and unit["title"].startswith("일반적 소비자분쟁해결기준"))
    ranges, cursor = [], 0
    for table in tables:
        if cursor < table["start"]:
            ranges.append((cursor, table["start"], None))
        ranges.append((table["start"], table["end"], table))
        cursor = table["end"]
    if cursor < len(text):
        ranges.append((cursor, len(text), None))
    if atomic_criteria:
        ranges = [(0, len(text), None)]
    if unit["content_type"] == "supplement":
        header = (f"[{document['law_name']} / 부칙 / 공포일 {unit['promulgation_date']} / "
                  f"공포번호 {unit['promulgation_number']} / {unit['title']}]\n"
                  "[개정별 부칙: 공포일과 시행일은 다름. 적용 여부는 시행일·경과조치 원문 확인]")
        context = "\n".join(unit["commencement_sections"])
    else:
        header = f"[{document['law_name']} / 별표 / {unit['title']} / 별표 시행일 {unit['effective_date']}]"
        # 표의 비고·일반기준도 문맥으로 유지한다.
        context = "\n".join(s["text"].strip() for s in segments if s["level"] != "표" and s["text"].strip())
        if atomic_criteria:
            context = ""  # 반환 본문에 이미 기준 전체가 있어 임베딩에 중복하지 않는다.
    boundaries = {s["end"] for s in segments}
    chunks = []
    for begin, stop, table in ranges:
        start = begin
        while start < stop:
            limit = min(start + size, stop)
            options = [b for b in boundaries if start < b <= limit]
            end = stop if table or atomic_criteria else (max(options) if options else limit)
            fragment = text[start:end]
            # 공백 전용 구간은 검색용 청크에서 제외하되 원본에는 남긴다.
            if fragment.strip():
                search_text = table["serialized_text"] if table else fragment
                chunks.append({"schema_version": 1, "document_type": "law",
                               "content_type": unit["content_type"],
                               "chunk_id": f"{unit['unit_id']}_chunk_{len(chunks):04d}",
                               "unit_id": unit["unit_id"], "article_id": None,
                               "version_id": document["version_id"], "law_id": document["law_id"],
                               "master_id": document["master_id"], "law_name": document["law_name"],
                               "unit_title": unit["title"], "effective_date": document["effective_date"],
                               "unit_effective_date": unit["effective_date"],
                               "promulgation_date": unit.get("promulgation_date"),
                               "promulgation_number": unit.get("promulgation_number"),
                               "temporal_scope": unit.get("temporal_scope", "appendix_at_collected_version"),
                               "as_of": document["as_of"], "version_scope": document["version_scope"],
                               "source": document["source"],
                               "source_url": unit.get("source_url", document["source_url"]),
                               "law_source_url": document["source_url"],
                               "source_file": document["source_file"], "raw_sha256": document["raw_sha256"],
                               "pdf_file": unit.get("pdf_file"), "pdf_sha256": unit.get("pdf_sha256"),
                               "source_unit_index": unit["source_unit_index"],
                               "text": fragment, "context_header": header,
                               "embedding_text": header + ("\n[시행일·일반기준 문맥]\n" + context if context else "") + "\n" + search_text,
                               "context_text": context, "start": start, "end": end,
                               "offset_basis": "unit.body_text", "atomic_table": bool(table),
                               "oversize_atomic_table": bool(table) and len(fragment) > size,
                               "table_rows": table["physical_rows"] if table else [],
                               "split_basis": "whole_general_criteria" if atomic_criteria else ("whole_table" if table else "paragraph_or_char_window"),
                               **({"atomic_unit": True, "oversize_atomic_unit": len(fragment) > size}
                                  if atomic_criteria else {}),
                               "warnings": document["warnings"] + unit["warnings"]})
            if end == stop:
                break
            start = end if options else max(start + 1, end - overlap)
    return [normalize_chunk(c, document) for c in chunks]


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, size=1200, overlap=120):
    """정제 법령을 청크로 저장하고 결과 보고서를 반환한다. 출력·API 호출은 없다."""
    if type(size) is not int or type(overlap) is not int or size <= 0 or not 0 <= overlap < size:
        raise ValueError("size는 양의 정수, overlap은 0 이상 size 미만 정수여야 합니다.")
    input_dir, output_dir = ensure_output(input_dir, output_dir)
    if output_dir.is_relative_to(PROCESSED_DATA_PATH):
        raise ValueError("processed를 덮어쓰지 않도록 chunk 폴더를 지정해주세요.")
    manifest = json.loads((input_dir / REPORT_NAME).read_text(encoding="utf-8"))
    if manifest["failures"]:
        raise ValueError("법령 전처리에 실패한 파일이 있습니다. 먼저 확인해주세요.")
    if not manifest["saved"]:
        raise ValueError("정제된 법령 파일이 없습니다.")
    chunks, reports, versions = [], [], set()
    for entry in manifest["saved"]:
        filename = entry["file"]
        if Path(filename).name != filename:
            raise ValueError("검토 목록의 파일 경로가 유효하지 않습니다.")
        doc = json.loads((input_dir / filename).read_text(encoding="utf-8"))
        if doc["version_id"] in versions:
            raise ValueError("정제 목록에 중복 법령 버전이 있습니다.")
        versions.add(doc["version_id"])
        if doc["raw_sha256"] != entry["raw_sha256"]:
            raise ValueError("정제 결과와 목록의 원본 해시가 다릅니다.")
        main_chunks = [c for a in doc["articles"] for c in make_chunks(doc, a, size, overlap)]
        if "supplements" not in doc or "appendices" not in doc:
            raise ValueError("부칙·별표 정제를 위해 preprocess_laws.py를 먼저 다시 실행해주세요.")
        extra_chunks = [c for u in doc["supplements"] + doc["appendices"] for c in make_aux_chunks(doc, u, size, overlap)]
        law_chunks = main_chunks + extra_chunks
        chunks.extend(law_chunks)
        reports.append({"law_name": doc["law_name"], "version_id": doc["version_id"],
                        "file": filename, "processed_sha256": sha256(input_dir / filename),
                        "article_count": doc["article_count"],
                        "chunked_article_count": len({c["article_id"] for c in main_chunks}),
                        "main_chunk_count": len(main_chunks),
                        "supplement_count": len(doc["supplements"]), "appendix_count": len(doc["appendices"]),
                        "supplement_chunk_count": sum(c["content_type"] == "supplement" for c in extra_chunks),
                        "appendix_chunk_count": sum(c["content_type"] == "appendix" for c in extra_chunks),
                        "chunk_count": len(law_chunks),
                        "deleted_count": sum(a["is_deleted"] for a in doc["articles"]),
                        "manual_review_count": sum(a["needs_manual_review"] for a in doc["articles"])})
    if len(chunks) != len({c["chunk_id"] for c in chunks}):
        raise ValueError("중복 청크 ID가 있습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / "chunks.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in chunks), encoding="utf-8")
    temporary.replace(target)
    report = {"metadata_version": METADATA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "law_count": len(reports),
              "chunk_count": len(chunks), "chunked_article_count": sum(e["chunked_article_count"] for e in reports),
              "size": size, "overlap": overlap,
              "split_policy": "문자 수 기준 최대 크기. 항·호·목 경계 우선, 긴 단일 본문 내부에서만 겹침",
              "chunks_sha256": sha256(target), "laws": reports,
              "content_type_counts": {kind: sum(c["content_type"] == kind for c in chunks)
                                      for kind in ("main_article", "supplement", "appendix")},
              "oversize_atomic_table_count": sum(c.get("oversize_atomic_table", False) for c in chunks),
              "scope": "본칙·개정별 부칙·별표 포함. 표는 크기를 초과해도 통째로 보존. 개정문·제개정이유는 검색 제외",
              "warnings": ["조문 청크 준비 완료이며 임베딩·Qdrant 적재·검색 평가는 아직 수행하지 않음"]}
    write_json(output_dir / "law_chunking_report.json", report)
    return report


def main(argv=None):
    """명령행 인자, 결과 출력 및 종료 코드를 처리한다."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=120)
    args = parser.parse_args(argv)
    if args.size <= 0 or not 0 <= args.overlap < args.size:
        parser.error("size는 양수이고 overlap은 size보다 작은 0 이상의 값이어야 합니다.")
    try:
        report = run(args.input_dir, args.output_dir, args.size, args.overlap)
    except (KeyError, TypeError, ValueError, OSError) as exc:
        print(f"청크 생성 실패: {exc}")
        return 1
    print(f"청크 생성: 법령 {report['law_count']}종 / 조문 {report['chunked_article_count']}개 / 청크 {report['chunk_count']}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
