"""안내 PDF를 원본 보존하며 페이지별 JSON으로 정제한다.

실행: uv run python src/scripts/preprocess_guides.py
필요 라이브러리: pdfplumber. OCR과 청크 생성은 수행하지 않는다.
"""

import argparse
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    PROCESSED_GUIDES_DATA_PATH,
    PROJECT_PATH,
    RAW_DATA_PATH,
    RAW_GUIDES_DATA_PATH,
)

ROOT = PROJECT_PATH
INPUT_DIR = RAW_GUIDES_DATA_PATH
OUTPUT_DIR = PROCESSED_GUIDES_DATA_PATH
TOC_LINE = re.compile(r"(?:\.\s*){5,}\d+\s*$")
HEADING = re.compile(r"^\d+(?:\.\d+)*\.\s+\S")
REVIEW_PATH = OUTPUT_DIR / "guide_visual_review.json"


def export_image(img, destination):
    """내장 원본을 복원한다. 페이지 밖으로 잘린 부분도 보존한다."""
    from PIL import Image

    data = img["stream"].get_data()
    fingerprint = hashlib.sha256(data).hexdigest()
    if data.startswith(b"\xff\xd8"):
        picture = Image.open(io.BytesIO(data))
    elif img.get("bits") == 8 and str(img["colorspace"][0]) == "/'Indexed'":
        picture = Image.frombytes("P", tuple(img["srcsize"]), data)
        picture.putpalette(img["colorspace"][3])
    elif img.get("bits") == 8 and str(img["colorspace"][0]) == "/'DeviceRGB'":
        picture = Image.frombytes("RGB", tuple(img["srcsize"]), data)
    else:
        raise ValueError("지원하지 않는 내장 이미지 형식: 시각 검토 필요")
    destination.parent.mkdir(parents=True, exist_ok=True)
    picture.convert("RGB").save(destination)
    return fingerprint


def outside_tables(obj, tables):
    if obj.get("object_type") != "char":
        return True
    x, y = (obj["x0"] + obj["x1"]) / 2, (obj["top"] + obj["bottom"]) / 2
    return not any(t["bbox"][0] <= x <= t["bbox"][2] and
                   t["bbox"][1] <= y <= t["bbox"][3] for t in tables)


def apply_visual_review(document, manifest):
    """PDF 및 이미지 해시가 모두 일치하는 수동 검토만 적용한다."""
    for page in document["pages"]:
        if page["page_kind"] != "content" or not page["images"]:
            continue
        reviews = [r for r in manifest.get("reviews", [])
                   if r["guide_id"] == document["guide_id"] and
                   r["page_number"] == page["page_number"]]
        valid = len(reviews) == len(page["images"])
        summaries = []
        for index, img in enumerate(page["images"]):
            matches = [r for r in reviews if r["image_index"] == index]
            if len(matches) != 1:
                valid = False
                continue
            review = matches[0]
            if (review.get("raw_sha256") != document["raw_sha256"] or
                    not img.get("stream_sha256") or
                    review.get("stream_sha256") != img["stream_sha256"] or
                    review.get("status") != "reviewed_full_embedded_image" or
                    not review.get("summary", "").strip()):
                valid = False
                continue
            summaries.append({**review, "text_origin": "manual_visual_summary",
                              "image_path": img.get("image_path")})
        page["visual_summaries"] = summaries if valid else []
        page["visual_review_required"] = not valid
        broken = "\ufffd" in page["text"] or bool(re.search(r"\(cid:\d+\)", page["text"]))
        page["needs_manual_review"] = not valid or page["ocr_required"] or broken
        page["embedding_candidate"] = bool(page["text"]) and not page["needs_manual_review"]
        if valid:
            page["warnings"] = [w for w in page["warnings"] if not w.startswith("이미지는")]
            page["warnings"].append("내장 이미지 전체 시각 검토 완료. 설명은 수동 요약이며 전문 OCR이 아님")
    return document


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_page(raw_text, page_number, page_count):
    """확인된 머리말/페이지 번호만 제거. 법률 숫자·기호·줄바꿈은 보존."""
    lines = raw_text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    removed = []
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if nonempty and re.fullmatch(r"찾기\s*쉬운\s*생활법령", lines[nonempty[0]].strip()):
        removed.append({"line": nonempty[0] + 1, "text": lines[nonempty[0]], "reason": "header"})
        lines[nonempty[0]] = ""
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if nonempty and re.fullmatch(rf"{page_number}\s*/\s*{page_count}", lines[nonempty[-1]].strip()):
        removed.append({"line": nonempty[-1] + 1, "text": lines[nonempty[-1]], "reason": "page_number"})
        lines[nonempty[-1]] = ""
    return "\n".join(line.rstrip() for line in lines).strip(), removed


def page_kind(text, page_number):
    if page_number == 1 and len(text.strip()) < 100:
        return "cover"
    if "이 정보는" in text and "기준으로 작성된 것입니다" in text and "법적 효력" in text:
        return "publication_notice"
    lines = [line for line in text.splitlines() if line.strip()]
    if lines and sum(bool(TOC_LINE.search(line)) for line in lines) / len(lines) >= 0.6:
        return "table_of_contents"
    return "content"


def headings(text):
    result, cursor = [], 0
    for line in text.splitlines(keepends=True):
        if HEADING.match(line.strip()):
            result.append({"text": line.strip(), "start": cursor, "end": cursor + len(line.rstrip("\n"))})
        cursor += len(line)
    return result


def preprocess(path, image_dir=None):
    import pdfplumber

    before = sha256(path)
    identifier = "guide_" + hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    pages = []
    with pdfplumber.open(path) as pdf:
        metadata = {str(k): str(v) for k, v in (pdf.metadata or {}).items()}
        count = len(pdf.pages)
        for number, page in enumerate(pdf.pages, 1):
            raw = page.extract_text() or ""
            text, removed = clean_page(raw, number, count)
            kind = page_kind(text, number)
            tables, warnings = [], []
            # PDF의 선으로 구획된 표만 추출. 자유 배치 텍스트를 표로 추정하지 않는다.
            for table in page.find_tables():
                cells = table.extract()
                tables.append({"bbox": list(table.bbox), "rows": cells})
            body_raw = (page.filter(lambda obj: outside_tables(obj, tables)).extract_text() or "") if tables else raw
            body_text = clean_page(body_raw, number, count)[0]
            images = []
            for index, img in enumerate(page.images):
                item = {"bbox": [img["x0"], img["top"], img["x1"], img["bottom"]],
                        "page_render_clipped": img["x0"] < 0 or img["x1"] > page.width or img["top"] < 0 or img["bottom"] > page.height}
                if kind == "content" and image_dir is not None:
                    destination = Path(image_dir) / f"{identifier}_p{number}_i{index}.png"
                    try:
                        item["stream_sha256"] = export_image(img, destination)
                        item["image_path"] = destination.relative_to(ROOT).as_posix() if destination.is_relative_to(ROOT) else str(destination)
                        item["image_sha256"] = sha256(destination)
                    except (ValueError, OSError, KeyError) as exc:
                        warnings.append(f"이미지 복원 실패: {exc}")
                images.append(item)
            ocr_needed = kind == "content" and len(re.sub(r"\s", "", text)) < 30
            if ocr_needed:
                warnings.append("본문 텍스트 부족: 원본 확인 및 OCR 검토 필요")
            if "\ufffd" in text or re.search(r"\(cid:\d+\)", text):
                warnings.append("깨진 문자 또는 CID 표기: 원본 대조 필요")
            if tables:
                warnings.append("표의 셀 구조를 별도 보존. 검색용 표 직렬화는 청크 생성 단계에서 처리")
            if images:
                warnings.append("이미지는 텍스트에 포함되지 않음. 이미지 위치를 보존했으며 시각 자료 확인 필요")
            visual_review = bool(images) and kind == "content"
            review = ocr_needed or visual_review or "\ufffd" in text or bool(re.search(r"\(cid:\d+\)", text))
            pages.append({"page_number": number, "page_kind": kind,
                          "raw_text": raw, "text": text, "body_text": body_text, "removed_lines": removed,
                          "headings": headings(text), "tables": tables, "images": images,
                          "ocr_required": ocr_needed, "needs_manual_review": review,
                          "visual_review_required": visual_review,
                          "embedding_candidate": kind == "content" and bool(text) and not review,
                          "warnings": warnings})
            page.close()
    if sha256(path) != before:
        raise ValueError("전처리 중 원본 PDF가 변경되었습니다.")
    combined = "\n".join(p["raw_text"] for p in pages[:5])
    match = re.search(r"이 정보는\s*(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일\s*기준", combined)
    basis = datetime(*map(int, match.groups())).date().isoformat() if match else None
    title = metadata.get("Title") or path.stem.removeprefix("[pdf]")
    identifier = "guide_" + hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    return {"schema_version": 2, "document_type": "official_guide", "guide_id": identifier,
            "title": title, "source_file": path.name,
            "source_path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
            "raw_sha256": before, "publisher": metadata.get("Author"),
            "source_site_url": "https://www.easylaw.go.kr" if "찾기쉬운 생활법령" in title else None,
            "source_url": None, "downloaded_at": None,
            "information_as_of": basis, "pdf_metadata": metadata,
            "page_count": len(pages), "pages": pages,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "warnings": ["문서별 원본 URL·다운로드 날짜 미확인. 생성일을 다운로드 날짜로 대체하지 않음",
                         "텍스트 줄바꿈 보존. PDF와 동일한 문단/시각 배치를 재현하지 않음"],
            "extraction": {"library": "pdfplumber", "version": pdfplumber.__version__,
                           "ocr_performed": False, "table_strategy": "lines"}}


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_source_coverage(report, input_dir=None):
    """추가·삭제·변경된 원본 또는 처리 상태 누락을 감지한다."""
    directory = Path(input_dir) if input_dir is not None else (ROOT / report['raw_input_dir'] if report.get('raw_input_dir') else RAW_GUIDES_DATA_PATH)
    current = {p.name:p for p in directory.glob('*.pdf')}
    records = report['saved'] + report['failures']
    names = [entry['source_file'] for entry in records]
    if len(names) != len(set(names)) or set(names) != set(current):
        missing = sorted(set(current)-set(names))
        stale = sorted(set(names)-set(current))
        raise ValueError(f'원본 PDF 처리 기록 불일치: 미기록={missing}, 원본 없음={stale}')
    inventory = report.get('input_files')
    if inventory is None:
        raise ValueError('원본 목록 기록이 없습니다. 전처리를 다시 실행하세요.')
    if len(inventory) != len(current) or {e['source_file'] for e in inventory} != set(current):
        raise ValueError('원본 PDF 목록 기록 불일치')
    for entry in inventory:
        if sha256(current[entry['source_file']]) != entry['raw_sha256']:
            raise ValueError('원본 PDF 지문 변경: '+entry['source_file'])
    return True


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(input_dir) or output_dir.is_relative_to(RAW_DATA_PATH):
        raise ValueError("원본 폴더 및 data/raw 아래에는 결과를 저장할 수 없습니다.")
    paths = sorted(input_dir.glob("*.pdf"))
    if not paths:
        raise ValueError("입력 폴더에 PDF가 없습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"processed_at": datetime.now(timezone.utc).isoformat(), "saved": [], "failures": [],
              "raw_input_dir": input_dir.relative_to(ROOT).as_posix() if input_dir.is_relative_to(ROOT) else str(input_dir),
              "input_files": [{"source_file":p.name,"raw_sha256":sha256(p)} for p in paths]}
    for path in paths:
        print(f"[추출 시작] {path.name}", flush=True)
        try:
            doc = preprocess(path, output_dir / "images")
            if REVIEW_PATH.exists():
                apply_visual_review(doc, json.loads(REVIEW_PATH.read_text(encoding="utf-8")))
            filename = f"{doc['guide_id']}.json"
            write_json(output_dir / filename, doc)
            entry = {"file": filename, "source_file": path.name, "raw_sha256": doc["raw_sha256"],
                     "page_count": doc["page_count"],
                     "retrieval_page_count": sum(p["embedding_candidate"] for p in doc["pages"]),
                     "ocr_page_count": sum(p["ocr_required"] for p in doc["pages"]),
                     "manual_review_page_count": sum(p["needs_manual_review"] for p in doc["pages"]),
                     "table_count": sum(len(p["tables"]) for p in doc["pages"]),
                     "image_page_count": sum(bool(p["images"]) for p in doc["pages"])}
            entry["visual_review_page_count"] = sum(p["visual_review_required"] for p in doc["pages"])
            report["saved"].append(entry)
            print(f"[완료] {path.name}: {entry['page_count']}페이지 / 표 {entry['table_count']}개", flush=True)
        except Exception as exc:
            report["failures"].append({"source_file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    report["usage"] = "saved 목록의 JSON을 읽고 pages.embedding_candidate를 확인하세요. 표의 rows와 페이지 번호를 보존한 청크 생성은 별도 단계입니다."
    validate_source_coverage(report, input_dir)
    write_json(output_dir / "guide_preprocessing_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report = run(args.input_dir, args.output_dir)
    except (ImportError, OSError, ValueError) as exc:
        print(f"전처리 실패: {exc}")
        return 1
    print(f"저장 {len(report['saved'])}건 / 실패 {len(report['failures'])}건")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
