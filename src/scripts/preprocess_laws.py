"""법령 원본 JSON을 조문별로 정제한다. 원본 파일을 수정하지 않는다.

uv run python src/scripts/preprocess_laws.py
본칙·부칙·별표를 정제한다. 별표 PDF 최초 수집: --download-appendices
"""

import argparse
import hashlib
import html
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urljoin, urlsplit

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    PROCESSED_LAWS_DATA_PATH,
    PROJECT_PATH,
    RAW_DATA_PATH,
    RAW_LAWS_DATA_PATH,
    RAW_LAW_APPENDICES_DATA_PATH,
)

ROOT = PROJECT_PATH
INPUT_DIR = RAW_LAWS_DATA_PATH
OUTPUT_DIR = PROCESSED_LAWS_DATA_PATH
REPORT_NAME = "law_preprocessing_report.json"
LEVELS = ("항", "호", "목")
APPENDIX_DIR = RAW_LAW_APPENDICES_DATA_PATH


def appendix_url(unit):
    link = unit.get("별표서식PDF파일링크", "")
    if not isinstance(link, str) or not link:
        raise ValueError("별표 PDF 링크가 없습니다.")
    url = urljoin("https://www.law.go.kr", link)
    parts = urlsplit(url)
    if parts.scheme != "https" or parts.netloc != "www.law.go.kr" or parts.path != "/LSW/flDownload.do":
        raise ValueError("국가법령정보센터 별표 다운로드 링크가 아닙니다.")
    return url


def appendix_path(version_id, unit):
    key = unit.get("@별표키", "")
    if not re.fullmatch(r"[0-9A-Za-z]+", key) or not re.fullmatch(r"\d+_\d+_\d{8}", version_id):
        raise ValueError("별표 또는 법령 버전 식별번호가 유효하지 않습니다.")
    return APPENDIX_DIR / f"{version_id}_appendix_{key}.pdf"


def download_appendices(input_dir=INPUT_DIR):
    """승인된 수집 단계. 기존 PDF 및 출처 기록은 덮어쓰지 않는다."""
    import requests

    APPENDIX_DIR.mkdir(parents=True, exist_ok=True)
    for source in sorted(Path(input_dir).glob("[0-9]*.json")):
        raw = json.loads(source.read_text(encoding="utf-8"))
        listing = raw["list_metadata"]
        version = f"{listing['법령ID']}_{listing['법령일련번호']}_{listing['시행일자']}"
        for unit in as_list(raw["detail"].get("별표", {}).get("별표단위")):
            path, url = appendix_path(version, unit), appendix_url(unit)
            record_path = path.with_suffix(".json")
            if path.exists():
                record = json.loads(record_path.read_text(encoding="utf-8"))
                if record["source_url"] != url or record["sha256"] != sha256(path):
                    raise ValueError("기존 별표 PDF의 출처 또는 해시가 다릅니다. 자동 덮어쓰기 중단")
                continue
            if record_path.exists():
                raise ValueError("PDF 없이 출처 기록만 남아 있습니다. 수동 확인 필요")
            try:
                response = requests.get(url, timeout=(10, 45))
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RuntimeError(f"별표 PDF 다운로드 실패: {type(exc).__name__}") from None
            if not response.content.startswith(b"%PDF-"):
                raise ValueError("별표 첨부파일 응답이 PDF가 아닙니다.")
            # PDF 구문을 확인한 후 원본 저장.
            import io
            import pdfplumber
            with pdfplumber.open(io.BytesIO(response.content)) as pdf:
                if not pdf.pages:
                    raise ValueError("별표 PDF 페이지가 없습니다.")
            with path.open("xb") as stream:
                stream.write(response.content)
            write_json(record_path, {"source_url": url, "version_id": version,
                                    "appendix_key": unit["@별표키"], "sha256": sha256(path),
                                    "source_json": source.name, "source_json_sha256": sha256(source),
                                    "downloaded_at": datetime.now(timezone.utc).isoformat()})
            print(f"[별표 수집] {unit['별표제목']}", flush=True)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clean_text(value):
    if not isinstance(value, str):
        raise ValueError("법령 텍스트 필드가 문자열이 아닙니다.")
    # <개정 ...>와 법률 기호를 유지한다. HTML로 해석하지 않는다.
    return value.replace("\r\n", "\n").replace("\r", "\n").strip()


def as_list(value):
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def iso_date(value):
    return datetime.strptime(value, "%Y%m%d").date().isoformat()


def write_json(path, data):
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def ensure_output(input_dir, output_dir):
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if output_dir.is_relative_to(input_dir) or output_dir.is_relative_to(RAW_DATA_PATH):
        raise ValueError("원본 폴더 또는 data/raw 아래에 결과를 저장할 수 없습니다.")
    return input_dir, output_dir


def xml_to_data(element):
    """수집 JSON과 원본 XML의 구조 일치를 검증하기 위한 변환."""
    if not list(element) and not element.attrib:
        return element.text or ""
    result = {"@" + k: v for k, v in element.attrib.items()}
    if element.text and element.text.strip():
        result["#text"] = element.text
    for child in element:
        value = xml_to_data(child)
        if child.tag in result:
            if not isinstance(result[child.tag], list):
                result[child.tag] = [result[child.tag]]
            result[child.tag].append(value)
        else:
            result[child.tag] = value
    return result


def body_segments(article):
    """조·항·호·목의 본문을 순서대로 펼치고 원래 노드의 경로를 보존한다."""
    segments, pieces, cursor = [], [], 0

    def append(value, path, level, label):
        nonlocal cursor
        text = clean_text(value)
        if not text:
            return
        if pieces:
            cursor += 1
        segments.append({"path": path, "level": level, "label": label,
                         "text": text, "start": cursor, "end": cursor + len(text)})
        pieces.append(text)
        cursor += len(text)

    def walk(node, path, level):
        if not isinstance(node, dict):
            raise ValueError(f"{path}: 항·호·목 구조가 객체가 아닙니다.")
        append(node.get(level + "내용", ""), path + "." + level + "내용",
               level, node.get(level + "번호", ""))
        for child_level in LEVELS:
            for index, child in enumerate(as_list(node.get(child_level))):
                walk(child, f"{path}.{child_level}[{index}]", child_level)

    append(article.get("조문내용", ""), "조문내용", "조", article.get("조문번호", ""))
    for level in LEVELS:
        for index, child in enumerate(as_list(article.get(level))):
            walk(child, f"{level}[{index}]", level)
    return "\n".join(pieces), segments


def update_hierarchy(stack, heading):
    match = re.match(r"^제\s*\d+\s*([편장절관])\s+", heading)
    if not match:
        return stack
    rank = "편장절관".index(match.group(1))
    return [entry for entry in stack if entry["rank"] < rank] + [{"rank": rank, "text": heading}]


def text_segments(text, field):
    """빈 줄 경계로 분할하되 구분 공백까지 포함해 본문 오프셋을 보존한다."""
    ends = [m.end() for m in re.finditer(r"\n[ \t]*\n(?:[ \t]*\n)*", text)] + [len(text)]
    start, result = 0, []
    for end in sorted(set(ends)):
        if end > start:
            result.append({"path": f"{field}[{len(result)}]", "level": "문단", "label": "",
                           "start": start, "end": end, "text": text[start:end]})
        start = end
    return result


def preprocess_supplements(raw, version_id):
    result, seen = [], set()
    for index, unit in enumerate(as_list(raw["detail"].get("부칙", {}).get("부칙단위"))):
        key = unit["@부칙키"]
        if not isinstance(key, str) or not re.fullmatch(r"[0-9A-Za-z]+", key) or key in seen:
            raise ValueError("중복 또는 유효하지 않은 부칙 식별번호입니다.")
        seen.add(key)
        text = clean_text(unit.get("부칙내용", ""))
        promulgated = iso_date(unit["부칙공포일자"])
        candidate = bool(text) and promulgated <= raw["as_of"]
        segments = text_segments(text, "부칙내용")
        commencement = [s["text"].strip() for s in segments
                        if re.search(r"(?:제\s*\d+조\s*|[①-⑳]\s*)?\(시행일\)", s["text"])]
        result.append({"unit_id": f"{version_id}_supplement_{key}", "content_type": "supplement",
                       "source_unit_index": index, "supplement_key": key,
                       "promulgation_date": promulgated, "promulgation_number": unit["부칙공포번호"],
                       "title": text.splitlines()[0] if text else "부칙",
                       "body_text": text, "body_segments": segments, "structure": unit,
                       "effective_date": None, "commencement_sections": commencement,
                       "temporal_scope": "amendment_specific_supplement",
                       "embedding_candidate": candidate, "needs_manual_review": not candidate,
                       "warnings": ["공포일은 시행일이 아님. 시행일·경과조치는 원문 문구를 확인해야 함",
                                    "개정별 부칙이며 해당 사건에 적용할 부칙을 자동 확정하지 않음"]})
    return result


def parse_box_tables(text):
    """API의 테두리 표에서 물리적 행·열을 보존. 병합 셀 값을 추정하지 않는다."""
    result, start, cursor = [], None, 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if re.match(r"^[┌┏][─━]", stripped):
            if start is not None:
                raise ValueError("닫히지 않은 별표 테두리 표입니다.")
            start = cursor
        cursor += len(line)
        if start is not None and re.match(r"^[└┗][─━]", stripped):
            block = text[start:cursor]
            rows = []
            serialized = []
            for row_line in block.splitlines():
                row_line = row_line.strip()
                if not row_line:
                    continue
                if row_line[0] in "│┃" and row_line[-1] in "│┃":
                    cells = [c.strip() for c in re.split(r"[│┃]", row_line[1:-1])]
                    rows.append(cells)
                    serialized.append(" | ".join(cells))
                else:
                    # 병합 헤더와 행 구분선도 남긴다.
                    serialized.append(row_line)
            if not rows:
                raise ValueError("별표 표의 셀 본문이 없습니다.")
            result.append({"start": start, "end": cursor, "text": block,
                           "physical_rows": rows, "column_counts": sorted({len(r) for r in rows}),
                           "serialized_text": "\n".join(serialized),
                           "interpretation": "물리적 행 순서 및 빈 셀 유지. 병합 셀·복합 헤더를 임의 확정하지 않음"})
            start = None
    if start is not None:
        raise ValueError("별표 표의 닫힘 테두리가 없습니다.")
    return result


def content_inventory(text):
    # PDF의 줄바꿈·공백·표 테두리를 제외한 문자 수 비교. 행·열 관계 검증을 대체하지 않는다.
    return Counter(re.findall(r"[가-힣A-Za-z0-9]", text))


def preprocess_appendices(raw, version_id, source_hash=None):
    import pdfplumber

    result, seen = [], set()
    for index, unit in enumerate(as_list(raw["detail"].get("별표", {}).get("별표단위"))):
        key = unit["@별표키"]
        path = appendix_path(version_id, unit)
        if key in seen:
            raise ValueError("중복 별표 식별번호입니다.")
        seen.add(key)
        source_url = appendix_url(unit)
        if not path.exists():
            raise ValueError("별표 PDF가 없습니다. --download-appendices로 먼저 수집해주세요.")
        record = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        pdf_hash = sha256(path)
        if (record["sha256"] != pdf_hash or record["source_url"] != source_url
                or record["version_id"] != version_id or record["appendix_key"] != key):
            raise ValueError("별표 PDF의 원본 해시·출처·버전이 일치하지 않습니다.")
        if source_hash and record["source_json_sha256"] != source_hash:
            raise ValueError("별표 PDF 수집 당시 법령 원본 해시와 현재 원본이 다릅니다.")
        text = clean_text(unit.get("별표내용", ""))
        if not text:
            raise ValueError("API 별표 본문이 비어 있습니다. PDF 직접 정제 검토가 필요합니다.")
        marker=html.unescape(unit['별표제목']).strip()
        retired=bool(re.fullmatch(r'(?:삭제|\[[^\]]+\]\s*으로\s*이동)\s*(?:<[^>]*>)?',marker))
        is_form = key.endswith("F")
        tables = [] if retired or is_form else parse_box_tables(text)
        pages = []
        with pdfplumber.open(path) as pdf:
            for number, page in enumerate(pdf.pages, 1):
                pages.append({"page_number": number, "text": page.extract_text() or "",
                              "tables": [{"bbox": list(t.bbox), "rows": t.extract()}
                                         for t in page.find_tables()]})
                page.close()
        valid_pdf_text = all(p["text"].strip() for p in pages)
        effective = iso_date(unit["별표시행일자"])
        inventory_match = content_inventory(text) == content_inventory("\n".join(p["text"] for p in pages))
        candidate = not retired and not is_form and valid_pdf_text and inventory_match and effective <= raw["as_of"]
        # 문단을 표 내부에서 나누지 않도록 표 범위는 독립된 원자 단위로 처리.
        segments, cursor = [], 0
        for table_index, table in enumerate(tables):
            if cursor < table["start"]:
                segments.extend(text_segments(text[cursor:table["start"]], "별표본문"))
                # 직전 생성된 문단들의 상대 좌표를 원문 좌표로 이동.
                for segment in segments:
                    if not segment.get("absolute"):
                        segment["start"] += cursor
                        segment["end"] += cursor
                        segment["absolute"] = True
            segments.append({"path": f"별표표[{table_index}]", "level": "표", "label": "",
                             "start": table["start"], "end": table["end"], "text": table["text"],
                             "table_index": table_index, "absolute": True})
            cursor = table["end"]
        if cursor < len(text):
            tail = text_segments(text[cursor:], "별표본문")
            for segment in tail:
                segment["start"] += cursor
                segment["end"] += cursor
            segments.extend(tail)
        for number, segment in enumerate(segments):
            segment.pop("absolute", None)
            if segment["level"] != "표":
                segment["path"] = f"별표본문[{number}]"
        if sha256(path) != pdf_hash:
            raise ValueError("전처리 중 별표 PDF가 변경됐습니다.")
        result.append({"unit_id": f"{version_id}_appendix_{key}", "content_type": "appendix",
                       "source_unit_index": index, "appendix_key": key,
                       "title": unit["별표제목"], "appendix_number": unit["별표번호"],
                       "branch_number": unit["별표가지번호"], "effective_date": effective,
                       "body_text": text, "body_segments": segments, "tables": tables,
                       "structure": unit, "pdf_file": path.relative_to(ROOT).as_posix(),
                       "pdf_sha256": pdf_hash, "source_url": source_url,
                       "downloaded_at": record["downloaded_at"], "pdf_pages": pages,
                       "pdf_comparison": {"character_inventory_match": inventory_match,
                                          "scope": "한글·영문·숫자 문자 수 비교. 행·열 의미 관계를 자동 확정하는 검증은 아님"},
                       "embedding_candidate": candidate, "needs_manual_review": not candidate,
                       "warnings": ["표 전체를 분할하지 않고 검색에 사용. 복합 헤더·병합 셀은 원본 PDF와 대조",
                                    "PDF 자동 표 추출은 페이지 간 행·열을 완전히 복원하지 못할 수 있어 API 테두리 표를 함께 보존"]})
        if retired:
            result[-1].update(needs_manual_review=False,exclusion_reason='삭제·이동된 별표/서식: 현행 검색 제외',
                             warnings=['삭제·이동된 별표/서식의 원문·PDF·출처는 보존하고 현행 검색에서 제외'])
        elif is_form:
            result[-1].update(exclusion_reason='별지 신청·인가 서식: 참고용 보존, 검색 제외',
                             warnings=['별지 서식은 빈 작성란·직인 등 시각 요소가 있어 원문·PDF를 참고용으로 보존. 자동 표 해석과 기본 검색 제외'])
    return result


def preprocess(path):
    original_hash = sha256(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("document_type") != "law":
        raise ValueError("법령 원본 JSON이 아닙니다.")
    info, listing = raw["metadata"], raw["list_metadata"]
    xml = ET.fromstring(raw["response_xml"])
    if xml.tag != "법령" or xml_to_data(xml) != raw["detail"]:
        raise ValueError("원본 XML과 JSON 구조가 일치하지 않습니다.")
    xml_info = xml.find("기본정보")
    if xml_info is None or {e.tag: "".join(e.itertext()) for e in xml_info} != info:
        raise ValueError("법령 메타데이터와 원본 XML이 일치하지 않습니다.")
    if (info["법령ID"].lstrip("0") != listing["법령ID"].lstrip("0")
            or info["법령명_한글"] != listing["법령명한글"]
            or info["시행일자"] != listing["시행일자"]):
        raise ValueError("법령 목록과 본문 식별정보가 다릅니다.")
    law_id, master_id = listing["법령ID"], listing["법령일련번호"]
    if not law_id.isdigit() or not master_id.isdigit():
        raise ValueError("법령 식별번호가 유효하지 않습니다.")
    effective = iso_date(info["시행일자"])
    as_of = date.fromisoformat(raw["as_of"])
    if date.fromisoformat(effective) > as_of:
        raise ValueError("수집 기준일 이후 시행 법령입니다.")
    version_id = f"{law_id}_{master_id}_{info['시행일자']}"
    units = as_list(raw["detail"]["조문"]["조문단위"])
    if len(units) != raw["article_count"]:
        raise ValueError("원본 조문단위 개수와 메타데이터가 다릅니다.")
    articles, structural, stack, seen = [], [], [], set()
    for index, unit in enumerate(units):
        text, segments = body_segments(unit)
        unnumbered = (unit.get("조문여부") == "전문" and len(units) == 1
                      and unit.get("조문번호") == "0" and bool(text)
                      and not re.match(r"^제\s*\d+\s*[편장절관]\s+", text))
        if unit.get("조문여부") == "전문" and not unnumbered:
            stack = update_hierarchy(stack, text)
            structural.append({"source_unit_index": index, "text": text, "structure": unit})
            continue
        if unit.get("조문여부") != "조문" and not unnumbered:
            raise ValueError("알 수 없는 조문단위 유형입니다.")
        number = unit["조문번호"]
        branch = unit.get("조문가지번호", "")
        if not number.isdigit() or (branch and not branch.isdigit()):
            raise ValueError("조문번호 또는 가지번호가 유효하지 않습니다.")
        article_id = f"{version_id}_art_{number}_{branch or '0'}"
        if article_id in seen:
            raise ValueError("같은 법령 버전에서 중복 조문번호가 발견됐습니다.")
        seen.add(article_id)
        label = "전문" if unnumbered else f"제{int(number)}조" + (f"의{int(branch)}" if branch else "")
        deleted = bool(re.fullmatch(r"제\s*\d+\s*조(?:\s*의\s*\d+)?\s+삭제(?:\s*<[^>]*>)*\s*",
                                    clean_text(unit.get("조문내용", ""))))
        warnings = []
        article_effective = None
        try:
            article_effective = iso_date(unit.get("조문시행일자", ""))
        except (ValueError, TypeError):
            warnings.append("조문 시행일이 없거나 해석 불가: 검토 필요")
        if article_effective and date.fromisoformat(article_effective) > as_of:
            warnings.append("수집 기준일 이후 시행 조문: 검색 제외")
        if not text:
            warnings.append("본문이 비어 있음: 검색 제외")
        if deleted and any(s["level"] != "조" for s in segments):
            raise ValueError("삭제 조문에 하위 본문이 남아 있어 확인이 필요합니다.")
        needs_review = bool(warnings)
        if unnumbered:
            warnings.append("원문은 조 번호 없는 전문. API의 조문번호 0·조문키를 보존하며 제1조 또는 제0조로 만들지 않음")
        articles.append({"article_id": article_id, "source_unit_index": index,
                         "source_article_key": unit.get("@조문키"),
                         "article_number": number, "branch_number": branch,
                         "article_label": label, "article_title": clean_text(unit.get("조문제목", "")),
                         "effective_date": article_effective, "hierarchy": [s["text"] for s in stack],
                         "structure": unit, "body_text": text, "body_segments": segments,
                         "references": unit.get("조문참고자료", ""), "is_deleted": deleted,
                         "needs_manual_review": needs_review,
                         "embedding_candidate": not deleted and not needs_review,
                         "warnings": warnings})
    supplements = preprocess_supplements(raw, version_id)
    appendices = preprocess_appendices(raw, version_id, original_hash) if raw["detail"].get("별표") else []
    if sha256(path) != original_hash:
        raise ValueError("전처리 중 원본이 변경됐습니다.")
    return {"schema_version": 1, "document_type": "law", "version_id": version_id,
            "law_id": law_id, "master_id": master_id, "law_name": info["법령명_한글"],
            "effective_date": effective, "as_of": raw["as_of"],
            "version_scope": raw["version_scope"], "source": raw["source"],
            "source_url": raw["source_url"], "source_file": path.name,
            "raw_sha256": original_hash, "metadata": info, "list_metadata": listing,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "source_unit_count": len(units), "article_count": len(articles),
            "articles": articles, "structural_headings": structural,
            "supplements": supplements, "appendices": appendices,
            "additional_materials": {k: v for k, v in raw["detail"].items() if k not in ("기본정보", "조문")},
            "scope": "본칙·개정별 부칙·별표 포함. 개정문·제개정이유는 원본 자료로 보존하며 검색에서 제외",
            "warnings": ["수집 당시 현행 버전. 과거 사건에 적용할 구법 버전은 별도 확인 필요"]}


def run(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
    input_dir, output_dir = ensure_output(input_dir, output_dir)
    paths = sorted(input_dir.glob("[0-9]*.json"))
    if not paths:
        raise ValueError("법령 원본 파일이 없습니다.")
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"processed_at": datetime.now(timezone.utc).isoformat(), "saved": [], "failures": []}
    for path in paths:
        try:
            doc = preprocess(path)
            filename = f"{doc['version_id']}.json"
            write_json(output_dir / filename, doc)
            entry = {"file": filename, "law_name": doc["law_name"], "version_id": doc["version_id"],
                     "source_file": path.name, "raw_sha256": doc["raw_sha256"],
                     "source_unit_count": doc["source_unit_count"], "article_count": doc["article_count"],
                     "deleted_count": sum(a["is_deleted"] for a in doc["articles"]),
                     "manual_review_count": sum(a["needs_manual_review"] for a in doc["articles"]),
                     "embedding_candidate_count": sum(a["embedding_candidate"] for a in doc["articles"])}
            entry.update({"supplement_count": len(doc["supplements"]), "appendix_count": len(doc["appendices"]),
                          "supplement_candidate_count": sum(u["embedding_candidate"] for u in doc["supplements"]),
                          "appendix_candidate_count": sum(u["embedding_candidate"] for u in doc["appendices"])})
            report["saved"].append(entry)
            print(f"[정제] {doc['law_name']}: 조문 {entry['article_count']}개 / 검색 후보 {entry['embedding_candidate_count']}개")
        except (KeyError, TypeError, ValueError, OSError, ET.ParseError) as exc:
            report["failures"].append({"source_file": path.name, "error": f"{type(exc).__name__}: {exc}"})
    report["usage"] = "saved 목록의 articles/supplements/appendices에서 embedding_candidate를 확인하세요. 부칙 공포일과 시행일을 구분하고 별표 표 전체를 보존합니다."
    write_json(output_dir / REPORT_NAME, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--download-appendices", action="store_true", help="별표 PDF 신규 수집. 기존 원본은 덮어쓰지 않음")
    args = parser.parse_args()
    try:
        if args.download_appendices:
            download_appendices(args.input_dir)
        report = run(args.input_dir, args.output_dir)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"정제 실패: {exc}")
        return 1
    print(f"정제 완료: 저장 {len(report['saved'])}건 / 실패 {len(report['failures'])}건")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
