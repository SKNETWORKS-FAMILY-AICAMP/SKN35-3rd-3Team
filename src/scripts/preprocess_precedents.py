"""검토 결과의 포함/부분 사용 판례를 정제한다.

uv run python src/scripts/preprocess_precedents.py
원본은 보존하고 data/processed/precedents에 결과를 저장한다.
"""

import hashlib
import json
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    PROCESSED_DATA_PATH,
    PROCESSED_PRECEDENTS_DATA_PATH,
    PROJECT_PATH,
    RAW_PRECEDENTS_DATA_PATH,
)


ROOT = PROJECT_PATH
REVIEW_PATH = PROCESSED_DATA_PATH / 'precedent_relevance_review.json'
OUTPUT_DIR = PROCESSED_PRECEDENTS_DATA_PATH
REPORT_PATH = PROCESSED_DATA_PATH / 'precedent_preprocessing_report.json'
TEXT_FIELDS = {
    "issues": "판시사항", "summary": "판결요지", "referenced_laws": "참조조문",
    "referenced_precedents": "참조판례", "full_text": "판례내용",
}

# 2026-10-10 원문 비교·발췌 검토 결과. 본문이 바뀌면 검토 승인을 재사용하지 않는다.
REVIEWED_HASHES = {
    "147076": "59df2e5a5981c3d726d04900ef23038ac85cca967c867fd4d1bc0a2ed1321232",
    "227759": "774cd24394e92992b9e321ba5bfd756224334a04b876afc56770a0caa7a002bf",
    "234477": "74d9ad73b0e354c5baaeebefaac3401a94ab04f29347183859ab366932f81115",
    "605583": "f1590661cf690d527c4c77b47ee0847025711363ccea366eba5264ef38612980",
    "600531": "a79c609cf0421cf939eb8e52b341e52bbdefc489d32a277a98c8fec9653dfb02",
    "614661": "144036fbf02fd3c5a009fb5cf07d5bbd6eeb409953fba444b334ce9985da59b0",
}
SPAN_ROLES = {
    "건물 임대차 사건의 전제사실": "civil_case_facts",
    "원상복구비와 원상복구기간의 판단": "civil_court_discussion",
    "보증금 공제와 잔액 판단": "civil_court_discussion",
    "명도 후 반환 지연손해금 판단": "civil_historical_interest",
    "심판범위": "procedural_scope",
    "피고인의 임금·퇴직금 주장": "party_argument",
    "법원의 임금·퇴직금 판단": "court_discussion_with_arguments",
    "임금·퇴직금 무죄 부분": "court_acquittal",
    "임금·퇴직금·해고예고수당 범죄사실": "court_recorded_criminal_facts",
    "임금·퇴직금 범죄사실": "court_recorded_criminal_facts",
    "근로기준법 범죄사실": "court_recorded_criminal_facts",
    "임금·퇴직금 공소기각 부분": "court_dismissal",
    "중고나라 미배송 사기 범죄사실": "court_recorded_criminal_facts",
    "중고나라 미배송 사기 증거": "evidence",
    "중고나라 미배송 사기 적용법조": "applied_law",
}
ROLE_LABELS = {
    "civil_case_facts": "학원 건물 임대차 사건의 사실관계: 주택 사건과 구분",
    "civil_court_discussion": "서울중앙지법 판단: 당사자 주장·증빙·특약 및 개별 인정액 포함",
    "civil_historical_interest": "2007년 하급심의 당시 이율 적용: 현행 이율과 구분",
    "procedural_scope": "심판 범위",
    "party_argument": "당사자 주장: 법원 판단과 구분",
    "court_discussion_with_arguments": "법원 검토: 공소사실·당사자 주장·하급심 및 당심 판단 포함",
    "court_acquittal": "법원의 무죄 판단",
    "court_dismissal": "법원의 공소기각 판단",
    "court_recorded_criminal_facts": "판결문에 기재된 범죄사실",
    "evidence": "증거 목록",
    "applied_law": "해당 쟁점 적용법조",
}


class TextParser(HTMLParser):
    """줄바꿈·표 셀 경계를 남기고 태그와 script/style만 제거한다."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.hidden += 1
        elif not self.hidden:
            if tag in {"br", "p", "div", "tr", "li", "h1", "h2", "h3"}:
                self.parts.append("\n")
            elif tag in {"td", "th"}:
                self.parts.append("\t")

    def handle_endtag(self, tag):
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        elif not self.hidden and tag in {"p", "div", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value):
    if not isinstance(value, str):
        raise ValueError("텍스트 필드는 문자열이어야 합니다.")
    parser = TextParser()
    parser.feed(value)
    parser.close()
    text = "".join(parser.parts).replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalized_heading(text):
    return re.sub(r"\s+", "", text)


def get_sections(text):
    matches = list(re.finditer(r"【([^】]+)】", text))
    if not matches:
        return [{"heading": "본문", "start": 0, "end": len(text), "text": text}]
    result = []
    if text[:matches[0].start()].strip():
        result.append({"heading": "서두", "start": 0, "end": matches[0].start(),
                       "text": text[:matches[0].start()]})
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        result.append({"heading": normalized_heading(match.group(1)),
                       "start": match.start(), "end": end, "text": text[match.start():end]})
    return result


# 확인한 원문의 제목 경계를 사용한다. 경계가 달라지면 자동 추측하지 않고 실패한다.
# start/end는 정제된 full_text 기준의 0-based, end-exclusive 위치이다.
PARTIAL_RULES = {
    "147076": [
        ("건물 임대차 사건의 전제사실", r"【이\s*유】\s*1\. 사안의 개요", r"(?m)^2\. 본소청구에 대한 판단", 0),
        ("원상복구비와 원상복구기간의 판단", r"(?m)^나\. 원상복구비용과 원상복구기간", r"(?m)^다\. 연체관리비", 0),
        ("보증금 공제와 잔액 판단", r"(?m)^가\. 임대차보증금반환청구에 대하여", r"(?m)^나\. 손해배상청구에 대하여", 0),
        ("명도 후 반환 지연손해금 판단", r"(?m)^다\. 반소청구의 결론", r"(?m)^4\. 결", 0),
    ],
    "227759": [
        ("심판범위", r"【이\s*유】1\. 이 법원의 심판범위", r"(?m)^2\. 항소이유의 요지", 0),
        ("피고인의 임금·퇴직금 주장", r"(?m)^나\) 공소외 2 회사 근로자들에 대한 근로기준법위반 및 근로자퇴직급여보장법위반의 점에 대하여", r"(?m)^다\) 위계공무집행방해의 점에 대하여", 0),
        ("법원의 임금·퇴직금 판단", r"(?m)^2\) 근로기준법위반 및 퇴직급여보장법위반의 점에 관한 판단", r"(?m)^3\) 위계공무집행방해의 점에 관한 판단", 0),
        ("임금·퇴직금 무죄 부분", r"【무죄부분】", None, 0),
    ],
    "234477": [
        ("임금·퇴직금·해고예고수당 범죄사실", r"(?m)^2\. 2018고단1928 -", r"(?m)^3\. 2018고단4098 -", 0),
        ("임금·퇴직금 범죄사실", r"(?m)^3\. 2018고단4098 -", r"(?m)^다\. 위계공무집행방해", 0),
        ("근로기준법 범죄사실", r"(?m)^4\. 2018고단6391 -", r"【증거의 요지】", 0),
        ("법원의 임금·퇴직금 판단", r"【피고인 및 변호인 주장에 관한 판단】", r"(?m)^2\. 위계공무집행방해죄가 성립하지 않는다는 주장", 0),
        ("임금·퇴직금 공소기각 부분", r"【공소기각 부분】", None, 0),
    ],
    "605583": [
        ("중고나라 미배송 사기 범죄사실", r"『2023고단1581』", r"『2024고단482』", 0),
        ("중고나라 미배송 사기 증거", r"『2023고단1581』", r"『2024고단482』", 1),
    ],
}


def partial_spans(identifier, text):
    if identifier not in PARTIAL_RULES:
        raise ValueError("부분 사용 판례의 검증된 구간 규칙이 없습니다.")
    spans = []
    for heading, start_pattern, end_pattern, occurrence in PARTIAL_RULES[identifier]:
        starts = list(re.finditer(start_pattern, text))
        if len(starts) != occurrence + 1 and identifier != "605583":
            raise ValueError(f"시작 경계가 없거나 중복됩니다: {heading}")
        if identifier == "605583" and len(starts) != 2:
            raise ValueError("중고거래 범죄사실·증거 경계의 개수가 변경되었습니다.")
        if len(starts) <= occurrence:
            raise ValueError(f"구간 경계가 없습니다: {heading}")
        start = starts[occurrence].start()
        if end_pattern:
            end_match = re.search(end_pattern, text[starts[occurrence].end():])
            if not end_match:
                raise ValueError(f"종료 경계가 없습니다: {heading}")
            end = starts[occurrence].end() + end_match.start()
        else:
            # 판사 이름·주석은 본문 마지막에 있을 수 있으므로 별도 요약하지 않는다.
            end = len(text)
        spans.append({"heading": heading, "start": start, "end": end,
                      "text": text[start:end], "role": SPAN_ROLES[heading]})
    if identifier == "605583":
        matches = list(re.finditer(r"형법 제347조 제1항\(사기의 점\)", text))
        if len(matches) != 1:
            raise ValueError("중고거래 사기 적용법조 경계가 변경되었습니다.")
        match = matches[0]
        spans.append({"heading": "중고나라 미배송 사기 적용법조", "start": match.start(),
                      "end": match.end(), "text": match.group(), "role": "applied_law"})
    return spans


def text_digest(text):
    return hashlib.sha256(re.sub(r"\s+", "", text).encode("utf-8")).hexdigest()


def preprocess(row, raw):
    identifier = row["precedent_id"]
    detail = raw["detail"]
    if str(detail.get("판례정보일련번호")) != identifier:
        raise ValueError("원본의 식별번호가 검토 목록과 다릅니다.")
    texts = {key: clean_text(detail.get(field, "")) for key, field in TEXT_FIELDS.items()}
    if not texts["full_text"]:
        raise ValueError("정제 후 본문이 비어 있습니다.")
    date = clean_text(detail.get("선고일자", ""))
    digits = re.sub(r"\D", "", date)
    date_iso = datetime.strptime(digits, "%Y%m%d").date().isoformat() if len(digits) == 8 else None
    sections = get_sections(texts["full_text"])
    is_partial = row["status"] == "부분 사용"
    selected = partial_spans(identifier, texts["full_text"]) if is_partial else sections
    retrieval_segments = []
    if is_partial:
        parts = []
        cursor = 0
        for s in selected:
            header = f"[{s['heading']} / {ROLE_LABELS[s['role']]}]"
            part = f"{header}\n{s['text'].strip()}"
            retrieval_segments.append({"start": cursor, "end": cursor + len(part),
                                       "heading": s['heading'], "role": s['role'],
                                       "context_header": header})
            parts.append(part)
            cursor += len(part) + 2
        retrieval_text = "\n\n".join(parts)
    else:
        # 참조 목록은 metadata로 보존하고, 검색용 본문은 요지·판시사항과 전문으로 구성한다.
        retrieval_text = "\n\n".join(f"[{title}]\n{texts[key]}" for key, title in
            [("issues", "판시사항"), ("summary", "판결요지"), ("full_text", "판례내용")] if texts[key])
    warnings = []
    if not texts["issues"] and not texts["summary"]:
        warnings.append("판시사항·판결요지 없음. 원문에 없는 요약을 생성하지 않았습니다.")
    if is_partial and identifier != "147076":
        warnings.append("관련 구간 발췌. 전체 사건의 형량·배상명령은 해당 쟁점만의 결과가 아닙니다.")
        warnings.append("당사자 주장·공소사실과 법원의 판단을 구분해서 읽어야 합니다.")
    if identifier == "147076":
        warnings.extend([
            "서울중앙지법 2007-05-31 학원 건물 임대차 하급심(원문 항소 표시). 주택 임대차 직접 판례·대법원 확정 판단으로 간주하지 않음. 사실관계·특약·증빙과 법원의 판단을 함께 발췌.",
            "통상손모 제외, 특약 합의 및 공사비·기간의 개별 판단 사례. 공사비 50% 및 5일은 이 사건 인정치로 일반 공제율·기간이 아님.",
            "원문 연 20%는 2007년 당시 소송촉진법 적용 이율. 현재 지연이자율로 사용하지 않음. 명도·이행 제공, 약정·송달·항쟁의 상당성 및 적용 시점별 현행 조문 대조 필요."])
    if date_iso is None:
        warnings.append("선고일자 정규화 실패. 원문 날짜를 확인해주세요.")
    digest = text_digest(texts["full_text"])
    reviewed = REVIEWED_HASHES.get(identifier) == digest
    review_note = None
    if reviewed:
        review_note = ("관련 발췌 구간과 주장·판단 맥락 확인 완료. 전체 사건 형량은 검색 입력에서 제외."
                      if is_partial else "동일 사건 두 자료를 비교: 공백·인용부호·표 경계 물음표만 다름. 대표는 614661.")
        if identifier == "147076":
            review_note = "학원 건물 사건의 전제사실·원상복구 특약 및 통상손모·견적과 공사기간·공제 잔액·명도 후 역사적 이율 판단을 원문 대조. 주택/상가 및 심급·현행 이율의 적용 범위 경고 보존."
    return {
        "schema_version": 1, "document_type": "precedent", "precedent_id": identifier,
        "case_name": clean_text(detail.get("사건명", "")),
        "case_number": clean_text(detail.get("사건번호", "")),
        "court": clean_text(detail.get("법원명", "")),
        "judgment_date": date_iso, "judgment_date_raw": date,
        "case_type": clean_text(detail.get("사건종류명", "")),
        "judgment_type": clean_text(detail.get("판결유형", "")),
        "source": raw.get("source", "국가법령정보센터"),
        "source_url": f"https://www.law.go.kr/LSW/precInfoP.do?precSeq={identifier}",
        "source_file": (RAW_PRECEDENTS_DATA_PATH / f"{identifier}.json").relative_to(PROJECT_PATH).as_posix(),
        "source_collected_at": raw.get("collected_at"),
        "collection_query": None if raw.get('collection_mode')=='explicit_id' else raw.get("query"), "domain": row["domain"],
        "relevance_status": row["status"], "relevance_reason": row["reason"],
        "texts": texts, "sections": sections, "selected_spans": selected,
        "retrieval_text": retrieval_text,
        "retrieval_segments": retrieval_segments,
        "retrieval_scope": "selected_spans" if is_partial else "full_document",
        "missing_fields": [field for key, field in TEXT_FIELDS.items() if not texts[key]],
        "warnings": warnings, "needs_manual_review": is_partial and not reviewed,
        "review_resolution": {"status": "reviewed" if reviewed else "not_applicable",
                              "review_date": ("2026-10-11" if identifier == "147076" else "2026-10-10") if reviewed else None,
                              "reviewed_full_text_sha256": digest if reviewed else None,
                              "note": review_note},
        "duplicate_candidates": row.get("duplicate_group", []),
        "duplicate_of": None, "embedding_candidate": True,
        "full_text_sha256": digest,
    }


def write_json(path, data):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def main():
    review = json.loads(REVIEW_PATH.read_text(encoding="utf-8"))
    rows = [r for r in review["records"] if r["status"] in {"포함", "부분 사용"}]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"processed_at": datetime.now(timezone.utc).isoformat(), "review_file": str(REVIEW_PATH),
              "selected_count": len(rows), "saved": [], "failures": [], "duplicates": [],
              "duplicate_candidates_retained": [], "manual_review": [], "raw_sha256": {}}
    documents = []
    for row in rows:
        identifier = row["precedent_id"]
        try:
            path = RAW_PRECEDENTS_DATA_PATH / f"{identifier}.json"
            raw_bytes = path.read_bytes()
            report["raw_sha256"][identifier] = hashlib.sha256(raw_bytes).hexdigest()
            documents.append(preprocess(row, json.loads(raw_bytes.decode("utf-8"))))
        except (ValueError, KeyError, OSError, TypeError) as exc:
            report["failures"].append({"precedent_id": identifier, "error": str(exc)})
    representatives = {}
    reviewed_pair = all(any(d["precedent_id"] == identifier and
                            d["full_text_sha256"] == REVIEWED_HASHES[identifier]
                            for d in documents) for identifier in ["600531", "614661"])
    for doc in sorted(documents, key=lambda d: d["precedent_id"]):
        # 같은 사건번호라도 법원·선고일이 다른 심급은 합치지 않는다.
        key = (doc["court"], doc["case_number"], doc["judgment_date_raw"],
               tuple(text_digest(doc["texts"][f]) for f in TEXT_FIELDS))
        if key in representatives:
            doc["duplicate_of"] = representatives[key]
            doc["embedding_candidate"] = False
            report["duplicates"].append({"precedent_id": doc["precedent_id"],
                                         "representative_id": doc["duplicate_of"]})
        else:
            representatives[key] = doc["precedent_id"]
        if reviewed_pair and doc["precedent_id"] == "600531":
            doc["duplicate_of"] = "614661"
            doc["embedding_candidate"] = False
            report["duplicates"].append({"precedent_id": "600531", "representative_id": "614661",
                                         "method": "reviewed_formatting_variants"})
        if doc["duplicate_candidates"] and not doc["duplicate_of"] and not (
            reviewed_pair and doc["precedent_id"] == "614661"
        ):
            others = [d for d in documents if d["precedent_id"] in doc["duplicate_candidates"]
                      and d["precedent_id"] != doc["precedent_id"]]
            if any(d["full_text_sha256"] != doc["full_text_sha256"] for d in others):
                doc["needs_manual_review"] = True
                doc["warnings"].append("같은 사건 메타데이터의 다른 자료와 본문이 달라 병합하지 않았습니다.")
                report["duplicate_candidates_retained"].append(doc["precedent_id"])
        write_json(OUTPUT_DIR / f"{doc['precedent_id']}.json", doc)
        report["saved"].append(doc["precedent_id"])
        if doc["needs_manual_review"]:
            report["manual_review"].append(doc["precedent_id"])
    # 실패한 ID의 오래된 산출물도 자동 삭제하지 않는다. 보고서 성공 ID만 사용한다.
    report["embedding_candidate_count"] = sum(d["embedding_candidate"] for d in documents)
    report["ready_without_manual_review_count"] = sum(
        d["embedding_candidate"] and not d["needs_manual_review"] for d in documents)
    report["usage"] = "saved 목록의 JSON만 읽고 embedding_candidate 및 needs_manual_review를 확인한 뒤 retrieval_text를 청크로 분할하세요. texts.full_text는 부분 사용 사건의 검색 입력으로 쓰지 마세요."
    write_json(REPORT_PATH, report)
    print(f"정제 완료: 대상 {len(rows)}, 저장 {len(documents)}, 실패 {len(report['failures'])}")
    print(f"중복 제외 {len(report['duplicates'])}, 추가 검토 {len(report['manual_review'])}")
    print(f"저장 위치: {OUTPUT_DIR}")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
