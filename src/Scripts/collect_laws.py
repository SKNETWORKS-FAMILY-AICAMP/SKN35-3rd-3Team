"""현행 법령 8개를 국가법령정보센터에서 버전별 원본 JSON으로 수집한다.

uv run python src/Scripts/collect_laws.py
uv run python src/Scripts/collect_laws.py --name 민법
"""

import argparse
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from datetime import date, datetime, timezone, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "data/raw/laws"
LAW_NAMES = ["민법", "형법", "주택임대차보호법", "근로기준법", "근로기준법 시행령",
             "근로자퇴직급여 보장법", "전자상거래 등에서의 소비자보호에 관한 법률", "공인중개사법"]
API_BASE = "https://www.law.go.kr/DRF/"


def name_key(name):
    return re.sub(r"\s+", "", name)


def xml_to_data(element):
    """반복 노드·속성·빈 필드를 보존한다. 원문 XML도 함께 저장한다."""
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


def request_xml(session, endpoint, oc, **params):
    try:
        response = session.get(API_BASE + endpoint,
            params={"OC": oc, "target": "eflaw", "type": "XML", **params}, timeout=(10, 30))
        response.raise_for_status()
    except requests.RequestException as exc:
        status = exc.response.status_code if exc.response is not None else None
        raise RuntimeError(f"API 연결 실패: {type(exc).__name__}, HTTP={status}") from None
    try:
        # 응답을 진단 출력하지 않는다. URL 내 인증값도 저장하지 않는다.
        text = response.content.decode("utf-8-sig")
        text = re.sub(r"([?&](?:amp;)?OC=)[^&\s<>\"']+", r"\1REDACTED", text, flags=re.I)
        return ET.fromstring(text), text
    except (ET.ParseError, UnicodeError):
        raise RuntimeError("API가 정상 UTF-8 XML을 반환하지 않았습니다.") from None


def select_law(root, name):
    if root.tag != "LawSearch":
        raise ValueError("법령 목록 응답이 아닙니다. API 권한을 확인해주세요.")
    matches = [e for e in root.findall("law")
               if name_key(e.findtext("법령명한글", "")) == name_key(name)]
    if len(matches) != 1:
        raise ValueError(f"정확히 일치하는 현행 법령을 하나로 확정하지 못했습니다: {name}, {len(matches)}건")
    entry = matches[0]
    values = {c.tag: "".join(c.itertext()) for c in entry}
    for key in ["법령ID", "법령일련번호", "시행일자"]:
        if not values.get(key, "").isdigit():
            raise ValueError(f"법령 목록의 {key}가 유효하지 않습니다.")
    return values


def validate_body(root, listing, as_of):
    info = root.find("기본정보")
    if root.tag != "법령" or info is None:
        raise ValueError("법령 본문 응답이 아닙니다.")
    metadata = {c.tag: "".join(c.itertext()) for c in info}
    if name_key(metadata.get("법령명_한글", "")) != name_key(listing["법령명한글"]):
        raise ValueError("목록과 본문의 법령명이 다릅니다.")
    if metadata.get("법령ID", "").lstrip("0") != listing["법령ID"].lstrip("0"):
        raise ValueError("목록과 본문의 법령ID가 다릅니다.")
    effective = metadata.get("시행일자", "")
    if effective != listing["시행일자"]:
        raise ValueError("목록과 본문의 시행일자가 다릅니다.")
    effective_date = datetime.strptime(effective, "%Y%m%d").date()
    if effective_date > as_of:
        raise ValueError("기준일 이후 시행예정 법령입니다.")
    articles = root.findall("./조문/조문단위")
    if not articles or not any(e.findtext("조문내용", "").strip() for e in articles):
        raise ValueError("법령 본문에 조문이 없습니다.")
    return metadata, len(articles)


def write_json(path, data):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def collect(names, as_of, delay):
    load_dotenv(ROOT / ".env")
    oc = os.environ.get("LAW_API_OC", "").strip()
    if not oc:
        raise ValueError("LAW_API_OC 환경변수가 필요합니다.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {"as_of": as_of.isoformat(), "started_at": datetime.now(timezone.utc).isoformat(),
              "requested": names, "saved": [], "skipped": [], "failures": [],
              "scope": "조회 시점의 현행 법령. 과거 판례 적용 구법은 별도 수집해야 합니다."}
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=2, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        for name in names:
            try:
                root, _ = request_xml(session, "lawSearch.do", oc,
                    search=1, query=name, nw=3, display=100, page=1)
                listing = select_law(root, name)
                # 버전을 고정해 목록과 본문 조회 사이에 현행판이 바뀌어도 혼합하지 않는다.
                time.sleep(delay)
                body, original_xml = request_xml(session, "lawService.do", oc,
                    MST=listing["법령일련번호"], efYd=listing["시행일자"])
                metadata, article_count = validate_body(body, listing, as_of)
                filename = f"{listing['법령ID']}_{listing['법령일련번호']}_{listing['시행일자']}.json"
                path = OUTPUT_DIR / filename
                item = {"law_name": metadata["법령명_한글"], "law_id": listing["법령ID"],
                        "master_id": listing["법령일련번호"], "effective_date": listing["시행일자"],
                        "article_count": article_count, "file": filename}
                if path.exists():
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    if previous.get("response_xml") == original_xml:
                        report["skipped"].append(item)
                        print(f"[건너뜀] {name}")
                        continue
                    raise ValueError("같은 버전의 기존 원문과 새 응답이 다릅니다. 자동 덮어쓰기를 중단합니다.")
                write_json(path, {"schema_version": 1, "document_type": "law", "source": "국가법령정보센터",
                    "source_url": f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={listing['법령일련번호']}",
                    "collected_at": datetime.now(timezone.utc).isoformat(), "as_of": as_of.isoformat(),
                    "query": name, "version_scope": "current_at_collection", "list_metadata": listing,
                    "metadata": metadata, "article_count": article_count,
                    "detail": xml_to_data(body), "response_xml": original_xml})
                report["saved"].append(item)
                print(f"[저장] {name}: 조문단위 {article_count}개, 시행일 {listing['시행일자']}")
            except (RuntimeError, ValueError, OSError) as exc:
                report["failures"].append({"law_name": name, "error": str(exc)})
                print(f"[실패] {name}: {exc}")
            time.sleep(delay)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(OUTPUT_DIR / f"collection_report_{stamp}.json", report)
    print(f"완료: 저장 {len(report['saved'])}, 건너뜀 {len(report['skipped'])}, 실패 {len(report['failures'])}")
    return 1 if report["failures"] else 0


def korea_today():
    return datetime.now(timezone(timedelta(hours=9))).date()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", action="append", help="수집할 법령명. 여러 번 지정 가능")
    parser.add_argument("--as-of", type=date.fromisoformat, default=korea_today(),
                        help="미래 시행 법령을 제외하는 기준일(기본: 한국의 오늘). 과거 버전 조회 기능은 아님.")
    parser.add_argument("--delay", type=float, default=0.5)
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("delay는 0 이상이어야 합니다.")
    try:
        return collect(args.name or LAW_NAMES, args.as_of, args.delay)
    except (ValueError, OSError) as exc:
        print(f"수집 시작 실패: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
