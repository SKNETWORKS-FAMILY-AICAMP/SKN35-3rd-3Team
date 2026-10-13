"""설정한 현행 법령를 국가법령정보센터에서 버전별 원본 JSON으로 수집한다.

uv run python src/scripts/collect_laws.py
uv run python src/scripts/collect_laws.py --name 민법
"""

import argparse
import json
import math
from contextlib import nullcontext
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

# 파일 직접 실행에서도 프로젝트 패키지를 불러옵니다.
if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.const.config import (
    PROJECT_PATH,
    RAW_LAWS_DATA_PATH,
)


ROOT = PROJECT_PATH
OUTPUT_DIR = RAW_LAWS_DATA_PATH
LAW_NAMES = ["민법", "형법", "주택임대차보호법", "근로기준법", "근로기준법 시행령",
             "근로자퇴직급여 보장법", "전자상거래 등에서의 소비자보호에 관한 법률", "공인중개사법",
             "방문판매 등에 관한 법률", "전자상거래 등에서의 소비자보호에 관한 법률 시행령",
             "최저임금법", "최저임금법 시행령", "주택임대차보호법 시행령",
             "할부거래에 관한 법률", "할부거래에 관한 법률 시행령",
             "소송촉진 등에 관한 특례법", "소송촉진 등에 관한 특례법 제3조제1항 본문의 법정이율에 관한 규정",
             "민사집행법", "민사집행규칙", "소비자기본법", "소비자기본법 시행령"]
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


def collect(names=None, as_of=None, delay=0.5, *, output_dir=None, env_path=None,
            oc=None, client=None, progress=None):
    """현행 법령을 수집·보존하고 수집 보고서 딕셔너리를 반환한다.

    API 호출 및 파일 저장을 수행한다. 제공한 client는 닫지 않는다.
    같은 버전의 다른 원문은 덮어쓰지 않고 failures에 기록한다.
    progress에는 진행 메시지를 받을 함수를 전달할 수 있다.
    """
    names = list(LAW_NAMES) if names is None else names
    if isinstance(names, (str, bytes)):
        raise TypeError("names는 법령명 목록이어야 합니다.")
    names = list(names)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("수집할 법령명이 필요합니다.")
    as_of = korea_today() if as_of is None else as_of
    if not isinstance(as_of, date) or isinstance(as_of, datetime):
        raise TypeError("as_of는 date 객체여야 합니다.")
    if isinstance(delay, bool) or not isinstance(delay, (int, float)) or not math.isfinite(delay) or delay < 0:
        raise ValueError("delay는 유한한 0 이상의 숫자여야 합니다.")
    if progress is not None and not callable(progress):
        raise TypeError("progress는 호출 가능한 함수여야 합니다.")
    output_dir = Path(output_dir if output_dir is not None else OUTPUT_DIR).resolve()
    if oc is None:
        load_dotenv(Path(env_path) if env_path is not None else ROOT / ".env")
        oc = os.environ.get("LAW_API_OC", "")
    if not isinstance(oc, str) or not oc.strip():
        raise ValueError("LAW_API_OC 환경변수가 필요합니다.")
    oc = oc.strip()
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {"as_of": as_of.isoformat(), "started_at": datetime.now(timezone.utc).isoformat(),
              "requested": names, "saved": [], "skipped": [], "failures": [],
              "scope": "조회 시점의 현행 법령. 과거 판례 적용 구법은 별도 수집해야 합니다."}
    with (requests.Session() if client is None else nullcontext(client)) as session:
        if client is None:
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
                path = output_dir / filename
                item = {"law_name": metadata["법령명_한글"], "law_id": listing["법령ID"],
                        "master_id": listing["법령일련번호"], "effective_date": listing["시행일자"],
                        "article_count": article_count, "file": filename}
                if path.exists():
                    previous = json.loads(path.read_text(encoding="utf-8"))
                    if previous.get("response_xml") == original_xml:
                        report["skipped"].append(item)
                        if progress is not None:
                            progress(f"[건너뜀] {name}")
                        continue
                    raise ValueError("같은 버전의 기존 원문과 새 응답이 다릅니다. 자동 덮어쓰기를 중단합니다.")
                write_json(path, {"schema_version": 1, "document_type": "law", "source": "국가법령정보센터",
                    "source_url": f"https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq={listing['법령일련번호']}",
                    "collected_at": datetime.now(timezone.utc).isoformat(), "as_of": as_of.isoformat(),
                    "query": name, "version_scope": "current_at_collection", "list_metadata": listing,
                    "metadata": metadata, "article_count": article_count,
                    "detail": xml_to_data(body), "response_xml": original_xml})
                report["saved"].append(item)
                if progress is not None:
                    progress(f"[저장] {name}: 조문단위 {article_count}개, 시행일 {listing['시행일자']}")
            except (RuntimeError, ValueError, OSError) as exc:
                report["failures"].append({"law_name": name, "error": str(exc)})
                if progress is not None:
                    progress(f"[실패] {name}: {exc}")
            time.sleep(delay)
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(output_dir / f"collection_report_{stamp}.json", report)
    return report


def korea_today():
    return datetime.now(timezone(timedelta(hours=9))).date()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", action="append", help="수집할 법령명. 여러 번 지정 가능")
    parser.add_argument("--as-of", type=date.fromisoformat, default=korea_today(),
                        help="미래 시행 법령을 제외하는 기준일(기본: 한국의 오늘). 과거 버전 조회 기능은 아님.")
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--env-file", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.delay < 0:
        parser.error("delay는 0 이상이어야 합니다.")
    try:
        report = collect(args.name or LAW_NAMES, args.as_of, args.delay,
                         output_dir=args.output_dir, env_path=args.env_file, progress=print)
        print(f"완료: 저장 {len(report['saved'])}, 건너뜀 {len(report['skipped'])}, 실패 {len(report['failures'])}")
        return 1 if report["failures"] else 0
    except (ValueError, OSError) as exc:
        print(f"수집 시작 실패: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
