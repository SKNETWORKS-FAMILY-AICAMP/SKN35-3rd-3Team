"""국가법령정보센터 판례를 수집해 원본 필드를 JSON으로 보관
보관경로 : data/raw/precedents

실행: 
uv run python src/scripts/collect_precedents.py --query 보증금반환 --limit 20
uv run python src/scripts/collect_precedents.py --query "부당해고" --limit 20
uv run python src/scripts/collect_precedents.py --query "중고거래" --limit 20
uv run python src/scripts/collect_precedents.py --query "물품대금 사기" --limit 20
uv run python src/scripts/collect_precedents.py --query "청약철회" --limit 20
uv run python src/scripts/collect_precedents.py --query "전자상거래" --limit 20
uv run python src/scripts/collect_precedents.py --query "임금체불" --limit 20
uv run python src/scripts/collect_precedents.py --id 147076

인증키 : .env의 LAW_API_OC
"""

import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "precedents"
API_BASE = "https://www.law.go.kr/DRF/"


def fields(element):
    """빈 필드와 본문 내 HTML 문자열을 포함해 API 필드를 보존한다."""
    return {child.tag: "".join(child.itertext()) for child in element}


def sanitize_listing(listing):
    """목록 링크의 OC 인증값만 제거하고 판례 본문은 건드리지 않는다."""
    result = dict(listing)
    for key, value in result.items():
        if isinstance(value, str) and "?" in value:
            parts = urlsplit(value)
            params = parse_qsl(parts.query, keep_blank_values=True)
            if any(name.lower() == "oc" for name, _ in params):
                result[key] = urlunsplit(parts._replace(query=urlencode(
                    [(name, val) for name, val in params if name.lower() != "oc"])))
    return result


def request_xml(session, endpoint, oc, **params):
    try:
        response = session.get(
            API_BASE + endpoint,
            params={"OC": oc, "target": "prec", "type": "XML", **params},
            timeout=(10, 30),
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        # requests 예외의 URL에는 인증값이 포함될 수 있으므로 출력하지 않는다.
        status = exc.response.status_code if exc.response is not None else None
        raise RuntimeError(f"API 요청 실패: {type(exc).__name__}, HTTP={status}") from None
    try:
        return ET.fromstring(response.content)
    except ET.ParseError:
        raise RuntimeError("API 응답이 올바른 XML이 아닙니다.") from None


def write_json(path, data):
    # 저장 중 중단되더라도 기존 JSON을 훼손하지 않는다.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def valid_saved(path, identifier):
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or not isinstance(data.get("detail"), dict):
            return False
        detail = data["detail"]
        body = detail.get("판례내용")
        return (
            data.get("precedent_id") == identifier
            and detail.get("판례정보일련번호") == identifier
            and isinstance(body, str) and bool(body.strip())
        )
    except (ValueError, OSError, AttributeError):
        return False


def collect(query, limit, delay, identifiers=None):
    if identifiers is not None:
        identifiers=list(dict.fromkeys(identifiers))
        if not identifiers or any(not i.isascii() or not i.isdigit() for i in identifiers):
            raise ValueError('판례 ID는 ASCII 숫자여야 합니다.')
        limit=len(identifiers)
    load_dotenv(PROJECT_ROOT / ".env")
    oc = os.environ.get("LAW_API_OC", "").strip()
    if not oc:
        raise RuntimeError(".env 또는 환경변수에 LAW_API_OC를 설정해주세요.")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "query": query if identifiers is None else None,
        "requested_ids": identifiers,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "limit": limit,
        "saved": [],
        "skipped": [],
        "failures": [],
    }
    seen = set()
    page = 1
    with requests.Session() as session:
        retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        session.mount("https://", HTTPAdapter(max_retries=retry))
        try:
            while len(seen) < limit:
                if identifiers is not None:
                    root=ET.Element('PrecSearch')
                    for identifier in identifiers:
                        entry=ET.SubElement(root,'prec')
                        ET.SubElement(entry,'판례일련번호').text=identifier
                else:
                    root = request_xml(
                        session, "lawSearch.do", oc, search=2, query=query,
                        display=min(100, limit), page=page,
                    )
                if root.tag != "PrecSearch":
                    raise RuntimeError("판례 목록 응답 형식이 아닙니다. 인증 및 API 권한을 확인해주세요.")
                entries = root.findall("prec")
                if not entries:
                    break
                before = len(seen)
                for entry in entries:
                    if len(seen) >= limit:
                        break
                    listing = sanitize_listing(fields(entry))
                    identifier = listing.get("판례일련번호", "").strip()
                    if not identifier.isdigit():
                        raise RuntimeError("목록 응답에 유효한 판례일련번호가 없습니다.")
                    if identifier in seen:
                        continue
                    seen.add(identifier)
                    path = OUTPUT_DIR / f"{identifier}.json"
                    if valid_saved(path, identifier):
                        report["skipped"].append(identifier)
                        print(f"[건너뜀] {identifier}")
                        continue
                    if path.exists():
                        report['failures'].append({'precedent_id':identifier,'error':'기존 raw 검증 실패: 덮어쓰기 금지'})
                        continue
                    time.sleep(delay)
                    try:
                        detail_root = request_xml(session, "lawService.do", oc, ID=identifier)
                        detail = fields(detail_root)
                        if not detail:
                            raise RuntimeError("API가 빈 본문 응답을 반환했습니다. 본문 제공 여부 또는 조회 권한을 확인해주세요.")
                        if detail.get("판례정보일련번호", "").strip() != identifier:
                            raise RuntimeError("본문 응답의 판례 식별번호가 요청과 다릅니다.")
                        if not detail.get("판례내용", "").strip():
                            raise RuntimeError("판례 본문이 비어 있습니다.")
                        write_json(path, {
                            "schema_version": 1,
                            "document_type": "precedent",
                            "precedent_id": identifier,
                            "source": "국가법령정보센터",
                            "source_url": f"https://www.law.go.kr/LSW/precInfoP.do?precSeq={identifier}",
                            "query": query if identifiers is None else None,
                            "collected_at": datetime.now(timezone.utc).isoformat(),
                            "list_metadata": listing if identifiers is None else {},
                            "collection_mode": "search" if identifiers is None else "explicit_id",
                            "detail": detail,
                        })
                        report["saved"].append(identifier)
                        print(f"[저장] {identifier}")
                    except (RuntimeError, OSError) as exc:
                        report["failures"].append({"precedent_id": identifier, "error": str(exc)})
                        print(f"[실패] {identifier}: {exc}")
                if len(seen) == before:
                    break
                page += 1
                time.sleep(delay)
        except RuntimeError as exc:
            report["failures"].append({"page": page, "error": str(exc)})
            print(f"[목록 조회 실패] {exc}")
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(OUTPUT_DIR / f"collection_report_{stamp}.json", report)
    print(f"완료: 저장 {len(report['saved'])}, 건너뜀 {len(report['skipped'])}, 실패 {len(report['failures'])}")
    print(f"저장 경로: {OUTPUT_DIR}")
    return 1 if report["failures"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="보증금반환", help="판례 본문 검색어")
    parser.add_argument("--id", action="append", help="확인한 판례일련번호로 직접 수집; 반복 지정 가능")
    parser.add_argument("--limit", type=int, default=20, help="검색 결과 중 처리할 최대 판례 수")
    parser.add_argument("--delay", type=float, default=0.5, help="요청 간 대기 시간(초)")
    args = parser.parse_args()
    if args.limit <= 0 or args.delay < 0 or not args.query.strip():
        parser.error("limit은 양수, delay는 0 이상, query는 비어 있지 않아야 합니다.")
    try:
        return collect(args.query, args.limit, args.delay, args.id)
    except (RuntimeError, OSError, ValueError) as exc:
        print(f"수집을 시작할 수 없습니다: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
