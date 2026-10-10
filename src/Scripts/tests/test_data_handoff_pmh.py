"""최종 구조 검증·전체 테스트·인계 문서 생성.

uv run python -m src.Scripts.tests.test_data_handoff_pmh
raw/기존 processed/청크는 읽기만 한다. 작성 대상은 최종 보고서와 인계서뿐이다.
unittest 발견 실행에서는 생성/전체 테스트 재귀 실행 없이 검증만 수행한다.
"""
import hashlib
import json
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.Scripts.chunk_metadata import validate_chunk
from src.Scripts.preprocess_guides import validate_source_coverage
from src.Scripts.tests.test_chunk_metadata_pmh import corpus

ROOT = Path(__file__).resolve().parents[3]
REPORT_PATH = ROOT / "data/processed/retrieval_evaluation/data_validation_report_pmh.json"
HANDOFF_PATH = ROOT / "docs/데이터_인계서_pmh.md"


def read(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_file(relative, expected=None):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
        raise ValueError(f"원본/입력 파일 누락 또는 경로 오류: {relative}")
    actual = file_hash(path)
    if expected is not None and expected != actual:
        raise ValueError(f"파일 지문 불일치: {relative}")
    return actual


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit():
    checks = []
    details = {}
    def check(name, action):
        try:
            details[name] = action()
            checks.append({"name": name, "passed": True})
        except (ValueError, KeyError, TypeError, OSError, AssertionError) as exc:
            checks.append({"name": name, "passed": False, "error": str(exc)})

    chunks = corpus()
    laws = read("data/processed/laws/law_preprocessing_report.json")
    guides = read("data/processed/guides/guide_preprocessing_report.json")
    precedents = read("data/processed/precedent_preprocessing_report.json")
    review = read("data/processed/precedent_relevance_review.json")
    consumer = read("data/processed/consumer_sources/consumer_preprocessing_report.json")
    collected = read("data/processed/consumer_sources/consumer_collection_report.json")
    law_docs = [read("data/processed/laws/" + e["file"]) for e in laws["saved"]]
    guide_docs = [read("data/processed/guides/" + e["file"]) for e in guides["saved"]]
    precedent_docs = [read(f"data/processed/precedents/{identifier}.json") for identifier in precedents["saved"]]
    consumer_docs = [read("data/processed/consumer_sources/" + e["file"]) for e in consumer["saved"]]

    def current_reports():
        for name, report in [("법령", laws), ("안내서", guides), ("판례", precedents), ("소비자 정제", consumer), ("소비자 수집", collected)]:
            require(not report["failures"], f"{name} 현재 보고서의 실패 기록 존재")
        return {"current_failure_count": 0, "note": "이전 raw 수집 보고서의 재시도 전 실패는 현재 실패와 구분"}
    check("current_pipeline_failures", current_reports)

    def raw_coverage():
        law_raw = {p.name for p in (ROOT/"data/raw/laws").glob("*.json") if p.stem[0].isdigit()}
        require(law_raw == {e["source_file"] for e in laws["saved"]}, "법령 원본 처리 기록 누락/추가")
        for entry in laws["saved"]:
            checked_file("data/raw/laws/" + entry["source_file"], entry["raw_sha256"])
        validate_source_coverage(guides)
        for doc in guide_docs:
            checked_file(doc["source_path"], doc["raw_sha256"])
        raw_ids = {p.stem for p in (ROOT/"data/raw/precedents").glob("*.json") if p.stem.isdigit()}
        require(raw_ids == {e["precedent_id"] for e in review["records"]}, "원본 판례 관련성 검토 기록 누락/추가")
        require(len(review["records"]) == len(raw_ids), "판례 검토 기록 중복")
        for row in review["records"]:
            require(bool(row.get("reason")), "판례 제외/선정 사유 누락")
        for identifier, digest in precedents["raw_sha256"].items():
            checked_file(f"data/raw/precedents/{identifier}.json", digest)
        referenced = set()
        history = []
        for path in sorted((ROOT/"data/raw/consumer_sources").glob("collection_report_*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            for entry in report["saved"]:
                checked_file(entry["raw_file"], entry["raw_sha256"])
                referenced.add(entry["raw_file"])
            history.append({"file": path.relative_to(ROOT).as_posix(), "failure_count": len(report["failures"])})
        for entry in collected["saved"]:
            checked_file(entry["raw_file"], entry["raw_sha256"])
            referenced.add(entry["raw_file"])
        files = {p.relative_to(ROOT).as_posix() for p in (ROOT/"data/raw/consumer_sources").iterdir()
                 if p.is_file() and p.name != "__init__.py" and not p.name.startswith("collection_report_")}
        require(files == referenced, "소비자 원본 중 수집 기록이 없거나 현재 파일이 없는 자료")
        return {"law_originals": len(law_raw), "guide_originals": len(guides["saved"]),
                "precedent_originals": len(raw_ids), "precedent_reviewed": len(review["records"]),
                "precedent_review_statuses": dict(Counter(r["status"] for r in review["records"])),
                "consumer_recorded_original_files": len(files), "consumer_collection_history": history}
    check("raw_source_coverage_and_hashes", raw_coverage)

    def appendix_sources():
        directory = ROOT/"data/raw/law_appendices"
        pdfs = {p.name for p in directory.glob("*.pdf")}
        records = {p.with_suffix(".pdf").name for p in directory.glob("*.json")}
        require(pdfs == records, "별표 PDF/원본 메타데이터 짝 불일치")
        used = {Path(u["pdf_file"]).name for d in law_docs for u in d["appendices"] if u.get("pdf_file")}
        require(pdfs == used, "별표 PDF 정제 연결 누락")
        for path in directory.glob("*.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            checked_file(path.with_suffix(".pdf").relative_to(ROOT), meta["sha256"])
            checked_file("data/raw/laws/"+meta["source_json"], meta["source_json_sha256"])
        return {"pdf_metadata_pairs": len(pdfs)}
    check("law_appendix_provenance", appendix_sources)

    def processed_coverage():
        for folder, expected, pattern in [("laws", {e["file"] for e in laws["saved"]}, "[0-9]*.json"),
                ("guides", {e["file"] for e in guides["saved"]}, "guide_*.json"),
                ("precedents", {identifier+".json" for identifier in precedents["saved"]}, "[0-9]*.json")]:
            controls = {"guide_preprocessing_report.json", "guide_visual_review.json"} if folder == "guides" else set()
            actual = {p.name for p in (ROOT/"data/processed"/folder).glob(pattern) if p.name not in controls}
            require(actual == expected, f"{folder} 정제 파일/보고서 불일치")
        control = {"consumer_collection_report.json", "consumer_preprocessing_report.json", "source_manifest.json", "parcel_terms_revision_comparison.json"}
        actual = {p.name for p in (ROOT/"data/processed/consumer_sources").glob("*.json") if p.name not in control}
        require(actual == {e["file"] for e in consumer["saved"]}, "소비자 정제 파일/보고서 불일치")
        aliases = {e["source_id"] for e in consumer["duplicate_aliases"]}
        require({d["source_id"] for d in consumer_docs} | aliases == {e["source_id"] for e in collected["saved"]}, "현재 수집 자료의 정제/별칭 기록 누락")
        for doc in consumer_docs:
            checked_file(doc["raw_file"], doc["raw_sha256"])
            if doc.get("derived_pdf_file"):
                checked_file(doc["derived_pdf_file"], doc["derived_pdf_sha256"])
        return {"laws": len(law_docs), "guides": len(guide_docs), "precedent_files": len(precedent_docs), "consumer_documents": len(consumer_docs)}
    check("processed_document_inventory", processed_coverage)

    def metadata():
        require(len(chunks) == len({c["chunk_id"] for c in chunks}), "중복 청크 ID")
        for chunk in chunks:
            validate_chunk(chunk)
            require(bool(chunk["context_header"].strip()), "빈 출처 헤더: "+chunk["chunk_id"])
            require(bool(chunk["domains"]), "분야 매핑 누락: "+chunk["chunk_id"])
        for source_path in {chunk["source_path"] for chunk in chunks}:
            checked_file(source_path)
        grouped = defaultdict(list)
        for c in chunks:
            grouped[hashlib.sha256(" ".join(c["text"].split()).encode()).hexdigest()].append(c["chunk_id"])
        duplicates = [ids for ids in grouped.values() if len(ids)>1]
        return {"chunk_count": len(chunks), "duplicate_chunk_ids": 0, "empty_text_or_header": 0,
                "unmapped_domains": 0, "source_url_missing": sum(not c.get("source_url") for c in chunks),
                "identical_body_groups": duplicates,
                "duplicate_body_policy": "공백 정규화 후 동일 본문. 출처·시점·문맥이 달라 자동 제외하지 않음"}
    check("chunk_metadata_and_duplicates", metadata)

    def chunk_reports():
        lr=read("data/chunk/laws/law_chunking_report.json")
        gr=read("data/chunk/guides/guide_chunking_report.json")
        cr=read("data/chunk/consumer_sources/consumer_chunking_report.json")
        counts=Counter(c["document_type"] for c in chunks)
        checked_file("data/chunk/laws/chunks.jsonl",lr["chunks_sha256"])
        checked_file("data/chunk/consumer_sources/chunks.jsonl",cr["chunks_sha256"])
        require(lr["chunk_count"] == counts["law"], "법령 청크 집계 불일치")
        require(cr["chunk_count"] == sum(counts[k] for k in ["administrative_rule","standard_terms","mediation_case"]), "소비자 청크 집계 불일치")
        require(gr["chunk_count"] == counts["official_guide"], "안내서 청크 집계 불일치")
        for entry in lr["laws"]:
            checked_file("data/processed/laws/"+entry["file"],entry["processed_sha256"])
        for entry in cr["inputs"]:
            checked_file("data/processed/consumer_sources/"+entry["file"],entry["processed_sha256"])
        guide_files={e["file"] for e in gr["saved"]}
        require(guide_files == {p.name for p in (ROOT/"data/chunk/guides").glob("*_chunks.json")}, "안내서 청크 파일/보고서 불일치")
        for entry in gr["saved"]:
            guide_chunk=read("data/chunk/guides/"+entry["file"])
            checked_file("data/processed/guides/"+guide_chunk["guide_id"]+".json",entry["processed_sha256"])
            require(entry["chunk_count"] == len(guide_chunk["chunks"]), "PDF별 청크 집계 불일치")
        baseline=read("data/processed/retrieval_evaluation/baseline_report.json")
        require(baseline["chunk_count"] == counts["precedent"], "판례 평가/청크 집계 불일치")
        return {"by_document_type":dict(counts),"guide_files":sorted(guide_files),"total":len(chunks)}
    check("chunk_reports_and_input_fingerprints", chunk_reports)

    def eligibility():
        ready={d["precedent_id"] for d in precedent_docs if d["embedding_candidate"] and not d["needs_manual_review"]}
        require(ready == {c["precedent_id"] for c in chunks if c["document_type"]=="precedent"}, "판례 검색 후보/청크 불일치")
        consumer_ready={d["source_id"] for d in consumer_docs if d["embedding_candidate"] and not d["needs_manual_review"]}
        require(consumer_ready == {c["source_id"] for c in chunks if c.get("source_id")}, "소비자 검색 후보/청크 불일치")
        excluded={e["source_id"] for e in consumer["excluded"]} | {e["source_id"] for e in consumer["duplicate_aliases"]}
        require(not excluded & {c.get("source_id") for c in chunks}, "제외 소비자 자료가 검색 청크에 포함")
        require(all(e.get("reason") for e in consumer["excluded"]), "소비자 제외 이유 누락")
        return {"precedent_ready":len(ready),"precedent_duplicates":precedents["duplicates"],
                "consumer_ready":len(consumer_ready),"consumer_excluded":consumer["excluded"],"consumer_duplicate_aliases":consumer["duplicate_aliases"]}
    check("eligibility_and_exclusion_records", eligibility)

    def warnings_preserved():
        target=next(c for c in chunks if c["chunk_id"]=="ftc_terms_11185_file_a321590d_terms_article_22_0000")
        require(target["source_anomaly"] and target["calculation_restricted_clause"] and target["warnings"], "택배 원문 경고 누락")
        require("200" in target["text"] and "원문 주의사항" in target["embedding_text"], "택배 원문/검색용 경고 누락")
        for c in chunks:
            if c["document_type"]=="precedent":
                for field in ["case_name","case_number","court","judgment_date","source_url"]:
                    require(c[field] in c["context_header"], "판례 출처 헤더 누락")
        return {"parcel_restricted_chunk_id":target["chunk_id"],"source_anomaly":target["source_anomaly"],
                "calculation_restricted_clause":target["calculation_restricted_clause"]}
    check("provenance_headers_and_parcel_warning", warnings_preserved)

    inventory=[]
    for folder in ["raw","processed","chunk"]:
        for p in sorted((ROOT/"data"/folder).rglob("*")):
            if p.is_file() and p.resolve()!=REPORT_PATH.resolve() and "__pycache__" not in p.parts:
                inventory.append({"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":file_hash(p)})
    coverage=read("data/processed/retrieval_evaluation/domain_coverage_report_pmh.json")
    current_chunk_fingerprint=hashlib.sha256(json.dumps(
        [(p.relative_to(ROOT).as_posix(),file_hash(p)) for p in sorted((ROOT/"data/chunk").rglob("*")) if p.is_file()],
        ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    check("domain_review_snapshot",lambda: (require(coverage["input_fingerprint"]==current_chunk_fingerprint,"분야 검토 이후 청크 변경: 재검토 필요") or {"question_count":coverage["question_count"],"review_statuses":dict(Counter(q["review_status"] for q in coverage["questions"]))}))
    return {"validation_scope":"수집본의 구조·출처·처리 기록·청크 일관성. 법률 답변 품질/현행 적용 검증 아님",
            "structural_passed":all(c["passed"] for c in checks),"checks":checks,"details":details,
            "inventory":inventory,"inventory_sha256":hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),
            "remaining_limits":["분야 검토: 기본 근거 28문항, 자료 보완 15문항, 추가 내용 검토 1문항, 원문 적용 보류 1문항인 현재 스냅샷",
                                "자료 추가·개정 후 분야별 내용 검토와 평가 보고서를 다시 갱신해야 함",
                                "PDF 모든 이미지/원문 시각 요소 및 신규 현행 개정의 전수 검증은 아님",
                                "공통 서비스 로더·임베딩·Qdrant 저장/검색·LLM 답변·MCP 연결은 미구현"]}


def render_handoff(report):
    counts=report["details"].get("chunk_reports_and_input_fingerprints",{}).get("by_document_type",{})
    suite=report.get("test_suite",{})
    coverage=read("data/processed/retrieval_evaluation/domain_coverage_report_pmh.json")
    baseline=read("data/processed/retrieval_evaluation/baseline_report.json")
    consumer_eval=read("data/processed/retrieval_evaluation/consumer_coverage_report.json")
    lines=["# 데이터 수집·전처리 인계서", "", "## 현재 상태와 인계 범위", "",
           f"- 검증 시각(UTC): {report.get('validated_at','미실행')}",
           f"- 구조 검증: {'통과' if report['structural_passed'] else '실패 항목 확인 필요'} · {len(report['checks'])}개 항목",
           f"- 전체 단위 테스트: {suite.get('tests_run',0)}개 · {'통과' if suite.get('successful') else '미실행 또는 실패'}",
           "- 수집·정제·청크 생성, 공통 메타데이터, 판례 출처 헤더, 초기 BM25 평가까지 구현했습니다.",
           "- 이 인계는 초기 데이터 스냅샷에 대한 것입니다. 부족 자료 보강과 실제 RAG 서비스 검증까지 완료했다는 의미가 아닙니다.","",
           "## 데이터 수와 읽을 파일", "", "| 자료 | 정제 문서 / 검색 대상 | 청크 수 | 로딩 형식 |", "|---|---|---:|---|",
           f"| 법령 | 8종 | {counts.get('law',0)} | `data/chunk/laws/chunks.jsonl` |",
           f"| 판례 | 정제 파일 29개 / 중복 제외 후보 27건 | {counts.get('precedent',0)} | `data/chunk/precedents/chunks.jsonl` |",
           f"| 안내서 | PDF 5개 | {counts.get('official_guide',0)} | `data/chunk/guides/guide_chunking_report.json`의 `saved[].file` → 파일별 `chunks` 배열 |",
           f"| 소비자 자료 | 정제 문서 37개 / 검색 대상 23개 | {sum(counts.get(k,0) for k in ['administrative_rule','mediation_case','standard_terms'])} | `data/chunk/consumer_sources/chunks.jsonl` |",
           f"| 합계 | 분야 간 공유 자료를 다시 합산하지 않음 | {sum(counts.values())} | JSONL 3개 + 안내서별 JSON 5개 |", "",
           "소비자 청크는 행정규칙 7개, 조정사례 98개, 표준약관 64개입니다. 파일 수·문서 수·청크 수는 서로 다른 집계입니다.","",
           "## 폴더와 원본 보존 정책", "",
           "| 위치 | 역할 |", "|---|---|",
           "| `data/raw/laws` | 법령 8종의 수집 API 응답 JSON과 수집 기록 |",
           "| `data/raw/law_appendices` | 별표 PDF 8개와 원본 메타데이터 JSON 8개. 정제 법령의 appendices와 연결 |",
           "| `data/raw/precedents` | 판례 원본 95건과 수집 기록. 전체 관련성 검토 후 29개 정제 파일, 27건 검색 후보 |",
           "| `data/raw/guides` | 주택임대차·해고근로자·중고거래 피해자·인터넷 쇼핑·택배 PDF 5개 |",
           "| `data/raw/consumer_sources` | 공정위 약관 HWP/HWPX·행정규칙 XML·KCA HTML·참고 PDF·이전 스냅샷·수집 기록 |",
           "| `data/processed/{laws,precedents,guides,consumer_sources}` | 정제 본문·구조·검색 후보/제외·검토 정보 |",
           "| `data/processed/retrieval_evaluation` | 검색·분야 충분성·최종 검증 보고서. 원문 데이터와 구분 |",
           "| `data/chunk/{laws,precedents,guides,consumer_sources}` | 임베딩 입력용 청크. 자료별 형식 유지 |", "",
           "수집 시 인증정보를 저장 전에 제거합니다. 이미 저장한 raw는 추가 정제를 위해 덮어쓰지 않습니다. 갱신본은 새 원본 식별자·해시·시점을 기록하고 processed와 chunk를 다시 생성합니다. raw 스냅샷은 인증정보 제거 이전의 최초 응답과 바이트 단위로 동일하다는 뜻이 아닙니다.",
           "폴더 개수는 같을 필요가 없습니다. 별표는 법령 정제/청크로 합류하며, 평가 보고서는 별도 위치에 있습니다. 이전 raw 수집 보고서의 실패는 현재 최신 보고서 실패와 구분합니다.","",
           "## 청크 로딩과 필드 계약", "",
           "1. JSONL은 빈 줄을 제외하고 한 줄씩 JSON 객체로 읽습니다.",
           "2. 안내서는 청크 보고서 `saved` 목록만 읽고 PDF별 JSON의 `chunks` 배열을 펼칩니다. 보고서 JSON을 청크로 읽지 않습니다.",
           "3. `chunk_id`는 청크 식별자입니다. `document_id`는 자료 유형과 원문 ID를 결합하며 법령은 개정본 `version_id`를 사용합니다.",
           "4. `embedding_text`를 임베딩 입력으로 사용하고 `text`, 출처 필드, 시점, 경고를 payload로 함께 보존합니다. 모델 토큰 한도는 별도로 확인합니다.",
           "5. 임베딩/재적재 전 현재 청크 해시와 검증 보고서를 대조합니다. ID 변경 없이 텍스트가 바뀌면 해당 벡터도 갱신해야 합니다.","",
           "| 필드 | 의미 |", "|---|---|",
           "| `metadata_version` | 공통 메타데이터 버전 1. 자료별 schema_version과 별개 |",
           "| `document_type` | law / precedent / official_guide / administrative_rule / mediation_case / standard_terms |",
           "| `document_id`, `title` | 원문 식별자와 문서 제목 |",
           "| `domains`, `domain_classification` | 복수 분야 후보 태그와 분류 근거. 기존 domain 원값 유지 |",
           "| `source_url`, `source_path` | 확인한 URL과 프로젝트 상대 원본 경로. URL 없는 PDF는 null |",
           "| `dates` | 시행일·조문/별표 시행일·공포일·판결일·조정일·개정일·자료 기준일 등을 ISO 날짜/null로 분리 |",
           "| `context_header`, `section_context_header` | 출처 헤더와 판례의 원래 구간 설명 |",
           "| `warnings`, `source_anomaly`, `calculation_restricted_clause` | 원문·추출 주의사항 및 계산 제한. 삭제하지 않음 |", "",
           "분야 태그는 문서/기존 분류 수준의 후보이며 모든 조문이 해당 질문의 근거라는 뜻이 아닙니다. 민법·형법·근로기준법의 일반 태그를 임의로 9개 분야에 모두 매핑하지 않습니다. 세부 계약/사건 적용 및 시점 판단은 별도입니다.","",
           "## 제외·중복·본문 구조 정책", "",
           "- 판례는 관련성 검토표의 사유를 보존합니다. 중복 대표 판례만 검색하며 당사자 주장·증거·법원 판단의 역할을 유지합니다.",
           "- 소비자 정제 문서 37개 중 14개는 참고용·이전 약관 등으로 검색 제외합니다. 별칭 1건은 대표 자료로 합칩니다. 2007년 택배 약관은 역사 자료로 보존하고 기본 검색에서 제외합니다.",
           "- 법령 삭제 조문·검토 보류 자료를 임의로 적재하지 않습니다. 개정별 부칙의 공포일을 시행일로 추정하지 않습니다.",
           "- 안내서 표지·목차 등 제외 페이지와 이유는 청크 보고서에 있습니다. 이미지 요약은 수동 설명이며 원문 전문 전사가 아닙니다.",
           "- 표·약관 조문 등 원자 단위는 문자 목표 크기를 넘을 수 있습니다. 동일 본문도 출처·시점·문맥이 다를 수 있어 자동 삭제하지 않습니다.","",
           f"현재 공백 정규화 후 동일 본문 그룹은 {len(report['details'].get('chunk_metadata_and_duplicates',{}).get('identical_body_groups',[]))}개이며 ID 중복과 구분해 검증 JSON에 기록했습니다. 원문 URL 없는 청크는 {report['details'].get('chunk_metadata_and_duplicates',{}).get('source_url_missing',0)}개로, 원본 경로를 보존합니다.", "",
           "## 원문 이상과 적용 제한", "",
           "택배 2020년 약관 제22조 제2항 제3호 나목은 `운송장기재운임액의 200`으로 끝나며 단위가 없습니다. 원본 HWP와 변환 PDF에서 확인했습니다. **200%로 임의 보정하거나 이 문장으로 배상액을 계산하지 않습니다.** 안내서의 표기와 대조가 필요하며 경고를 검색·답변 단계까지 전달해야 합니다. 현재 서비스 실행 단계의 계산 제한 기능은 미구현입니다.",
           "조정사례의 금액·이율·사실관계를 다른 사건에 그대로 적용하지 않습니다. 수집본을 현행 적용 가능 자료로 자동 간주하지 않습니다. PDF 원문 URL이 확인되지 않은 경우 기관 홈페이지를 원문 URL로 대체하지 않습니다.","",
           "## 검색·분야 검토 결과와 보강 목록", "",
           f"- 판례 10문항: BM25 Hit@3={baseline['hit_at_3']:.0%}, MRR@3={baseline['mrr_at_3']:.3f}.",
           f"- 소비자 12문항: 보강 포함 기대 출처 Hit@3={consumer_eval['after']['expected_source_hit_at_3']:.1%}, Hit@5={consumer_eval['after']['expected_source_hit_at_5']:.1%}. 판례 헤더 보완 후 Top 5가 12/12에서 11/12로 변경됐습니다.",
           "- 분야 검토: 선정 질문 45개. 자료 내용 검토와 검색 진단을 분리합니다. 키워드 적중만으로 충분하다고 판정하지 않습니다.", "",
           "| 분야 | 기본 근거 | 보완 | 추가 검토 | 적용 보류 |", "|---|---:|---:|---:|---:|"]
    for item in coverage["summary"]:
        c=item["statuses"]
        lines.append(f"| {item['domain']} | {c.get('supported',0)} | {c.get('partial',0)} | {c.get('needs_review',0)} | {c.get('restricted',0)} |")
    lines += ["", "분야별 세부 질문·청크 ID·부족 자료는 [분야 검토 보고서](../data/processed/retrieval_evaluation/domain_coverage_report_pmh.md)를 참조합니다. 원문을 이미 수집했는지 먼저 확인한 후 필요한 발췌·정제 또는 추가 수집을 결정합니다.", "",
              "우선 보강 후보는 방문판매법/계속거래 환급 산정기준, 전자상거래법 시행령, 최저임금법/시행령/해당 연도 고시, 주택임대차보호법 시행령의 지역·시점별 기준입니다. 민사집행 절차, 소비자 피해구제·신고 준비서류, 품목별 분쟁해결기준은 서비스 질문 범위를 정한 뒤 보강합니다.","",
              "## 재현 실행 순서", "", "프로젝트 루트에서 실행합니다. 필요한 의존성은 pyproject.toml/uv.lock에 기록되어 있습니다. 수집 API 인증은 로컬 환경변수로 설정하며 값은 공유하지 않습니다.", "", "```bash",
              "# 원본 및 관련성/시각 검토 기록을 준비한 후 정제",
              "uv run python src/Scripts/preprocess_laws.py",
              "uv run python src/Scripts/preprocess_precedents.py",
              "uv run python src/Scripts/preprocess_guides.py",
              "uv run python src/Scripts/preprocess_consumer_sources.py",
              "# 청크 생성",
              "uv run python src/Scripts/chunk_laws.py",
              "uv run python src/Scripts/chunk_guides.py",
              "uv run python src/Scripts/chunk_consumer_sources.py",
              "uv run python -m src.Scripts.tests.test_precedent_retrieval_pmh",
              "# 검색/분야 평가",
              "uv run python -m src.Scripts.tests.test_consumer_coverage_pmh",
              "uv run python -m src.Scripts.tests.test_domain_coverage_pmh",
              "# 최종 감사·전체 테스트·인계서 갱신",
              "uv run python -m src.Scripts.tests.test_data_handoff_pmh",
              "# 보고서 생성 없이 전체 테스트만 실행",
              'uv run python -m unittest discover -s src/Scripts/tests -t . -p "test*_pmh.py"', "```", "",
              "분야 검토 코드는 선정 근거의 본문 지문이 바뀌면 추가 검토로 표시합니다. 질문·근거 메모와 지문을 내용 확인 없이 갱신하지 않습니다. 원본 추가 시 수집 목록·검토표·보고서에 반영한 뒤 위 순서로 재생성합니다.","",
              "## 다음 담당자의 작업", "",
              "1. 이 스냅샷의 검증 결과와 부족 자료 범위를 확인합니다.",
              "2. 두 저장 형식을 지원하는 서비스 로더와 임베딩 실행·실패 처리·토큰 제한 검증을 구현합니다.",
              "3. Qdrant 적재·갱신·검색과 필터를 구현합니다. 출처 적중과 정확한 근거 청크 적중을 구분해 평가합니다.",
              "4. LLM 답변의 근거·출처 표시, 근거 부족 시 대응, 원문 경고 전달을 구현하고 MCP를 연결합니다.",
              "5. 미사용 질문으로 실제 RAG 답변·계산·시점·출처 정확성을 평가합니다. 단위 테스트 통과를 답변 정확성으로 해석하지 않습니다.","",
              "## 검증 산출물", "",
              "- [최종 검증 JSON](../data/processed/retrieval_evaluation/data_validation_report_pmh.json): 항목별 결과·파일 크기/해시·테스트 결과·제외 기록.",
              "- [검증 실행 파일](../src/Scripts/tests/test_data_handoff_pmh.py): 직접 실행 시 보고서와 이 문서를 작성하고 전체 테스트를 실행합니다.",
              "- 기존 수집·정제·청크 파일은 최종 감사에서 변경하지 않습니다. 생성 코드·검토 기록·보고서는 재현을 위해 함께 인계합니다."]
    return "\n".join(lines)+"\n"


def write_outputs(report):
    REPORT_PATH.parent.mkdir(parents=True,exist_ok=True)
    HANDOFF_PATH.parent.mkdir(parents=True,exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    HANDOFF_PATH.write_text(render_handoff(report),encoding="utf-8")


def evaluate():
    report=audit()
    report["validated_at"]=datetime.now(timezone.utc).isoformat()
    report["test_suite"]={"status":"not_run"}
    write_outputs(report)
    # 기존 검증 파일이 이 위치에 TemporaryDirectory를 만들므로 빈 작업 폴더를 준비한다.
    (ROOT / "tmp").mkdir(exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(ROOT/"src/Scripts/tests"),pattern="test*_pmh.py",top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    report["test_suite"]={"tests_run":result.testsRun,"successful":result.wasSuccessful(),
                          "failures":[str(t) for t,_ in result.failures],"errors":[str(t) for t,_ in result.errors],
                          "skipped":[str(t) for t,_ in result.skipped]}
    report["handoff_ready_for_initial_pipeline"]=report["structural_passed"] and result.wasSuccessful()
    write_outputs(report)
    print(json.dumps({"structural_passed":report["structural_passed"],"tests":report["test_suite"]},ensure_ascii=False))
    return 0 if report["handoff_ready_for_initial_pipeline"] else 1


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=audit()

    def test_all_structure_checks_pass(self):
        self.assertTrue(self.report["structural_passed"],self.report["checks"])

    def test_document_and_chunk_counts_are_consistent(self):
        counts=self.report["details"]["chunk_reports_and_input_fingerprints"]
        self.assertEqual(counts["total"],sum(counts["by_document_type"].values()))
        self.assertEqual(len(counts["guide_files"]),self.report["details"]["processed_document_inventory"]["guides"])

    def test_exclusion_and_original_warning_records_are_retained(self):
        details=self.report["details"]
        self.assertTrue(details["eligibility_and_exclusion_records"]["precedent_duplicates"])
        self.assertTrue(details["eligibility_and_exclusion_records"]["consumer_excluded"])
        self.assertTrue(details["provenance_headers_and_parcel_warning"]["calculation_restricted_clause"])

    def test_saved_report_matches_current_data_snapshot(self):
        saved=json.loads(REPORT_PATH.read_text(encoding="utf-8"))
        for key in ["inventory_sha256","checks","details"]:
            self.assertEqual(saved[key],self.report[key])

    def test_path_escape_and_wrong_fingerprint_fail(self):
        with self.assertRaises(ValueError):
            checked_file("../outside_workspace.json")
        with self.assertRaises(ValueError):
            checked_file("data/chunk/laws/chunks.jsonl","wrong_hash")


if __name__=="__main__":
    raise SystemExit(evaluate())
