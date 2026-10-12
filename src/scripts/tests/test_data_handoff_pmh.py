"""최종 구조 검증·전체 테스트·인계 문서 생성.

uv run python -m src.scripts.tests.test_data_handoff_pmh
raw/기존 processed/청크는 읽기만 한다. 작성 대상은 최종 보고서와 인계서뿐이다.
unittest 발견 실행에서는 생성/전체 테스트 재귀 실행 없이 검증만 수행한다.
"""
import hashlib
import json
import unittest
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.const.config import (
    CHUNK_CONSUMER_SOURCES_DATA_PATH,
    CHUNK_DATA_PATH,
    CHUNK_GUIDES_DATA_PATH,
    CHUNK_LAWS_DATA_PATH,
    DATA_PATH,
    PROCESSED_CONSUMER_SOURCES_DATA_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_GUIDES_DATA_PATH,
    PROCESSED_LAWS_DATA_PATH,
    PROCESSED_PRECEDENTS_DATA_PATH,
    PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH,
    PROJECT_PATH,
    RAW_CONSUMER_SOURCES_DATA_PATH,
    RAW_LAWS_DATA_PATH,
    RAW_LAW_APPENDICES_DATA_PATH,
    RAW_LAW_HISTORY_DATA_PATH,
    RAW_PRECEDENTS_DATA_PATH,
)

from src.scripts.chunk_metadata import validate_chunk
from src.scripts.preprocess_guides import validate_source_coverage
from src.scripts.tests.test_chunk_metadata_pmh import corpus

ROOT = PROJECT_PATH
REPORT_PATH = PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH / 'data_validation_report_pmh.json'
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
    laws = read(PROCESSED_LAWS_DATA_PATH / 'law_preprocessing_report.json')
    guides = read(PROCESSED_GUIDES_DATA_PATH / 'guide_preprocessing_report.json')
    precedents = read(PROCESSED_DATA_PATH / 'precedent_preprocessing_report.json')
    review = read(PROCESSED_DATA_PATH / 'precedent_relevance_review.json')
    consumer = read(PROCESSED_CONSUMER_SOURCES_DATA_PATH / 'consumer_preprocessing_report.json')
    collected = read(PROCESSED_CONSUMER_SOURCES_DATA_PATH / 'consumer_collection_report.json')
    law_docs = [read(PROCESSED_LAWS_DATA_PATH / e["file"]) for e in laws["saved"]]
    guide_docs = [read(PROCESSED_GUIDES_DATA_PATH / e["file"]) for e in guides["saved"]]
    precedent_docs = [read(PROCESSED_PRECEDENTS_DATA_PATH / f'{identifier}.json') for identifier in precedents["saved"]]
    consumer_docs = [read(PROCESSED_CONSUMER_SOURCES_DATA_PATH / e["file"]) for e in consumer["saved"]]

    def current_reports():
        for name, report in [("법령", laws), ("안내서", guides), ("판례", precedents), ("소비자 정제", consumer), ("소비자 수집", collected)]:
            require(not report["failures"], f"{name} 현재 보고서의 실패 기록 존재")
        return {"current_failure_count": 0, "note": "이전 raw 수집 보고서의 재시도 전 실패는 현재 실패와 구분"}
    check("current_pipeline_failures", current_reports)

    def raw_coverage():
        law_raw = {p.name for p in RAW_LAWS_DATA_PATH.glob("*.json") if p.stem[0].isdigit()}
        require(law_raw == {e["source_file"] for e in laws["saved"]}, "법령 원본 처리 기록 누락/추가")
        for entry in laws["saved"]:
            checked_file(RAW_LAWS_DATA_PATH / entry["source_file"], entry["raw_sha256"])
        validate_source_coverage(guides)
        for doc in guide_docs:
            checked_file(doc["source_path"], doc["raw_sha256"])
        raw_ids = {p.stem for p in RAW_PRECEDENTS_DATA_PATH.glob("*.json") if p.stem.isdigit()}
        require(raw_ids == {e["precedent_id"] for e in review["records"]}, "원본 판례 관련성 검토 기록 누락/추가")
        require(len(review["records"]) == len(raw_ids), "판례 검토 기록 중복")
        for row in review["records"]:
            require(bool(row.get("reason")), "판례 제외/선정 사유 누락")
        for identifier, digest in precedents["raw_sha256"].items():
            checked_file(RAW_PRECEDENTS_DATA_PATH / f"{identifier}.json", digest)
        referenced = set()
        history = []
        for path in sorted(RAW_CONSUMER_SOURCES_DATA_PATH.glob("collection_report_*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            for entry in report["saved"]:
                checked_file(entry["raw_file"], entry["raw_sha256"])
                referenced.add(entry["raw_file"])
            history.append({"file": path.relative_to(ROOT).as_posix(), "failure_count": len(report["failures"])})
        for entry in collected["saved"]:
            checked_file(entry["raw_file"], entry["raw_sha256"])
            referenced.add(entry["raw_file"])
        files = {p.relative_to(ROOT).as_posix() for p in RAW_CONSUMER_SOURCES_DATA_PATH.iterdir()
                 if p.is_file() and p.name != "__init__.py" and not p.name.startswith("collection_report_")}
        require(files == referenced, "소비자 원본 중 수집 기록이 없거나 현재 파일이 없는 자료")
        return {"law_originals": len(law_raw), "guide_originals": len(guides["saved"]),
                "precedent_originals": len(raw_ids), "precedent_reviewed": len(review["records"]),
                "precedent_review_statuses": dict(Counter(r["status"] for r in review["records"])),
                "consumer_recorded_original_files": len(files), "consumer_collection_history": history}
    check("raw_source_coverage_and_hashes", raw_coverage)

    def appendix_sources():
        directory = RAW_LAW_APPENDICES_DATA_PATH
        pdfs = {p.name for p in directory.glob("*.pdf")}
        records = {p.with_suffix(".pdf").name for p in directory.glob("*.json")}
        require(pdfs == records, "별표 PDF/원본 메타데이터 짝 불일치")
        used = {Path(u["pdf_file"]).name for d in law_docs for u in d["appendices"] if u.get("pdf_file")}
        require(pdfs == used, "별표 PDF 정제 연결 누락")
        for path in directory.glob("*.json"):
            meta = json.loads(path.read_text(encoding="utf-8"))
            checked_file(path.with_suffix(".pdf").relative_to(ROOT), meta["sha256"])
            checked_file(RAW_LAWS_DATA_PATH / meta["source_json"], meta["source_json_sha256"])
        return {"pdf_metadata_pairs": len(pdfs)}
    check("law_appendix_provenance", appendix_sources)

    def consumer_appendix_review():
        import xml.etree.ElementTree as ET
        record_file=(PROCESSED_CONSUMER_SOURCES_DATA_PATH / "consumer_appendix_review_pmh.json").relative_to(ROOT).as_posix()
        record=read(record_file)
        record_hash=checked_file(record_file)
        checked_file(record['parent_raw_file'],record['parent_raw_sha256'])
        xml=ET.parse(ROOT/record['parent_raw_file']).getroot()
        doc=next(d for d in consumer_docs if d['source_id']==record['parent_source_id'])
        require({a['appendix_number'] for a in record['appendices']}=={'0003','0004'},'별표 대조 기록 범위 오류')
        pages=0
        question_marks=0
        for entry in record['appendices']:
            unit=next(u for u in doc['units'] if u.get('appendix_number')==entry['appendix_number'])
            node=next(n for n in xml.iter('별표단위') if n.findtext('별표번호')==entry['appendix_number'])
            require(unit['source_xml_text']==node.findtext('별표내용'),'별표 XML 본문 보존 실패')
            require(hashlib.sha256(unit['source_xml_text'].encode()).hexdigest()==unit['source_xml_text_sha256'],'XML 본문 지문 불일치')
            checked_file(entry['pdf_file'],entry['pdf_sha256'])
            require(doc['effective_date']==record['effective_date'],'대조 시행일 불일치')
            require(unit['source_comparison']['status']=='reviewed_against_collected_official_pdf','PDF 대조 상태 누락')
            require(unit['source_comparison']['source_effective_date']==record['effective_date'],'대조 시행일 누락')
            require(len(unit['pdf_tables'])==len(entry['pages']),'별표 대조 페이지 누락')
            require(all(t['character_inventory_match'] for t in unit['pdf_tables']),'PDF 문자 누락/중복')
            related=[c for c in chunks if c.get('source_id')==doc['source_id'] and c.get('unit_id')==unit['unit_id']]
            require(len(related)==1,'별표 단일 청크 정책 위반')
            c=related[0]
            require(c['text']==unit['text'] and c['pdf_tables']==unit['pdf_tables'],'별표 본문/좌표/병합 청크 불일치')
            require(c['pdf_source_url']==entry['pdf_source_url']==c['body_source_url'],'PDF 출처 링크 불일치')
            require(c['review_record_file']==record_file and c['review_record_sha256']==record_hash,'대조 기록 연결 오류')
            require(c['source_xml_text_sha256']==unit['source_xml_text_sha256'],'청크 XML 지문 누락')
            require(all(w in c['embedding_text'] for w in unit['warnings']),'별표 경고 임베딩 누락')
            require(not any(x.get('source_id')==entry['pdf_source_id'] for x in chunks),'첨부 PDF 중복 임베딩')
            if entry['appendix_number']=='0004':
                require(c['required_context_chunk_ids']==['consumer_dispute_standard_appendix_0003_0000'],'내용연수의 별표Ⅲ 연결 누락')
            pages+=len(entry['pages'])
            question_marks+=unit['source_comparison']['xml_question_mark_count']
        return {'reviewed_appendices':2,'reviewed_pdf_pages':pages,'preserved_xml_question_marks':question_marks,
                'review_record_file':record_file,'review_record_sha256':record_hash,'effective_date':record['effective_date']}
    check('consumer_appendix_pdf_review', consumer_appendix_review)

    def pilates_law_versions():
        from src.scripts.review_pilates_law_versions import REPORT, build_report
        comparison=read(REPORT.relative_to(ROOT))
        require(comparison==build_report(),'필라테스 시점 대조 기록 불일치')
        for relative,sha in comparison['inputs'].items():
            checked_file(relative,sha)
        historical={p.relative_to(ROOT).as_posix() for p in RAW_LAW_HISTORY_DATA_PATH.iterdir() if p.is_file()}
        require(historical=={p for p in comparison['inputs'] if p.startswith('data/raw/law_history/')},'과거 원문 대조 처리 누락')
        require(not any('law_history' in c['source_path'] for c in chunks),'과거 원문 기본 검색 편입')
        record_hash=checked_file(REPORT.relative_to(ROOT))
        for case in comparison['cases']:
            doc=next(d for d in consumer_docs if d['source_id']==case['source_id'])
            require(doc['law_version_review']['review_record_sha256']==record_hash,'사례 대조 기록 지문 누락')
            require(doc['decision_date'] is None,'미확인 조정일 임의 생성')
            selected=[c for c in chunks if c.get('source_id')==case['source_id']]
            require(bool(selected),'대조 사례 청크 누락')
            for c in selected:
                require(c['law_version_review']==doc['law_version_review'],'시점 대조 청크 불일치')
                require(all(w in c['embedding_text'] for w in doc['warnings']),'사례 적용 제한 임베딩 누락')
                require(c['required_context_chunk_ids']==case['required_context_chunk_ids'],'사례 연결 조문 누락')
                if c['role'] in {'reasoning','decision','cited_law'}:
                    require(case['supporting_law_context'] in c['embedding_text'],'약관법 참고 원문 문맥 누락')
        return {'cases':len(comparison['cases']),'historical_reference_files':len(historical),
                'comparison_rows':len(comparison['comparisons']),
                'identical_rows':sum(r['status']=='same_except_whitespace' for r in comparison['comparisons']),
                'different_definition_rows':sum(r['status']!='same_except_whitespace' for r in comparison['comparisons']),
                'review_record_file':REPORT.relative_to(ROOT).as_posix()}
    check('pilates_case_law_version_comparison',pilates_law_versions)

    def processed_coverage():
        for folder, expected, pattern in [("laws", {e["file"] for e in laws["saved"]}, "[0-9]*.json"),
                ("guides", {e["file"] for e in guides["saved"]}, "guide_*.json"),
                ("precedents", {identifier+".json" for identifier in precedents["saved"]}, "[0-9]*.json")]:
            controls = {"guide_preprocessing_report.json", "guide_visual_review.json"} if folder == "guides" else set()
            actual = {p.name for p in (PROCESSED_DATA_PATH/folder).glob(pattern) if p.name not in controls}
            require(actual == expected, f"{folder} 정제 파일/보고서 불일치")
        control = {"consumer_collection_report.json", "consumer_preprocessing_report.json", "source_manifest.json", "parcel_terms_revision_comparison.json", "consumer_appendix_review_pmh.json", "pilates_law_version_comparison_pmh.json"}
        actual = {p.name for p in PROCESSED_CONSUMER_SOURCES_DATA_PATH.glob("*.json") if p.name not in control}
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
        lr=read(CHUNK_LAWS_DATA_PATH / 'law_chunking_report.json')
        gr=read(CHUNK_GUIDES_DATA_PATH / 'guide_chunking_report.json')
        cr=read(CHUNK_CONSUMER_SOURCES_DATA_PATH / 'consumer_chunking_report.json')
        counts=Counter(c["document_type"] for c in chunks)
        checked_file(CHUNK_LAWS_DATA_PATH / "chunks.jsonl",lr["chunks_sha256"])
        checked_file(CHUNK_CONSUMER_SOURCES_DATA_PATH / "chunks.jsonl",cr["chunks_sha256"])
        require(lr["chunk_count"] == counts["law"], "법령 청크 집계 불일치")
        consumer_records=[json.loads(line) for line in (CHUNK_CONSUMER_SOURCES_DATA_PATH / 'chunks.jsonl').read_text(encoding="utf-8").splitlines() if line.strip()]
        pdf_records=[c for c in chunks if c["document_type"]=="official_guide" and not c.get("source_id")]
        require(cr["chunk_count"] == len(consumer_records), "소비자·공식 HTML 청크 집계 불일치")
        require(gr["chunk_count"] == len(pdf_records), "PDF 안내서 청크 집계 불일치")
        for entry in lr["laws"]:
            checked_file(PROCESSED_LAWS_DATA_PATH / entry["file"],entry["processed_sha256"])
        for entry in cr["inputs"]:
            checked_file(PROCESSED_CONSUMER_SOURCES_DATA_PATH / entry["file"],entry["processed_sha256"])
        guide_files={e["file"] for e in gr["saved"]}
        require(guide_files == {p.name for p in CHUNK_GUIDES_DATA_PATH.glob("*_chunks.json")}, "안내서 청크 파일/보고서 불일치")
        for entry in gr["saved"]:
            guide_chunk=read(CHUNK_GUIDES_DATA_PATH / entry["file"])
            checked_file(PROCESSED_GUIDES_DATA_PATH / (guide_chunk["guide_id"] + ".json"),entry["processed_sha256"])
            require(entry["chunk_count"] == len(guide_chunk["chunks"]), "PDF별 청크 집계 불일치")
        baseline=read(PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH / 'baseline_report.json')
        require(baseline["chunk_count"] == counts["precedent"], "판례 평가/청크 집계 불일치")
        return {"by_document_type":dict(counts),"guide_files":sorted(guide_files),"by_storage_group":{"laws":lr["chunk_count"],"precedents":counts["precedent"],"guides":len(pdf_records),"consumer_sources":len(consumer_records)},"html_official_guide_chunks":sum(c["document_type"]=="official_guide" for c in consumer_records),"total":len(chunks)}
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

    def supporting_sources():
        required={'방문판매 등에 관한 법률','전자상거래 등에서의 소비자보호에 관한 법률 시행령',
                  '최저임금법','최저임금법 시행령','주택임대차보호법 시행령',
                  '방문판매 등에 관한 법률 시행령','방문판매 등에 관한 법률 시행규칙','최저임금법 시행규칙',
                  '할부거래에 관한 법률','할부거래에 관한 법률 시행령',
                  '소송촉진 등에 관한 특례법','소송촉진 등에 관한 특례법 제3조제1항 본문의 법정이율에 관한 규정',
                  '민사집행법','민사집행규칙','소비자기본법','소비자기본법 시행령'}
        require(required.issubset({d['law_name'] for d in law_docs}),'추가 법령 정제 누락')
        notices={d['source_id']:d for d in consumer_docs}
        for identifier in ['continual_contract_refund','minimum_wage_2026','minimum_wage_simple_labor']:
            require(notices[identifier]['units'] and notices[identifier]['embedding_candidate'],'고시 본문/검색 후보 누락')
        for identifier in ['moel_wage_procedure','moel_wage_evidence_1000323154','kca_overseas_types','kca_overseas_checklist','kca_overseas_returns','kca_fitness_closure_gym','kca_fitness_closure_pilates','kca_damage_redress_procedure','kca_redress_requirements','kca_redress_submission','kca_mediation_procedure','easylaw_consumer_small_claim','easylaw_consumer_payment_order','easylaw_payment_order_application','easylaw_civil_enforcement','easylaw_housing_termination','easylaw_housing_restoration','easylaw_housing_distribution']:
            doc=notices[identifier]
            require(doc['embedding_candidate'] and doc['scope_included'] and doc['scope_excluded'],'공식 안내 처리·제외 범위 누락')
            require(doc['effective_date'] is None,'공식 안내 답변일을 시행일로 대체함')
            if identifier.startswith('easylaw_'):
                related=[c for c in chunks if c.get('source_id')==identifier]
                require(len(related)==len(doc['units']),'민사 회수 안내 단위·청크 누락')
                for c in related:
                    require(c['information_as_of']==doc['information_as_of'],'민사 안내 기준일 누락')
                    require(all(w in c['embedding_text'] for w in doc['warnings']),'민사 안내 개정 공지·회수 제한 임베딩 누락')
            if identifier in {'kca_redress_requirements','kca_redress_submission','kca_mediation_procedure'}:
                related=[c for c in chunks if c.get('source_id')==identifier]
                require(len(related)==len(doc['units']),'소비자 절차 안내 청크 누락')
                require(doc['information_as_of'] is None,'소비자 절차 안내 작성일 임의 생성')
                for c in related:
                    require(c.get('required_context_chunk_ids'),'소비자 절차 관련 법령 연결 누락')
                    require(all(w in c['embedding_text'] for w in c['warnings']),'소비자 절차 제한사항 임베딩 누락')
        chunk_ids={c['chunk_id'] for c in chunks}
        for c in chunks:
            require(set(c.get('required_context_chunk_ids',[])) <= chunk_ids,'공식 안내의 관련 법령 청크 누락')
        wage=notices['minimum_wage_2026']
        checked_file(wage['derived_pdf_file'],wage['derived_pdf_sha256'])
        target=[c for c in chunks if c.get('source_id')=='minimum_wage_2026']
        require(len(target)==1 and target[0]['valid_until']=='2026-12-31','최저임금 고시 적용 종료일 누락')
        require(target[0]['derived_pdf_sha256']==wage['derived_pdf_sha256'],'최저임금 고시 PDF 연결 누락')
        for c in chunks:
            for identifier in c.get('required_context_unit_ids',[]):
                require(any(x.get('unit_id')==identifier for x in chunks),'최저임금 적용 부칙 청크 연결 누락')
        return {'added_law_names':sorted(required),'notice_ids':['continual_contract_refund','minimum_wage_2026','minimum_wage_simple_labor'],
                'minimum_wage_period':[wage['valid_from'],wage['valid_until']]}
    check('supporting_sources_and_validity_period',supporting_sources)

    inventory=[]
    for folder in ["raw","processed","chunk"]:
        for p in sorted((DATA_PATH/folder).rglob("*")):
            if p.is_file() and p.resolve()!=REPORT_PATH.resolve() and "__pycache__" not in p.parts:
                inventory.append({"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":file_hash(p)})
    coverage=read(PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH / 'domain_coverage_report_pmh.json')
    current_chunk_fingerprint=hashlib.sha256(json.dumps(
        [(p.relative_to(ROOT).as_posix(),file_hash(p)) for p in sorted(CHUNK_DATA_PATH.rglob("*")) if p.is_file()],
        ensure_ascii=False,sort_keys=True).encode()).hexdigest()
    check("domain_review_snapshot",lambda: (require(coverage["input_fingerprint"]==current_chunk_fingerprint,"분야 검토 이후 청크 변경: 재검토 필요") or {"question_count":coverage["question_count"],"review_statuses":dict(Counter(q["review_status"] for q in coverage["questions"]))}))
    return {"validation_scope":"수집본의 구조·출처·처리 기록·청크 일관성. 법률 답변 품질/현행 적용 검증 아님",
            "structural_passed":all(c["passed"] for c in checks),"checks":checks,"details":details,
            "inventory":inventory,"inventory_sha256":hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),
            "remaining_limits":["분야 내용 검토 상태: "+json.dumps(dict(Counter(q["review_status"] for q in coverage["questions"])),ensure_ascii=False),
                                "자료 추가·개정 후 분야별 내용 검토와 평가 보고서를 다시 갱신해야 함",
                                "PDF 모든 이미지/원문 시각 요소 및 신규 현행 개정의 전수 검증은 아님",
                                "공통 서비스 로더·임베딩·Qdrant 저장/검색·LLM 답변·MCP 연결은 미구현"]}


def render_handoff(report):
    counts=report["details"].get("chunk_reports_and_input_fingerprints",{}).get("by_document_type",{})
    stored=report["details"]["chunk_reports_and_input_fingerprints"]["by_storage_group"]
    html_guides=report["details"]["chunk_reports_and_input_fingerprints"]["html_official_guide_chunks"]
    suite=report.get("test_suite",{})
    coverage=read(PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH / 'domain_coverage_report_pmh.json')
    review_counts=Counter(q["review_status"] for q in coverage["questions"])
    baseline=read(PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH / 'baseline_report.json')
    consumer_eval=read(PROCESSED_RETRIEVAL_EVALUATION_DATA_PATH / 'consumer_coverage_report.json')
    docs=report['details']['processed_document_inventory']
    eligibility=report['details']['eligibility_and_exclusion_records']
    lines=["# 데이터 수집·전처리 인계서", "", "## 현재 상태와 인계 범위", "",
           f"- 검증 시각(UTC): {report.get('validated_at','미실행')}",
           f"- 구조 검증: {'통과' if report['structural_passed'] else '실패 항목 확인 필요'} · {len(report['checks'])}개 항목",
           f"- 전체 단위 테스트: {suite.get('tests_run',0)}개 · {'통과' if suite.get('successful') else '미실행 또는 실패'}",
           "- 수집·정제·청크 생성, 공통 메타데이터, 판례 출처 헤더, 초기 BM25 평가까지 구현했습니다.",
           "- 이 인계는 초기 데이터 스냅샷에 대한 것입니다. 부족 자료 보강과 실제 RAG 서비스 검증까지 완료했다는 의미가 아닙니다.","",
           "## 데이터 수와 읽을 파일", "", "| 자료 | 정제 문서 / 검색 대상 | 청크 수 | 로딩 형식 |", "|---|---|---:|---|",
           f"| 법령 | {docs['laws']}종 | {counts.get('law',0)} | `data/chunk/laws/chunks.jsonl` |",
           f"| 판례 | 정제 파일 {docs['precedent_files']}개 / 중복 제외 후보 {eligibility['precedent_ready']}건 | {counts.get('precedent',0)} | `data/chunk/precedents/chunks.jsonl` |",
           f"| 안내서 | PDF 5개 | {stored['guides']} | `data/chunk/guides/guide_chunking_report.json`의 `saved[].file` → 파일별 `chunks` 배열 |",
           f"| 소비자 자료 | 정제 문서 {docs['consumer_documents']}개 / 검색 대상 {eligibility['consumer_ready']}개 | {stored['consumer_sources']} | `data/chunk/consumer_sources/chunks.jsonl` |",
           f"| 합계 | 분야 간 공유 자료를 다시 합산하지 않음 | {sum(counts.values())} | JSONL 3개 + 안내서별 JSON 5개 |", "",
           f"소비자·공식 HTML 청크는 행정규칙 {counts.get('administrative_rule',0)}개, 조정사례 {counts.get('mediation_case',0)}개, 표준약관 {counts.get('standard_terms',0)}개, 공식 HTML 안내 {html_guides}개입니다. 파일 수·문서 수·청크 수는 서로 다른 집계입니다.","",
           "소비자분쟁해결기준 별표 Ⅲ(품질보증기간·부품보유기간)과 별표 Ⅳ(내용연수표)를 추가 정제했습니다. 각 표는 조건·비고·병합 관계의 손실을 막기 위해 원문 전체를 한 청크로 보존합니다. 수집된 2025-12-18 시행본 공식 PDF 4페이지를 시각 대조했고, 표 셀 좌표·병합·공통 조건·빈 셀을 보존합니다. XML의 물음표 70개를 일괄 치환하지 않고 XML 본문은 source_xml_text에 보존하며 PDF 본문을 검색에 사용합니다. 대조 기록은 consumer_appendix_review_pmh.json이고 원본 PDF·XML 지문에 연결합니다. 별표Ⅳ는 required_context_chunk_ids로 별표Ⅲ에 연결하며 임베딩 전 토큰 제한을 확인해야 합니다.","",
           "45문항 분야 검토 중 품질보증 질문에 소비자기본법 시행령 제8조·별표 1의 일반적 해결기준을 추가 연결했습니다. 구매일·인도일·교환일의 기산 원칙, 판매일 확인이 곤란한 경우 제조일/수입통관일 이후 3월 기준 및 그 날짜가 미표시된 경우 사업자의 판매일 입증 예외를 원문 그대로 보존합니다. 별표 1의 해당 PDF 페이지는 시각 대조했고, 기존 고시 별표 Ⅲ·Ⅳ도 대조하여 구매일 증명 질문의 기본 근거를 확보했습니다. 개별 제품·계약 적용과 최신 개정 전수 검증은 별도입니다. 검색 비교는 양쪽 모두 현재 선정 근거를 사용합니다.","",
           "한국소비자원의 피해구제 요건·신청서 양식 안내, 접수 방법·증빙서류 안내, 분쟁조정 절차·성립 효력 안내 HTML 3건을 추가해 14개 청크로 보존합니다. 소비자기본법·시행령 원문(2026-09-11 시행본)을 수집·정제하고 기존 피해구제 절차/제외 조건과 새 안내의 연결 법령을 함께 확인하도록 구성했습니다. 전문분야 필수제출 조건과 일반분야를 구분하고, 온라인 신청의 6개월 조건을 소멸시효로 해석하지 않도록 경고합니다. 특정 사건 팝업은 일반 안내에서 제외했고 첨부 HWP/DOC 양식·신청 기능·영상은 미검증 범위로 기록했습니다. 안내 작성일은 확인되지 않아 null로 보존하며 주소·접수 방식은 신청 전 재확인해야 합니다. 합의권고와 조정의 효력을 구분하고 법 제67조의 수락·수락 간주 및 불수락 조건을 연결합니다.","",
           "별표 Ⅱ는 귀금속·보석 및 전자제품·사무용기기 품목을 추가했습니다. 전자제품의 연속 표 2개와 비고를 함께 보존하며 모든 품목이나 개인 간 중고거래에 일괄 적용하지 않습니다. 민법 제140·141조와 근로기준법 시행령 제7조의2의 수집본을 선정 근거에 연결했습니다.","",
           "헬스장·필라테스 HWPX 환급 표 3개는 셀 좌표·병합 범위를 검증해 해지 주체, 이용 전후 조건, 원문 수식, 공통 정의를 행별로 연결합니다. 원문 본문은 보존하고 생활 표현 별칭은 검색 점수에만 사용합니다. 법률 적용과 환급액 계산은 검증 범위가 아닙니다.","",
           f"전처리 전후 고정 근거 비교: 45문항 선정 청크 {coverage['preprocessing_comparison']['before_chunk_hits']} → {coverage['preprocessing_comparison']['after_chunk_hits']}건. 하락 질문: {', '.join(coverage['preprocessing_comparison']['regressed']) or '없음'}. 수정 전 근거 ID를 고정하여 새 근거 추가에 따른 평가 변화와 구분합니다.","",
           "고용노동부 노동포털 신고·처리 절차와 1350 준비자료 안내 HTML 2건을 보강했습니다. `document_type=official_guide`이며 소비자 자료 JSONL에 저장합니다. PDF 안내서 5개와 저장 위치별 집계를 구분합니다. 준비자료는 예시이고 증빙 부족 시 조사 안내도 보존합니다. 상담 답변일은 `information_as_of`이며 법령 시행일로 간주하지 않습니다. 메뉴·다른 상담·대지급금 등 제외 범위와 이유는 정제 문서에 기록합니다.", "",
           "해외직구 거래 유형·체크리스트·수령/반품 단계의 KCA 공식 HTML 3건을 추가했습니다. 직접배송·배송대행·구매대행(쇼핑몰형/위임형)의 정의·과정과 비교표를 보존하며, 반품 사유·운송 구간에 따른 비용·보상 조건을 구분합니다. 국내 구매대행의 청약철회 안내는 `required_context_chunk_ids`로 전자상거래법 제17·18조에 연결했고 해당 예외 주의사항을 `embedding_text`에도 포함합니다. 표는 `official_table`에 셀·열 제목을 보존합니다. 게시·수정일 미표시로 안내 기준일·시행일은 null을 유지합니다. 관세·지역 세금 및 개별 해외 국가의 준거법·약정은 이번 검증 범위 밖입니다.", "",
           "헬스장·필라테스 폐업 대응 및 소비자원 피해구제 절차 공식 HTML 3건을 보강했습니다. 할부 조건·잔여금 지급 거절, 잔여 횟수·해지일·내용증명 증빙, 상담→신청→조사→합의권고→조정 절차를 보존합니다. 사업자 부도·폐업으로 연락 불가능하거나 소재파악이 안 되는 경우의 피해구제 제외 조건을 별도 청크로 보존하고 `required_context_chunk_ids` 및 임베딩 본문 경고에 연결합니다. 헬스장 게시일은 2024-12-11, 필라테스 웹진 발행 연월은 2025-05로 보존하며 법령 시행일로 대체하지 않습니다. `fitness_closed`는 기본 민사 회수 안내를 보강했으나 최근 개정 원문 대조·개별 계약·채무자 재산·도산 검토가 남아 부분 확보 상태를 유지합니다.", "",
           '법제처 공식 HTML 안내 4건(소액사건·소비자 지급명령·지급명령 신청 절차·강제집행)을 민사 회수 참고 자료로 추가했습니다. 원문 작성 기준일 2026-09-15를 보존하고, 소액사건 3,000만원 이하 및 분할 금지·송달/주소보정·2주 이의·소송 전환·집행문 예외·재산명시/조회 조건·채권집행을 소제목 단위로 정제했습니다. 폐업 안내의 위험/상담 청크에 18개 관련 청크를 연결합니다. 지급명령 신청 안내의 민사소송법 2026-10-02 개정 검토·향후 업데이트 공지를 임베딩 경고로 보존합니다. 강제집행 안내의 대여금 집행권원 종류/가압류 예시는 기본 검색에서 제외하고, 집행문 원칙과 예외는 같은 청크에 보존합니다. 법원 사이트는 접근 차단으로 수집하지 않고 법제처 공식 안내를 사용했습니다. 이는 기본 절차 근거 보강이며 환급 보장이나 현행 개정법 적용 확정이 아닙니다.', "",
           "할부거래법(시행 2026-09-08)·시행령(시행 2026-07-01) 원문을 추가 수집·정제·청크에 반영했습니다. 법 제2·3·4·16조와 시행령 제2·4·11조를 폐업 안내의 `required_context_chunk_ids`로 연결합니다. 안내의 ‘3개월’ 원문은 유지하고 법령의 ‘2개월 이상에 걸쳐 3회 이상 지급’ 정의·신용카드 20만원 조건·항변 사유·신용제공자 통지·미지급 잔여금·서면 효력 및 적용제외를 함께 확인하도록 경고에 명시합니다. 수집 시점 현행본이며 2024/2025 안내 당시 구법·개별 계약일 적용을 확정하지 않습니다. Vector DB/답변 단계에서 연결 근거를 함께 로딩하는 기능은 별도 구현 대상입니다.", "",
           "공식 PDF 별표의 `pdf_tables`에는 페이지·셀 좌표·행/열 병합 범위·공통 조건 연결과 정제 본문 위치를 보존합니다. `source_comparison`은 대조 범위·시행일이며 `review_record_file/sha256`은 검토 기록입니다. `body_source_url`은 PDF 본문 출처입니다. `source_xml_*`은 XML 본문 지문·위치를 별도로 보존하고, 청크의 본문 위치는 XML이 아닌 정제 PDF 표 텍스트 기준입니다. 원본 PDF는 기존 참고 첨부로 남기고 별도 중복 청크를 만들지 않습니다.", "", "필라테스 폐강·양도 조정사례 2건의 계약·해지 시점을 공식 연혁 목록과 대조했습니다. 비교용 법령 원문 8개 및 연혁 HTML 4개는 data/raw/law_history에 참고용으로 보존하고 기본 법령 검색에는 편입하지 않습니다. 비교 26항목 중 25항목은 공백을 제외하면 동일하며, 2022년 시행본과 수집본의 제2조 전체 차이(2023년 개정)와 제10호 계속거래 정의의 동일성을 구분했습니다. 약관법 제9조 당시 본문은 reasoning/decision/cited_law 청크의 embedding_text 참고 문맥에 포함하고 출처·시행일을 표시합니다. 법 제52조·시행령 제13조 등 수집본 연결은 required_context_chunk_ids에 보존합니다. 정확한 조정일은 미상(null), 계약 월은 월 정밀도로 유지합니다. 조정의 위약금 면제·128,700원/386,100원·연15%는 사건별 조건이며 일반 계약에 자동 적용하지 않습니다. 대조 기록은 pilates_law_version_comparison_pmh.json이고 인계 대상입니다. 전체 법령의 동일성·최신 개정·개별 계약 적용을 확정한 것은 아닙니다.", "", "## 폴더와 원본 보존 정책", "",
           "| 위치 | 역할 |", "|---|---|",
           f"| `data/raw/laws` | 법령 {docs['laws']}종의 수집 API 응답 JSON과 수집 기록 |",
           f"| `data/raw/law_appendices` | 별표 PDF·원본 메타데이터 {report['details']['law_appendix_provenance']['pdf_metadata_pairs']}쌍. 정제 법령의 appendices와 연결 |",
           f"| `data/raw/precedents` | 원본 판례 {report['details']['raw_source_coverage_and_hashes']['precedent_originals']}건과 수집 기록. 정제 {docs['precedent_files']}개 / 검색 후보 {eligibility['precedent_ready']}건 |",
           "| `data/raw/guides` | 주택임대차·해고근로자·중고거래 피해자·인터넷 쇼핑·택배 PDF 5개 |",
           "| `data/raw/law_history` | 필라테스 사례 시점 비교용 원문 JSON 8개·연혁 HTML 4개. 기본 검색 제외 |",
           "| `data/raw/consumer_sources` | 공정위 약관 HWP/HWPX·행정규칙 XML·KCA 및 고용노동부 HTML·참고 PDF·이전 스냅샷·수집 기록 |",
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
           "| `embedding_text`, `text` | 각각 임베딩 입력과 보존 본문. 검색 별칭을 원문으로 취급하지 않음 |",
           "| `required_context_unit_ids` | 함께 확인할 원문 단위의 ID 목록. 청크 ID가 아니며 대상 청크의 `unit_id`로 연결 |",
           "| `search_passages`, `search_representation_version` | 로컬 BM25 점수용 구간·표현과 그 버전. 모든 자료에 존재하는 공통 필수 필드는 아님 |",
           "| `warnings`, `source_anomaly`, `calculation_restricted_clause` | 원문·추출 주의사항 및 계산 제한. 삭제하지 않음 |", "",
           "`required_context_unit_ids`가 있으면 각 ID와 `unit_id`가 일치하는 청크들을 찾아 본문과 함께 확인합니다. 한 원문 단위가 여러 청크로 나뉘면 모두 연결하고 문서·개정본을 확인합니다. 연결 대상이 없으면 누락으로 기록하며 본칙만으로 결론을 확정하지 않습니다. 현재 최저임금법 제6조의 연결은 2018년 부칙을 함께 확인하기 위한 것입니다. 이 필드는 서비스가 자동으로 부칙을 조회한다는 의미가 아닙니다.","",
           "`search_passages`의 `start`·`end`는 부모 `text`의 문자 위치이며 `text[start:end]`가 해당 구간 원문입니다. `search_text`가 있으면 별칭 등을 포함한 점수용 표현으로 사용합니다. 검색 결과의 `matched_span`만으로 답변·환급 계산을 하지 말고 부모 본문·공통 정의·출처·경고를 함께 전달합니다. 임베딩에는 `embedding_text`를 사용하고 모델 토큰 한도를 확인합니다.","",
           "분야 태그는 문서/기존 분류 수준의 후보이며 모든 조문이 해당 질문의 근거라는 뜻이 아닙니다. 민법·형법·근로기준법의 일반 태그를 임의로 9개 분야에 모두 매핑하지 않습니다. 세부 계약/사건 적용 및 시점 판단은 별도입니다.","",
           "## 제외·중복·본문 구조 정책", "",
           "- 판례는 관련성 검토표의 사유를 보존합니다. 중복 대표 판례만 검색하며 당사자 주장·증거·법원 판단의 역할을 유지합니다.",
           f"- 소비자 정제 문서 {docs['consumer_documents']}개 중 {len(eligibility['consumer_excluded'])}개는 참고용·이전 약관 등으로 검색 제외합니다. 별칭 {len(eligibility['consumer_duplicate_aliases'])}건은 대표 자료로 합칩니다. 최저임금 고시 PDF의 참고 문서는 주 고시 정제 문서에서 본문으로 사용하므로 누락과 구분합니다. 2007년 택배 약관은 역사 자료로 보존하고 기본 검색에서 제외합니다.",
           "- 법령 삭제 조문·검토 보류 자료를 임의로 적재하지 않습니다. 개정별 부칙의 공포일을 시행일로 추정하지 않습니다.",
           "- 안내서 표지·목차 등 제외 페이지와 이유는 청크 보고서에 있습니다. 이미지 요약은 수동 설명이며 원문 전문 전사가 아닙니다.",
           "- 표·약관 조문 등 원자 단위는 문자 목표 크기를 넘을 수 있습니다. 동일 본문도 출처·시점·문맥이 다를 수 있어 자동 삭제하지 않습니다.","",
           f"현재 공백 정규화 후 동일 본문 그룹은 {len(report['details'].get('chunk_metadata_and_duplicates',{}).get('identical_body_groups',[]))}개이며 ID 중복과 구분해 검증 JSON에 기록했습니다. 원문 URL 없는 청크는 {report['details'].get('chunk_metadata_and_duplicates',{}).get('source_url_missing',0)}개로, 원본 경로를 보존합니다.", "",
           "## 원문 이상과 적용 제한", "",
           "택배 2020년 약관 제22조 제2항 제3호 나목은 `운송장기재운임액의 200`으로 끝나며 단위가 없습니다. 원본 HWP와 변환 PDF에서 확인했습니다. **200%로 임의 보정하거나 이 문장으로 배상액을 계산하지 않습니다.** 안내서의 표기와 대조가 필요하며 경고를 검색·답변 단계까지 전달해야 합니다. 현재 서비스 실행 단계의 계산 제한 기능은 미구현입니다.",
           "최저임금 고시는 2026년 적용 기간을 명시합니다. 법 제6조 본칙의 상여금·복리후생비 비율은 연결된 2018년 부칙의 연도별 특례와 함께 확인해야 하며, 시행규칙·직종 고시 없이 급여 산입·수습 감액을 확정하지 않습니다. 주택임대차 시행령의 현행 지역별 기준을 과거 담보권 시점에 일괄 적용하지 않습니다.",
           "최저임금법 시행규칙 제2조에는 원문 이미지 태그와 `?勵加給` 대체문자가 있습니다. 원문을 임의 보정하지 않고 해당 문자·이미지 대조 필요 사항을 분야 검토에 기록했습니다. 삭제된 별표1·2는 산입범위의 현행 표로 사용하지 않습니다. 시행규칙 제2·3조, 별표3과 단순노무직종 지정 고시를 본법·부칙·연간 고시와 함께 확인하며 실제 업무의 직업분류를 자동 확정하지 않습니다. 방문판매법 시행령·시행규칙은 현재 수집본으로, 과거 조정사례에 동일 개정본이 적용됐다고 간주하지 않습니다.",
           "조정사례의 금액·이율·사실관계를 다른 사건에 그대로 적용하지 않습니다. 수집본을 현행 적용 가능 자료로 자동 간주하지 않습니다. PDF 원문 URL이 확인되지 않은 경우 기관 홈페이지를 원문 URL로 대체하지 않습니다.","",
           "보증금 반환 보강: 소송촉진법·법정이율 규정·민사집행법·규칙 4종과 공식 주택 안내 3건(13개 소제목 청크), 판례 147076의 관련 구간을 추가했습니다. 법정이율 규정은 조 번호 없는 전문으로 API 조문번호 0을 보존하며 제1조로 만들지 않습니다. 반환 지연은 인도·이행 제공·약정·송달과 소송촉진법 제3조 예외를 함께 확인합니다.",
           "판례 147076은 2007년 서울중앙지법의 학원 건물 임대차 사건(원문 항소 표시)입니다. 주택 직접 판례·대법원 확정 판단으로 간주하지 않습니다. 당시 연20%와 공사비 50%·5일 인정은 현재 일반 이율·공제율·기간으로 적용하지 않습니다. 사실관계·특약·견적·공사기간·잔액 판단을 발췌하고 경고를 임베딩 본문까지 보존했습니다.",
           "배당 안내 원문 인용 불일치: 민사집행규칙 제91조는 수집 현행본에서 강제관리 수익 처리이고, 민사집행법 제148조 제3호는 선등기 가압류채권자입니다. `easylaw_housing_distribution.json`의 `citation_discrepancies`와 청크 경고에 기록했습니다. 원문 인용을 임의 수정하지 않고 종기·서면의 법 제84·88조/규칙 제48조 및 배당 범위 법 제148조를 연결했습니다. 임차권등기 예외의 개별 적용은 추가 확인합니다. 안내의 배당 순위 표는 현행 조세법 대조 전 검색 제외하며 이유와 원본을 보존합니다.","",
           "## 검색·분야 검토 결과와 보강 목록", "",
           f"- 판례 10문항: BM25 Hit@3={baseline['hit_at_3']:.0%}, MRR@3={baseline['mrr_at_3']:.3f}.",
           f"- 소비자 12문항: 동일 데이터의 기존 검색 Hit@5={consumer_eval['legacy_after']['expected_source_hit_at_5']:.1%} → 개선 검색 Hit@5={consumer_eval['after']['expected_source_hit_at_5']:.1%}, 개선 Hit@3={consumer_eval['after']['expected_source_hit_at_3']:.1%}.",
           "- 개선 BM25는 본문 구간·HWPX 셀 문자로 점수를 계산하고 생활 표현을 검색 용어로 확장합니다. 본문 토큰은 1배·출처 헤더는 0.5배로 점수화합니다. 환급 행은 조건·수식과 환불 계산/환급금 계산 별칭으로 검색하며 공통 정의는 전체 원문과 표 메타데이터에 보존합니다. 당사자 주장 점수는 0.6배, 문서 버전당 최대 2개 청크로 제한하며 원문 전체·역할·경고를 반환합니다. 설정은 로컬 평가용이며 서비스 검색·답변 제한은 별도 구현이 필요합니다.",
           f"- 45문항 동일 자료·선정 근거 비교: 출처 {coverage['search_comparison']['legacy_source_hits']} → {coverage['search_comparison']['improved_source_hits']}건, 선정 청크 {coverage['search_comparison']['legacy_chunk_hits']} → {coverage['search_comparison']['improved_chunk_hits']}건. 청크 적중 하락 질문: {', '.join(coverage['search_comparison']['chunk_regressed']) or '없음'}.",
           f"- 출처 적중 하락 질문: {', '.join(coverage['search_comparison']['source_regressed']) or '없음'}. 선정 근거가 Top 5 밖에 있는 질문은 분야 검토 보고서에 기록합니다. 이번 설정을 전체 서비스의 최종 검색 정책으로 확정한 것은 아닙니다.",
           f"- 원문 보강 직전 근거 ID 고정 누적 비교(이후 검색 설정 변경 포함): {coverage['source_supplement_comparison']['before']} → {coverage['source_supplement_comparison']['after']}건. 하락 질문: {', '.join(coverage['source_supplement_comparison']['regressed']) or '없음'}.",
           f"- 검색 표현·헤더 가중치 수정 직전 선정 근거 고정 비교: {coverage['search_refinement_comparison']['before']} → {coverage['search_refinement_comparison']['after']}건. 하락: {', '.join(coverage['search_refinement_comparison']['regressed']) or '없음'}.",
           f"- 소비자 12문항의 선정 청크 진단: {consumer_eval['search_refinement_comparison']['chunk_before']} → {consumer_eval['search_refinement_comparison']['chunk_after']}건. 출처 적중: {consumer_eval['search_refinement_comparison']['source_before']} → {consumer_eval['search_refinement_comparison']['source_after']}건.",
           f"- 위임 규정·직종 고시 4종 보강 직전 선정 근거 ID 고정 비교: {coverage['delegated_source_comparison']['before']} → {coverage['delegated_source_comparison']['after']}건. 하락: {', '.join(coverage['delegated_source_comparison']['regressed']) or '없음'}.",
           "- 분야 검토: 선정 질문 45개. 자료 내용 검토와 검색 진단을 분리합니다. 키워드 적중만으로 충분하다고 판정하지 않습니다.", "",
           "| 분야 | 기본 근거 | 보완 | 추가 검토 | 적용 보류 |", "|---|---:|---:|---:|---:|"]
    for item in coverage["summary"]:
        c=item["statuses"]
        lines.append(f"| {item['domain']} | {c.get('supported',0)} | {c.get('partial',0)} | {c.get('needs_review',0)} | {c.get('restricted',0)} |")
    lines += ["", "분야별 세부 질문·청크 ID·부족 자료는 [분야 검토 보고서](../data/processed/retrieval_evaluation/domain_coverage_report_pmh.md)를 참조합니다. 원문을 이미 수집했는지 먼저 확인한 후 필요한 발췌·정제 또는 추가 수집을 결정합니다.", "",
              "우선 원문 보강 7종은 완료했습니다: 방문판매법, 전자상거래법 시행령, 최저임금법·시행령, 주택임대차보호법 시행령, 계속거래 환급 산정기준, 2026년 적용 최저임금 고시. 고시 API 본문이 빈 경우 공식 첨부 PDF를 본문으로 사용했고 원본·페이지·표·해시·적용 기간을 연결했습니다. 이후 보류 작업을 재개하여 방문판매법 시행령·시행규칙, 최저임금법 시행규칙·단순노무직종 지정 고시 4종을 추가 수집·정제·청크에 반영했습니다. 필라테스 두 사례의 계약·해지 시점 대조는 완료했고, 남은 자료·대조는 다른 사례·안내 당시 구법, 실제 급여 항목·업무의 직업분류, 주택임대차 시행령 과거 버전, 최근 민사소송법 개정·민사집행 관련 현행 원문 대조 등입니다. 구매일 증빙 안내 보강과 별표Ⅲ·Ⅳ 수집본 대조는 완료했습니다. 임금체불 신고 절차·준비자료는 공식 HTML 2건, 해외직구 유형별 안내는 공식 HTML 3건, 헬스장·필라테스 폐업·피해구제 절차는 공식 HTML 3건으로 보강했습니다.","",
              "## 처음 실행할 때의 환경 설정", "",
              "Python 3.12 이상과 uv가 설치된 환경에서 프로젝트 루트를 작업 디렉터리로 사용합니다. `pyproject.toml`·`uv.lock`과 현재 원본·정제·청크·검토 기록을 함께 인계받아야 합니다.","",
              "```bash", "uv sync --locked", "```", "",
              "국가법령정보센터 API로 새 법령·판례·행정규칙을 수집할 때는 프로젝트 루트의 로컬 `.env`에 아래 항목을 설정하거나 같은 이름의 환경변수를 설정합니다. 예시의 자리표시자는 실제 발급받은 OC 값으로 바꿉니다. 실제 인증값은 커밋하거나 인계 문서에 기록하지 않습니다.","",
              "```dotenv", "LAW_API_OC=<발급받은_OC_값>", "```", "",
              "이미 인계받은 파일로 아래 검증을 실행할 때는 `LAW_API_OC`가 필요하지 않으며 수집 API를 호출하지 않습니다. uv 의존성을 처음 설치할 때는 네트워크가 필요할 수 있습니다.","",
              "```bash", 'uv run python -m unittest discover -s src/scripts/tests -t . -p "test*_pmh.py"', "```", "",
              "## 재현 실행 순서", "", "다음 명령은 정제·청크·평가 보고서를 다시 생성하므로, 처음 인계받을 때는 위 테스트로 현재 스냅샷을 먼저 확인합니다. 데이터 갱신이 필요한 경우 아래 순서로 실행합니다.", "", "```bash",
              "# 원본 및 관련성/시각 검토 기록을 준비한 후 정제",
              "uv run python src/scripts/preprocess_laws.py",
              "# 소비자 안내가 연결할 최신 법령 청크를 먼저 생성",
              "uv run python src/scripts/chunk_laws.py",
              "uv run python src/scripts/preprocess_precedents.py",
              "uv run python src/scripts/preprocess_guides.py",
              "# 필라테스 시점 비교: 기존 원본으로 오프라인 기록 재생성",
              "uv run python -m src.scripts.review_pilates_law_versions",
              "uv run python src/scripts/preprocess_consumer_sources.py",
              "# 나머지 청크 생성",
              "uv run python src/scripts/chunk_guides.py",
              "uv run python src/scripts/chunk_consumer_sources.py",
              "uv run python -m src.scripts.tests.test_precedent_retrieval_pmh",
              "# 검색/분야 평가",
              "uv run python -m src.scripts.tests.test_consumer_coverage_pmh",
              "uv run python -m src.scripts.tests.test_domain_coverage_pmh",
              "# 최종 감사·전체 테스트·인계서 갱신",
              "uv run python -m src.scripts.tests.test_data_handoff_pmh",
              "# 보고서 생성 없이 전체 테스트만 실행",
              'uv run python -m unittest discover -s src/scripts/tests -t . -p "test*_pmh.py"', "```", "",
              "분야 검토 코드는 선정 근거의 본문 지문이 바뀌면 추가 검토로 표시합니다. 질문·근거 메모와 지문을 내용 확인 없이 갱신하지 않습니다. 원본 추가 시 수집 목록·검토표·보고서에 반영한 뒤 위 순서로 재생성합니다.","",
              "## 자료 보강 재개 현황", "",
              "사용자의 재개 요청으로 부족한 근거 정리 → 필요한 원문 수집 → 정제·청크 반영 → 누락·검색 재검증 → 인계 문서 갱신을 진행했습니다. 첫 보강으로 위임 규정·직종 고시 4종을 반영했으며 나머지 분야의 자료 보강까지 완료한 상태는 아닙니다.","",
              f"현재 {coverage['question_count']}문항 내용 검토는 기본 근거 확보 {review_counts.get('supported',0)}문항, 보완 필요 {review_counts.get('partial',0)}문항, 추가 검토 {review_counts.get('needs_review',0)}문항, 원문 적용 보류 {review_counts.get('restricted',0)}문항입니다. 원문 적용 보류는 내용 검토 상태이며, 작업 재개 여부와 구분합니다. 택배 약관의 단위 누락 경고는 위 원문 이상 항목을 확인합니다.","",
              "후속 보강은 분야 검토 보고서의 질문 ID와 선정 근거를 기준으로 원문 부족·정제 누락·검색 미적중을 먼저 구분합니다. 이미 수집한 근거는 재활용하고, 부족한 자료만 수집한 뒤 정제·청크·검색 평가·최종 감사 순서로 갱신합니다. 코드 수정은 파일 목록을 먼저 제시하고 사용자 확인 후 진행하며 새 테스트 파일 이름은 `_pmh.py`로 끝나게 합니다.","",
              "## 다음 담당자의 작업", "",
              "1. 이 스냅샷의 검증 결과와 부족 자료 범위를 확인합니다.",
              "2. 두 저장 형식을 지원하는 서비스 로더와 임베딩 실행·실패 처리·토큰 제한 검증을 구현합니다.",
              "3. Qdrant 적재·갱신·검색과 필터를 구현합니다. 출처 적중과 정확한 근거 청크 적중을 구분해 평가합니다.",
              "4. LLM 답변의 근거·출처 표시, 근거 부족 시 대응, 원문 경고 전달을 구현하고 MCP를 연결합니다.",
              "5. 미사용 질문으로 실제 RAG 답변·계산·시점·출처 정확성을 평가합니다. 단위 테스트 통과를 답변 정확성으로 해석하지 않습니다.","",
              "## 검증 산출물", "",
              "- [최종 검증 JSON](../data/processed/retrieval_evaluation/data_validation_report_pmh.json): 항목별 결과·파일 크기/해시·테스트 결과·제외 기록.",
              "- [검증 실행 파일](../src/scripts/tests/test_data_handoff_pmh.py): 직접 실행 시 보고서와 이 문서를 작성하고 전체 테스트를 실행합니다.",
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
    suite=unittest.defaultTestLoader.discover(str(ROOT/"src/scripts/tests"),pattern="test*_pmh.py",top_level_dir=str(ROOT))
    result=unittest.TextTestRunner(verbosity=1).run(suite)
    report["test_suite"]={"tests_run":result.testsRun,"successful":result.wasSuccessful(),
                          "failures":[str(t) for t,_ in result.failures],"errors":[str(t) for t,_ in result.errors],
                          "skipped":[str(t) for t,_ in result.skipped]}
    report["handoff_ready_for_initial_pipeline"]=report["structural_passed"] and result.wasSuccessful()
    write_outputs(report)
    print(json.dumps({"structural_passed":report["structural_passed"],"tests":report["test_suite"]},ensure_ascii=False))
    return 0 if report["handoff_ready_for_initial_pipeline"] else 1


class HandoffTests(unittest.TestCase):
    def test_simple_labor_notice_reprocesses_as_binding_notice_not_dispute_guideline(self):
        from src.scripts.preprocess_consumer_sources import preprocess
        collected=read(PROCESSED_CONSUMER_SOURCES_DATA_PATH / 'consumer_collection_report.json')
        source=next(s for s in collected['saved'] if s['source_id']=='minimum_wage_simple_labor')
        doc=preprocess(source)
        self.assertTrue(doc['embedding_candidate'])
        self.assertEqual(doc['effective_date'],'20180320')
        self.assertEqual(doc['administrative_metadata']['행정규칙일련번호'],'2100000118064')
        self.assertIn('대분류 9',doc['units'][0]['text'])
        self.assertEqual(doc['warnings'],['공식 고시. 적용 범위·예외·유효 기간과 상위 법령을 함께 확인. 계산 자동화는 별도 검증.'])

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
            checked_file(CHUNK_LAWS_DATA_PATH / "chunks.jsonl","wrong_hash")


if __name__=="__main__":
    raise SystemExit(evaluate())
