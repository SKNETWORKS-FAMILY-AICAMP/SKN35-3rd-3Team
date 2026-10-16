"""공통 메타데이터, 보존성, 실제 저장 데이터 및 생성 코드의 일관성 검증."""
import copy
import json
import unittest

from src.const.config import (
    CHUNK_CONSUMER_SOURCES_DATA_PATH,
    CHUNK_GUIDES_DATA_PATH,
    CHUNK_LAWS_DATA_PATH,
    CHUNK_PRECEDENTS_DATA_PATH,
    PROCESSED_CONSUMER_SOURCES_DATA_PATH,
    PROCESSED_DATA_PATH,
    PROCESSED_GUIDES_DATA_PATH,
    PROCESSED_LAWS_DATA_PATH,
    PROCESSED_PRECEDENTS_DATA_PATH,
    PROJECT_PATH,
    RAW_LAWS_DATA_PATH,
)

from src.scripts.chunk_metadata import normalize_chunk, validate_chunk, normalize_date
from src.scripts import chunk_laws, chunk_guides, chunk_consumer_sources
from src.scripts.tests.test_precedent_retrieval_pmh import make_chunks as precedent_chunks

ROOT = PROJECT_PATH


def corpus():
    result = []
    for directory in (CHUNK_LAWS_DATA_PATH, CHUNK_PRECEDENTS_DATA_PATH, CHUNK_CONSUMER_SOURCES_DATA_PATH):
        path = directory / "chunks.jsonl"
        result.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    root = CHUNK_GUIDES_DATA_PATH
    manifest = json.loads((root / "guide_chunking_report.json").read_text(encoding="utf-8"))
    for entry in manifest["saved"]:
        result.extend(json.loads((root / entry["file"]).read_text(encoding="utf-8"))["chunks"])
    return result


class MetadataTests(unittest.TestCase):
    def test_consumer_framework_laws_are_general_consumer_sources(self):
        records=corpus()
        for name in ['소비자기본법','소비자기본법 시행령']:
            chunks=[c for c in records if c.get('law_name')==name]
            self.assertTrue(chunks)
            self.assertEqual(len({c['document_id'] for c in chunks}),1)
            for c in chunks:
                self.assertEqual(c['domains'],['consumer_general'])
                self.assertEqual(c['dates']['effective_date'],'2026-09-11')
                self.assertEqual(normalize_chunk(c),c)

    def test_unknown_purchase_date_rule_and_its_exception_stay_in_one_chunk(self):
        import re
        report=json.loads((PROCESSED_LAWS_DATA_PATH / 'law_preprocessing_report.json').read_text(encoding='utf-8'))
        entry=next(e for e in report['saved'] if e['law_name']=='소비자기본법 시행령')
        d=json.loads((PROCESSED_LAWS_DATA_PATH/entry['file']).read_text(encoding='utf-8'))
        u=next(u for u in d['appendices'] if u['title'].startswith('일반적 소비자분쟁해결기준'))
        generated=chunk_laws.make_aux_chunks(d,u,size=300,overlap=30)
        self.assertEqual(len(generated),1)
        c=generated[0]
        self.assertTrue(c['atomic_unit'])
        self.assertTrue(c['oversize_atomic_unit'])
        self.assertEqual(c['text'],u['body_text'])
        self.assertEqual((c['start'],c['end']),(0,len(u['body_text'])))
        compact=re.sub(r'\s+','',c['text'])
        for phrase in ['영수증을받지아니하거나분실한경우','제조일이나수입통관일부터3월이',
                       '사업자가그판매일자를입증하여야한다']:
            self.assertIn(phrase,compact)
        self.assertEqual(c['embedding_text'].count(c['text']),1)
        saved=[x for x in corpus() if x.get('unit_id')==u['unit_id']]
        self.assertEqual(len(saved),1)
        self.assertEqual(saved[0]['text'],u['body_text'])

    def test_housing_procedure_laws_are_mapped_as_general_civil_sources(self):
        from src.scripts.chunk_metadata import LAW_DOMAINS
        for name in ['소송촉진 등에 관한 특례법','소송촉진 등에 관한 특례법 제3조제1항 본문의 법정이율에 관한 규정','민사집행법','민사집행규칙']:
            self.assertEqual(LAW_DOMAINS[name],['civil_general'])
        chunks=[c for c in corpus() if c.get('law_name')=='소송촉진 등에 관한 특례법 제3조제1항 본문의 법정이율에 관한 규정' and c.get('article_label')=='전문']
        self.assertEqual(len(chunks),1)
        self.assertIn('연 100분의 12',chunks[0]['text'])
        self.assertNotIn('제1조',chunks[0]['context_header'])

    def precedent(self):
        return {"chunk_id": "123_0000", "precedent_id": "123", "case_name": "사건명",
                "domain": "해고·임금", "source_url": "https://www.law.go.kr",
                "judgment_date": "20240123", "text": "본문", "embedding_text": "경고\n본문",
                "warnings": ["원문 이상"], "calculation_restricted_clause": "단위 미확인"}

    def test_preserves_original_metadata_text_and_cautions_without_mutation(self):
        original = self.precedent()
        before = copy.deepcopy(original)
        normalized = normalize_chunk(original, {"source_file": "data/raw/precedents/123.json"})
        self.assertEqual(original, before)
        for key, value in original.items():
            self.assertEqual(normalized[key], value)
        self.assertEqual(normalized["document_type"], "precedent")
        self.assertEqual(normalized["domains"], ["labor_dismissal", "labor_wage"])
        self.assertEqual(normalized["dates"]["judgment_date"], "2024-01-23")
        self.assertIsNone(normalized["dates"]["effective_date"])
        self.assertEqual(normalized["source_path"], "data/raw/precedents/123.json")
        self.assertEqual(normalize_chunk(normalized), normalized)

    def test_html_guide_domain_mapping_and_idempotence_do_not_change_pdf_classification(self):
        value={'chunk_id':'html_0','document_type':'official_guide','guide_id':'html',
               'source_id':'html','source_format':'html','title':'공식 안내','domain':'임금체불',
               'text':'원문','embedding_text':'원문','information_as_of':'2026-05-14'}
        normalized=normalize_chunk(value)
        self.assertEqual(normalized['domains'],['labor_wage'])
        self.assertEqual(normalize_chunk(normalized),normalized)
        self.assertEqual(normalized['dates']['information_as_of'],'2026-05-14')
        self.assertIsNone(normalized['dates']['effective_date'])
        pdf={**value,'source_format':'pdf','source_file':'[pdf]주택임대차.pdf'}
        self.assertEqual(normalize_chunk(pdf)['domains'],['housing_lease'])

    def test_unknown_domain_and_missing_source_are_not_invented(self):
        value = self.precedent()
        value["domain"] = "미분류 분야"
        normalized = normalize_chunk(value)
        self.assertEqual(normalized["domains"], [])
        self.assertEqual(normalized["domain"], "미분류 분야")
        self.assertEqual(normalized["domain_classification"], "unmapped")
        self.assertIsNone(normalized["source_path"])

    def test_broad_laws_are_not_assigned_every_dispute_domain(self):
        value = {"chunk_id": "law1", "document_type": "law", "version_id": "v1",
                 "law_name": "민법", "source_file": "v1.json", "text": "목적", "embedding_text": "목적"}
        normalized = normalize_chunk(value)
        self.assertEqual(normalized["domains"], ["civil_general"])
        self.assertEqual(normalized["source_path"], (RAW_LAWS_DATA_PATH / "v1.json").relative_to(PROJECT_PATH).as_posix())
        self.assertEqual(normalized["document_id"], "law:v1")

    def test_new_delegated_laws_keep_body_and_version_with_document_level_domains(self):
        for name, expected in [
                ("방문판매 등에 관한 법률 시행령", ["fitness_refund", "consumer_general"]),
                ("방문판매 등에 관한 법률 시행규칙", ["fitness_refund", "consumer_general"]),
                ("최저임금법 시행규칙", ["labor_wage"])]:
            with self.subTest(law_name=name):
                value={"chunk_id":"law1", "document_type":"law", "version_id":"v1",
                       "law_name":name, "source_file":"v1.json", "text":"원문 본문",
                       "embedding_text":"출처\n원문 본문", "effective_date":"20260102",
                       "warnings":["수집본의 시점 확인 필요"]}
                before=copy.deepcopy(value)
                normalized=normalize_chunk(value)
                self.assertEqual(value,before)
                for field in value:
                    self.assertEqual(normalized[field],value[field])
                self.assertEqual(normalized["domains"],expected)
                self.assertEqual(normalized["domain_classification"],"law_document")
                self.assertEqual(normalize_chunk(normalized),normalized)

    def test_installment_laws_have_consumer_domains_without_changing_original_text(self):
        for name in ['할부거래에 관한 법률','할부거래에 관한 법률 시행령']:
            value={'chunk_id':'v1_0','document_type':'law','version_id':'v1','law_name':name,
                   'source_file':'v1.json','text':'법령 원문','embedding_text':'법령 원문','effective_date':'20260908'}
            before=copy.deepcopy(value)
            result=normalize_chunk(value)
            self.assertEqual(value,before)
            self.assertEqual(result['domains'],['fitness_refund','consumer_general'])
            self.assertEqual(result['text'],before['text'])
            self.assertEqual(result['dates']['effective_date'],'2026-09-08')
            self.assertEqual(normalize_chunk(result),result)

    def test_invalid_dates_and_unsafe_paths_fail(self):
        for value in ("20240230", "2024", "어제", 20240123):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_date(value)
        for value in ("../secret.json", "C:/secret.json", "/secret.json"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_chunk(self.precedent(), {"source_file": value})

    def test_empty_text_and_duplicate_domains_fail_validation(self):
        normalized = normalize_chunk(self.precedent())
        normalized["text"] = " "
        with self.assertRaises(ValueError):
            validate_chunk(normalized)
        normalized = normalize_chunk(self.precedent())
        normalized["domains"] *= 2
        with self.assertRaises(ValueError):
            validate_chunk(normalized)

    def test_saved_corpus_has_valid_metadata_unique_ids_and_existing_sources(self):
        chunks = corpus()
        self.assertTrue(chunks)
        self.assertEqual(len(chunks), len({c["chunk_id"] for c in chunks}))
        for chunk in chunks:
            with self.subTest(chunk_id=chunk["chunk_id"]):
                validate_chunk(chunk)
                self.assertEqual(normalize_chunk(chunk), chunk)
                self.assertTrue(chunk["domains"], "현재 확보한 자료의 분류 매핑 누락")
                self.assertIsNotNone(chunk["source_path"])
                self.assertTrue((ROOT / chunk["source_path"]).is_file())

    def test_saved_chunks_match_all_four_generators(self):
        stored = {c["chunk_id"]: c for c in corpus()}
        generated = []
        manifest = json.loads((PROCESSED_LAWS_DATA_PATH / 'law_preprocessing_report.json').read_text(encoding="utf-8"))
        for entry in manifest["saved"]:
            doc = json.loads((PROCESSED_LAWS_DATA_PATH / entry["file"]).read_text(encoding="utf-8"))
            generated.extend(c for a in doc["articles"] for c in chunk_laws.make_chunks(doc, a))
            generated.extend(c for u in doc["supplements"] + doc["appendices"] for c in chunk_laws.make_aux_chunks(doc, u))
        manifest = json.loads((PROCESSED_GUIDES_DATA_PATH / 'guide_preprocessing_report.json').read_text(encoding="utf-8"))
        for entry in manifest["saved"]:
            doc = json.loads((PROCESSED_GUIDES_DATA_PATH / entry["file"]).read_text(encoding="utf-8"))
            generated.extend(chunk_guides.make_chunks(doc))
        manifest = json.loads((PROCESSED_CONSUMER_SOURCES_DATA_PATH / 'consumer_preprocessing_report.json').read_text(encoding="utf-8"))
        for entry in manifest["saved"]:
            doc = json.loads((PROCESSED_CONSUMER_SOURCES_DATA_PATH / entry["file"]).read_text(encoding="utf-8"))
            generated.extend(chunk_consumer_sources.make_chunks(doc))
        manifest = json.loads((PROCESSED_DATA_PATH / 'precedent_preprocessing_report.json').read_text(encoding="utf-8"))
        for identifier in manifest["saved"]:
            doc = json.loads((PROCESSED_PRECEDENTS_DATA_PATH / f"{identifier}.json").read_text(encoding="utf-8"))
            if doc["embedding_candidate"] and not doc["needs_manual_review"]:
                generated.extend(precedent_chunks(doc))
        self.assertEqual(len(generated), len(stored))
        for chunk in generated:
            with self.subTest(chunk_id=chunk["chunk_id"]):
                self.assertEqual(chunk, stored[chunk["chunk_id"]])
