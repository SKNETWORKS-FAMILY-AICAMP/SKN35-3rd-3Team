"""필라테스 시점 대조의 버전 선택·근거·실패 차단 검증(네트워크 없음)."""
import copy
import json
import unittest
from unittest.mock import patch

from src.const.config import (
    CHUNK_LAWS_DATA_PATH,
    PROCESSED_CONSUMER_SOURCES_DATA_PATH,
)

from src.scripts.review_pilates_law_versions import (
    ROOT, RAW, REPORT, LAW, DECREE, RULE, TERMS, HISTORY_IDS,
    article, build_report, checked, digest, history_rows, read, select_at, validate_body, verified_case, sanitize_response,
)


class PilatesLawVersionTests(unittest.TestCase):
    def test_authentication_is_removed_from_urls_and_hidden_inputs_before_storage(self):
        secret='fixture_auth_token'
        sample='<input type="hidden" id="OC" value="'+secret+'"/><a href="?x=1&amp;OC='+secret+'&amp;y=2">법령</a>'
        result=sanitize_response(sample,secret)
        self.assertNotIn(secret,result)
        self.assertIn('value="REDACTED"',result)
        self.assertIn('&amp;y=2',result)
        self.assertIn('법령',result)
        with self.assertRaises(ValueError):sanitize_response('unrecognized '+secret,secret)
        history=(RAW/f"history_{HISTORY_IDS[LAW]}_through_2023.html").read_text(encoding='utf-8')
        with self.assertRaises(ValueError):history_rows(history.replace('value="REDACTED"','value="fixture_auth_token"'),LAW)

    @classmethod
    def setUpClass(cls):
        cls.report=read(REPORT)

    def test_saved_comparison_is_reproducible_offline(self):
        self.assertEqual(self.report,build_report())
        self.assertEqual(len(self.report['comparisons']),26)
        self.assertEqual(sum(r['status']=='same_except_whitespace' for r in self.report['comparisons']),25)
        for path,sha in self.report['inputs'].items():
            self.assertEqual(digest(checked(path)),sha)

    def test_contract_month_is_not_an_invented_exact_day_or_decision_date(self):
        cases={c['source_id']:c for c in self.report['cases']}
        group=cases['kca_case_1003739114']
        self.assertIsNone(group['decision_date'])
        self.assertEqual(group['events'][0]['value'],'2023-04')
        self.assertEqual(group['events'][0]['precision'],'month')
        self.assertEqual(group['events'][0]['law_versions'][LAW],'249019')
        self.assertEqual(group['events'][1]['law_versions'][LAW],'252407')
        assigned=cases['kca_case_1003640779']
        self.assertEqual(assigned['events'][0]['law_versions'][LAW],'237455')
        self.assertEqual(assigned['events'][0]['law_versions'][DECREE],'246957')
        self.assertEqual(assigned['events'][1]['law_versions'][DECREE],'248655')
        self.assertEqual(assigned['events'][1]['law_versions'][RULE],'223955')
        for c in cases.values():
            self.assertIsNone(c['decision_date'])
            doc=read(PROCESSED_CONSUMER_SOURCES_DATA_PATH / f"{c['source_id']}.json")
            units={u['unit_id']:u['text'] for u in doc['units']}
            for e in c['events']:
                self.assertEqual(units[e['evidence_unit_id']][e['evidence_start']:e['evidence_end']],e['evidence_text'])

    def test_definition_change_is_separate_from_unchanged_continuing_transaction(self):
        rows=[r for r in self.report['comparisons'] if r['historical_master_id']=='237455' and r['article_number']=='2']
        self.assertEqual({r['subclause']:r['status'] for r in rows},
                         {None:'changed_non_target_definition','10':'same_except_whitespace'})
        for r in self.report['comparisons']:
            self.assertEqual(digest(r['historical_text'].encode()),r['historical_text_sha256'])
            self.assertEqual(digest(r['collected_text'].encode()),r['collected_text_sha256'])

    def test_version_boundary_and_ambiguous_history_fail_closed(self):
        rows=history_rows((RAW/f"history_{HISTORY_IDS[LAW]}_through_2023.html").read_text(encoding='utf-8'),LAW)
        self.assertEqual(select_at(rows,'2023-03-20')['master_id'],'237455')
        self.assertEqual(select_at(rows,'2023-03-21')['master_id'],'249019')
        self.assertEqual(select_at(rows,'2023-07-10')['master_id'],'249019')
        self.assertEqual(select_at(rows,'2023-07-11')['master_id'],'252407')
        with self.assertRaises(ValueError):select_at([], '2023-01-01')
        with self.assertRaises(ValueError):select_at([rows[0],{**rows[0],'master_id':'different'}],'2023-07-14')
        with self.assertRaises(ValueError):history_rows('<table></table>',LAW)

    def test_wrong_law_name_version_missing_article_and_changed_case_are_rejected(self):
        data=read(RAW/'225063_20211230.json')
        with self.assertRaises(ValueError):validate_body(data['response_xml'],DECREE,'20211230')
        with self.assertRaises(ValueError):validate_body(data['response_xml'],TERMS,'20230101')
        root=validate_body(data['response_xml'],TERMS,'20211230')
        with self.assertRaises(ValueError):article(root,'999')
        doc=read(PROCESSED_CONSUMER_SOURCES_DATA_PATH / 'kca_case_1003640779.json')
        changed=copy.deepcopy(doc);changed['units'][0]['text']+='changed'
        with self.assertRaisesRegex(ValueError,'본문 변경'):verified_case(changed)
        with patch('src.scripts.review_pilates_law_versions.build_report',return_value={}):
            with self.assertRaisesRegex(ValueError,'현재 입력 불일치'):verified_case(doc)
        with self.assertRaises(ValueError):checked('../outside.json')
        with self.assertRaises(ValueError):checked('data/raw/law_history/225063_20211230.json','bad')

    def test_historical_sources_are_references_not_default_law_chunks(self):
        chunks=[json.loads(s) for s in (CHUNK_LAWS_DATA_PATH / 'chunks.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        self.assertTrue(all('law_history' not in c['source_path'] for c in chunks))
        self.assertEqual(self.report['refund_notice']['notice_number'],'2019-9')
        self.assertEqual(self.report['refund_notice']['effective_date'],'20191119')
        for case in self.report['cases']:
            self.assertTrue(set(case['required_context_chunk_ids'])<={c['chunk_id'] for c in chunks})
            self.assertIn('제9조',case['supporting_law_context'])
            self.assertTrue(case['warnings'])


if __name__=='__main__':unittest.main()
