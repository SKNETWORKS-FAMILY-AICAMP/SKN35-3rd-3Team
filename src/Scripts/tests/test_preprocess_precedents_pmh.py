"""실행: uv run python -m unittest src.Scripts.tests.test_preprocess_precedents_pmh"""

import hashlib
import json
import unittest

from src.Scripts.preprocess_precedents import (
    ROOT, TEXT_FIELDS, clean_text, get_sections, partial_spans, preprocess, text_digest,
)


class PreprocessingTests(unittest.TestCase):
    def test_html_boundaries_and_legal_notation(self):
        value = '<p>「민법」 제390조&nbsp; 1,290만 원</p><p>2024다216187<br/>甲 &amp; 乙</p><script>삭제</script>'
        cleaned = clean_text(value)
        self.assertEqual(cleaned, '「민법」 제390조 1,290만 원\n\n2024다216187\n甲 & 乙')

    def test_source_sections_are_exact_slices(self):
        body = '서두\n【주    문】무죄\n【이    유】판단 내용'
        sections = get_sections(body)
        self.assertEqual([s['heading'] for s in sections], ['서두', '주문', '이유'])
        for section in sections:
            self.assertEqual(section['text'], body[section['start']:section['end']])

    def test_unknown_or_changed_partial_boundaries_fail_closed(self):
        with self.assertRaises(ValueError):
            partial_spans('unknown', '본문')
        with self.assertRaises(ValueError):
            partial_spans('605583', '제목이 달라진 새로운 본문')

    def test_all_reviewed_sources_and_provenance(self):
        report = json.loads((ROOT / 'data/processed/precedent_preprocessing_report.json').read_text(encoding='utf-8'))
        review = json.loads((ROOT / 'data/processed/precedent_relevance_review.json').read_text(encoding='utf-8'))
        selected = {r['precedent_id']: r for r in review['records'] if r['status'] in {'포함', '부분 사용'}}
        self.assertEqual(set(selected), set(report['saved']))
        self.assertFalse(report['failures'])
        for identifier, row in selected.items():
            with self.subTest(identifier=identifier):
                raw_path = ROOT / 'data/raw/precedents' / f'{identifier}.json'
                raw_bytes = raw_path.read_bytes()
                self.assertEqual(hashlib.sha256(raw_bytes).hexdigest(), report['raw_sha256'][identifier])
                raw = json.loads(raw_bytes.decode('utf-8'))
                regenerated = preprocess(row, raw)
                saved = json.loads((ROOT / 'data/processed/precedents' / f'{identifier}.json').read_text(encoding='utf-8'))
                self.assertEqual(saved['texts'], regenerated['texts'])
                self.assertEqual(saved['retrieval_text'], regenerated['retrieval_text'])
                self.assertTrue(saved['retrieval_text'].strip())
                for section in saved['sections'] + saved['selected_spans']:
                    self.assertEqual(section['text'], saved['texts']['full_text'][section['start']:section['end']])
                # 요지가 없던 원본에 요약이 생성되지 않는지 확인한다.
                for key, field in TEXT_FIELDS.items():
                    if not raw['detail'].get(field, '').strip():
                        self.assertEqual(saved['texts'][key], '')

    def test_merged_case_excerpts_do_not_leak_other_crimes(self):
        expected = {
            '605583': ('중고나라', ['『2023고단1022』', '『2024고단482』', '횡령금 112,565,000원', '징역 1년 8월']),
            '234477': ('임금', ['1. 2018고단1833 -', '다. 위계공무집행방해', '2. 위계공무집행방해죄가 성립하지 않는다는 주장', '징역 3년']),
            '227759': ('퇴직금', ['가. 제1 원심판결[', '3) 위계공무집행방해의 점에 관한 판단', '선고형의 결정:징역 6년']),
        }
        for identifier, (included, excluded) in expected.items():
            data = json.loads((ROOT / 'data/processed/precedents' / f'{identifier}.json').read_text(encoding='utf-8'))
            self.assertIn(included, data['retrieval_text'])
            for text in excluded:
                self.assertNotIn(text, data['retrieval_text'])
            self.assertEqual(data['retrieval_scope'], 'selected_spans')
            self.assertFalse(data['needs_manual_review'])
            self.assertEqual(data['review_resolution']['status'], 'reviewed')
            self.assertTrue(all(s['role'] in {'procedural_scope', 'party_argument',
                'court_discussion_with_arguments', 'court_acquittal', 'court_dismissal',
                'court_recorded_criminal_facts', 'evidence', 'applied_law'} for s in data['selected_spans']))

    def test_duplicate_is_excluded_only_when_cleaned_content_matches(self):
        load = lambda identifier: json.loads((ROOT / 'data/processed/precedents' / f'{identifier}.json').read_text(encoding='utf-8'))
        original, duplicate = load('606973'), load('614721')
        self.assertEqual(duplicate['duplicate_of'], original['precedent_id'])
        self.assertFalse(duplicate['embedding_candidate'])
        self.assertEqual([text_digest(original['texts'][k]) for k in TEXT_FIELDS],
                         [text_digest(duplicate['texts'][k]) for k in TEXT_FIELDS])
        formatting_duplicate, representative = load('600531'), load('614661')
        self.assertEqual(formatting_duplicate['duplicate_of'], '614661')
        self.assertFalse(formatting_duplicate['embedding_candidate'])
        self.assertTrue(representative['embedding_candidate'])
        self.assertFalse(representative['needs_manual_review'])

    def test_changed_reviewed_body_requires_review_again(self):
        review = json.loads((ROOT / 'data/processed/precedent_relevance_review.json').read_text(encoding='utf-8'))
        row = next(r for r in review['records'] if r['precedent_id'] == '605583')
        raw = json.loads((ROOT / 'data/raw/precedents/605583.json').read_text(encoding='utf-8'))
        raw['detail']['판례내용'] += '<br/>변경된 본문'
        document = preprocess(row, raw)
        self.assertTrue(document['needs_manual_review'])
        self.assertNotEqual(document['review_resolution']['status'], 'reviewed')


if __name__ == '__main__':
    unittest.main()
