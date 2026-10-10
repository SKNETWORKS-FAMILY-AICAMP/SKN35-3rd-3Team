import json
import unittest
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from unittest.mock import Mock

from src.Scripts.preprocess_laws import (
    INPUT_DIR, OUTPUT_DIR, REPORT_NAME, body_segments, clean_text,
    ensure_output, preprocess, sha256, xml_to_data, parse_box_tables,
    preprocess_supplements, content_inventory, ROOT,
)
from src.Scripts.chunk_laws import OUTPUT_DIR as CHUNK_DIR, make_chunks, make_aux_chunks


def fixture_path(article_xml):
    xml = ('<법령><기본정보><법령ID>000001</법령ID><법령명_한글>시험법</법령명_한글>'
           '<시행일자>20260101</시행일자></기본정보><조문>' + article_xml + '</조문></법령>')
    root = ET.fromstring(xml)
    detail = xml_to_data(root)
    raw = {"document_type": "law", "metadata": detail['기본정보'],
           "list_metadata": {"법령ID": "000001", "법령명한글": "시험법", "시행일자": "20260101", "법령일련번호": "123"},
           "response_xml": xml, "detail": detail, "as_of": "2026-10-10", "version_scope": "current_at_collection",
           "source": "국가법령정보센터", "source_url": "https://www.law.go.kr/LSW/lsInfoP.do?lsiSeq=123",
           "article_count": len(root.findall('./조문/조문단위'))}
    path = Mock()
    path.name = 'fixture.json'
    path.read_bytes.return_value = b'unchanged fixture'
    path.read_text.return_value = json.dumps(raw, ensure_ascii=False)
    return path


def unit(body, number='1', branch='', effective='20260101'):
    return ('<조문단위 조문키="0001001"><조문번호>' + number + '</조문번호><조문가지번호>' + branch +
            '</조문가지번호><조문여부>조문</조문여부><조문제목>시험</조문제목><조문시행일자>' + effective +
            '</조문시행일자><조문내용>' + body + '</조문내용></조문단위>')


class LawPreprocessingTests(unittest.TestCase):
    def test_nested_body_keeps_all_levels_in_source_order(self):
        raw = {'조문번호': '3', '조문내용': '제3조의2(제목) <개정 2026.1.1>',
               '항': [{'항번호': '①', '항내용': '① 조건',
                       '호': [{'호번호': '1.', '호내용': '1. 대상',
                               '목': [{'목번호': '가.', '목내용': '가. 예외\n'},
                                      {'목번호': '나.', '목내용': '나. 금액 10,000원'}]}]},
                      {'항번호': '②', '항내용': '② 7일 이내'}]}
        text, segments = body_segments(raw)
        self.assertEqual([s['level'] for s in segments], ['조', '항', '호', '목', '목', '항'])
        self.assertIn('<개정 2026.1.1>', text)
        for s in segments:
            self.assertEqual(text[s['start']:s['end']], s['text'])
        self.assertEqual(clean_text('  「민법」 제741조\r\n① 7일  '), '「민법」 제741조\n① 7일')

    def test_unlabeled_paragraph_container_keeps_subparagraphs(self):
        text, segments = body_segments({'조문내용': '제2조(정의)', '항': {'호': {'호번호': '1.', '호내용': '1. 정의'}}})
        self.assertEqual(text, '제2조(정의)\n1. 정의')
        self.assertEqual(segments[1]['path'], '항[0].호[0].호내용')

    def test_deleted_articles_are_retained_but_not_candidates(self):
        doc = preprocess(fixture_path(unit('제1조 삭제 &lt;2020.1.1&gt;')))
        a = doc['articles'][0]
        self.assertTrue(a['is_deleted'])
        self.assertFalse(a['embedding_candidate'])
        self.assertEqual(make_chunks(doc, a), [])
        self.assertEqual(a['structure']['조문내용'], '제1조 삭제 <2020.1.1>')
        active = preprocess(fixture_path(unit('제1조(요건) 일부 문구가 삭제된 경우에도 적용한다.')))
        self.assertFalse(active['articles'][0]['is_deleted'])

    def test_branch_articles_have_distinct_ids_and_future_dates_are_held(self):
        doc = preprocess(fixture_path(unit('제1조(시험) 본문') + unit('제1조의2(시험) 본문', branch='2')))
        self.assertEqual([a['article_label'] for a in doc['articles']], ['제1조', '제1조의2'])
        self.assertNotEqual(doc['articles'][0]['article_id'], doc['articles'][1]['article_id'])
        future = preprocess(fixture_path(unit('제1조(시험) 본문', effective='20270101')))
        self.assertFalse(future['articles'][0]['embedding_candidate'])
        self.assertTrue(future['articles'][0]['needs_manual_review'])

    def test_duplicate_articles_and_mismatched_xml_fail_closed(self):
        with self.assertRaises(ValueError):
            preprocess(fixture_path(unit('제1조 본문') + unit('제1조 본문')))
        path = fixture_path(unit('제1조 본문'))
        raw = json.loads(path.read_text.return_value)
        raw['detail']['조문']['조문단위']['조문내용'] = '바뀐 본문'
        path.read_text.return_value = json.dumps(raw)
        with self.assertRaises(ValueError):
            preprocess(path)

    def test_raw_output_is_rejected(self):
        with self.assertRaises(ValueError):
            ensure_output(INPUT_DIR, INPUT_DIR / 'processed')

    def test_saved_laws_preserve_every_source_unit_and_nested_body(self):
        report = json.loads((OUTPUT_DIR / REPORT_NAME).read_text(encoding='utf-8'))
        self.assertEqual(report['failures'], [])
        self.assertEqual(len(report['saved']), 8)
        for entry in report['saved']:
            path = INPUT_DIR / entry['source_file']
            raw = json.loads(path.read_text(encoding='utf-8'))
            doc = json.loads((OUTPUT_DIR / entry['file']).read_text(encoding='utf-8'))
            self.assertEqual(sha256(path), doc['raw_sha256'])
            self.assertEqual(doc['raw_sha256'], entry['raw_sha256'])
            units = raw['detail']['조문']['조문단위']
            combined = doc['articles'] + doc['structural_headings']
            self.assertEqual(sorted(x['source_unit_index'] for x in combined), list(range(len(units))))
            for item in combined:
                self.assertEqual(item['structure'], units[item['source_unit_index']])
            for a in doc['articles']:
                self.assertEqual((a['body_text'], a['body_segments']), body_segments(a['structure']))
            self.assertEqual(doc['additional_materials'],
                             {k: v for k, v in raw['detail'].items() if k not in ('기본정보', '조문')})


class LawChunkingTests(unittest.TestCase):
    def test_short_article_stays_whole_and_long_article_has_no_gaps(self):
        doc = preprocess(fixture_path(unit('제1조(시험) ' + '본문 조건 ' * 300)))
        a = doc['articles'][0]
        chunks = make_chunks(doc, a, size=150, overlap=20)
        self.assertGreater(len(chunks), 1)
        covered = set()
        for c in chunks:
            self.assertLessEqual(len(c['text']), 150)
            self.assertEqual(c['text'], a['body_text'][c['start']:c['end']])
            self.assertEqual(c['parent_context'], [])
            covered.update(range(c['start'], c['end']))
        self.assertEqual(len(covered), len(a['body_text']))
        short = preprocess(fixture_path(unit('제1조(시험) 짧은 본문')))
        self.assertEqual(len(make_chunks(short, short['articles'][0])), 1)

    def test_structural_splits_keep_parent_context_for_subparagraphs(self):
        doc = preprocess(fixture_path(unit('제1조(시험) 본문')))
        a = doc['articles'][0]
        a['body_text'], a['body_segments'] = body_segments({'조문내용': '제1조(조건)',
            '항': {'항번호': '①', '항내용': '① 다음 대상에 한해 적용한다.',
                   '호': [{'호번호': '1.', '호내용': '1. 첫 번째 대상 ' + '설명 ' * 30},
                          {'호번호': '2.', '호내용': '2. 두 번째 대상'}]}})
        chunks = make_chunks(doc, a, size=80, overlap=10)
        nested = [c for c in chunks if '항[0].호[0].호내용' in c['segment_paths']]
        self.assertTrue(nested)
        self.assertTrue(any('① 다음 대상에 한해 적용한다.' in c['embedding_text'] for c in nested))
        for previous, current in zip(chunks, chunks[1:]):
            if previous['split_basis'] == 'structural_boundary':
                self.assertEqual(previous['end'], current['start'])

    def test_invalid_chunk_parameters_and_offsets_are_rejected(self):
        doc = preprocess(fixture_path(unit('제1조 본문')))
        a = doc['articles'][0]
        for size, overlap in [(0, 0), (100, 100), (100, -1), (True, 0)]:
            with self.assertRaises(ValueError):
                make_chunks(doc, a, size, overlap)
        a['body_segments'][0]['end'] -= 1
        with self.assertRaises(ValueError):
            make_chunks(doc, a)

    def test_generated_chunks_cover_all_candidates_with_consistent_sources(self):
        report = json.loads((CHUNK_DIR / 'law_chunking_report.json').read_text(encoding='utf-8'))
        chunks_path = CHUNK_DIR / 'chunks.jsonl'
        chunks = [json.loads(line) for line in chunks_path.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(report['chunks_sha256'], sha256(chunks_path))
        self.assertEqual(len(chunks), report['chunk_count'])
        self.assertEqual(len({c['chunk_id'] for c in chunks}), len(chunks))
        grouped = defaultdict(list)
        for c in chunks:
            if c['content_type'] == 'main_article':
                grouped[c['article_id']].append(c)
        expected_ids = set()
        for entry in report['laws']:
            path = OUTPUT_DIR / entry['file']
            self.assertEqual(entry['processed_sha256'], sha256(path))
            doc = json.loads(path.read_text(encoding='utf-8'))
            for a in doc['articles']:
                actual = grouped.get(a['article_id'], [])
                self.assertEqual(actual, make_chunks(doc, a, report['size'], report['overlap']))
                if not a['embedding_candidate']:
                    self.assertFalse(actual)
                    continue
                expected_ids.add(a['article_id'])
                coverage = set()
                for c in actual:
                    coverage.update(range(c['start'], c['end']))
                    self.assertEqual(c['source_url'], doc['source_url'])
                    self.assertEqual(c['raw_sha256'], doc['raw_sha256'])
                    self.assertEqual(c['article_effective_date'], a['effective_date'])
                    self.assertTrue(c['embedding_text'].startswith(c['context_header']))
                self.assertEqual(len(coverage), len(a['body_text']))
        self.assertEqual(set(grouped), expected_ids)
        self.assertEqual(len(expected_ids), report['chunked_article_count'])


class AdditionalLawMaterialTests(unittest.TestCase):
    def test_supplement_commencement_is_not_inferred_from_promulgation(self):
        raw = {'as_of': '2026-10-10', 'detail': {'부칙': {'부칙단위': {
            '@부칙키': '20260101123', '부칙공포일자': '20260101', '부칙공포번호': '123',
            '부칙내용': '부칙 <제123호>\n\n제1조(시행일) 공포 후 6개월부터 시행한다.\n\n제2조(경과조치) 종전 규정에 따른다.'}}}}
        unit = preprocess_supplements(raw, '000001_123_20260101')[0]
        self.assertEqual(unit['promulgation_date'], '2026-01-01')
        self.assertIsNone(unit['effective_date'])
        self.assertIn('공포 후 6개월', unit['commencement_sections'][0])
        self.assertEqual(unit['structure'], raw['detail']['부칙']['부칙단위'])

    def test_box_table_keeps_empty_cells_and_does_not_infer_merged_headers(self):
        text = '표 제목\n┌──┬──┐\n│구분│금액│\n├──┼──┤\n│가  │100 │\n│나  │    │\n└──┴──┘\n비고: 예외'
        table = parse_box_tables(text)[0]
        self.assertEqual(table['physical_rows'], [['구분', '금액'], ['가', '100'], ['나', '']])
        self.assertEqual(text[table['start']:table['end']], table['text'])
        self.assertIn('나 | ', table['serialized_text'])
        with self.assertRaises(ValueError):
            parse_box_tables('┌──┬──┐\n│가│100│')

    def test_all_supplements_and_appendices_preserve_source_units(self):
        totals = {'supplements': 0, 'appendices': 0}
        for path in OUTPUT_DIR.glob('[0-9]*.json'):
            doc = json.loads(path.read_text(encoding='utf-8'))
            raw = json.loads((INPUT_DIR / doc['source_file']).read_text(encoding='utf-8'))
            for field, source in [('supplements', '부칙'), ('appendices', '별표')]:
                original = raw['detail'].get(source, {}).get(source + '단위', [])
                original = original if isinstance(original, list) else [original]
                self.assertEqual([u['structure'] for u in doc[field]], original)
                self.assertEqual(len({u['unit_id'] for u in doc[field]}), len(original))
                totals[field] += len(original)
                for u in doc[field]:
                    self.assertEqual(u['body_text'], u['structure'][source + '내용'].strip())
                    if field == 'supplements':
                        self.assertIsNone(u['effective_date'])
                    else:
                        self.assertEqual(sha256(ROOT / u['pdf_file']), u['pdf_sha256'])
                        self.assertTrue(u['pdf_comparison']['character_inventory_match'])
                        self.assertEqual(content_inventory(u['body_text']),
                                         content_inventory(''.join(p['text'] for p in u['pdf_pages'])))
        self.assertEqual(totals, {'supplements': 206, 'appendices': 8})

    def test_auxiliary_chunks_cover_nonwhitespace_and_keep_tables_atomic(self):
        chunks = [json.loads(line) for line in (CHUNK_DIR / 'chunks.jsonl').read_text(encoding='utf-8').splitlines()]
        grouped = defaultdict(list)
        for c in chunks:
            if c['content_type'] != 'main_article':
                grouped[c['unit_id']].append(c)
        expected_ids = set()
        for path in OUTPUT_DIR.glob('[0-9]*.json'):
            doc = json.loads(path.read_text(encoding='utf-8'))
            for unit in doc['supplements'] + doc['appendices']:
                actual = grouped.get(unit['unit_id'], [])
                self.assertEqual(actual, make_aux_chunks(doc, unit))
                if not unit['embedding_candidate']:
                    self.assertFalse(actual)
                    continue
                expected_ids.add(unit['unit_id'])
                covered = set()
                for c in actual:
                    self.assertEqual(c['text'], unit['body_text'][c['start']:c['end']])
                    self.assertEqual(c['raw_sha256'], doc['raw_sha256'])
                    covered.update(range(c['start'], c['end']))
                    if c['content_type'] == 'supplement':
                        self.assertIsNone(c['unit_effective_date'])
                        self.assertEqual(c['promulgation_date'], unit['promulgation_date'])
                    if c['atomic_table']:
                        table = next(t for t in unit['tables'] if t['start'] == c['start'])
                        self.assertEqual(c['end'], table['end'])
                        self.assertEqual(c['table_rows'], table['physical_rows'])
                        self.assertIn(table['serialized_text'], c['embedding_text'])
                self.assertTrue(all(i in covered or ch.isspace() for i, ch in enumerate(unit['body_text'])))
                for table in unit.get('tables', []):
                    matching = [c for c in actual if c['atomic_table'] and c['start'] == table['start']]
                    self.assertEqual(len(matching), 1)
        self.assertEqual(set(grouped), expected_ids)
        self.assertEqual(len(expected_ids), 214)

    def test_compensation_table_preserves_grade_amount_pairs(self):
        doc = json.loads(next(OUTPUT_DIR.glob('001872*.json')).read_text(encoding='utf-8'))
        appendix = doc['appendices'][0]
        rows = appendix['tables'][0]['physical_rows']
        self.assertEqual(rows[0], ['등급', '재해보상', '등급', '재해보상'])
        self.assertEqual(rows[1], ['제1급', '1,340일분', '제8급', '450일분'])
        self.assertEqual(rows[-1], ['제7급', '560일분', '제14급', '50일분'])
        import pdfplumber
        with pdfplumber.open(ROOT / appendix['pdf_file']) as pdf:
            self.assertEqual(rows, pdf.pages[0].find_tables()[0].extract())


if __name__ == '__main__':
    unittest.main()
