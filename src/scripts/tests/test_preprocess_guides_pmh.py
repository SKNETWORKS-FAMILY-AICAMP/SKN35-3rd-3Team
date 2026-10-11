import json
import copy
import unittest
import tempfile
from pathlib import Path

from src.scripts.preprocess_guides import (
    INPUT_DIR, OUTPUT_DIR, ROOT, clean_page, headings, page_kind, run, sha256, apply_visual_review, outside_tables, validate_source_coverage,
)


class GuidePreprocessingTests(unittest.TestCase):
    def test_cleanup_only_removes_identified_boundary_lines(self):
        raw = '찾기쉬운 생활법령\n「민법」 제741조: 7일 / 14일\n찾기쉬운 생활법령\n금액 10,000원\n4 / 35'
        text, removed = clean_page(raw, 4, 35)
        self.assertEqual(text, '「민법」 제741조: 7일 / 14일\n찾기쉬운 생활법령\n금액 10,000원')
        self.assertEqual([r['reason'] for r in removed], ['header', 'page_number'])
        self.assertEqual(clean_page('본문\n4 / 36', 4, 35)[0], '본문\n4 / 36')

    def test_page_classification_does_not_drop_body_with_one_dotted_line(self):
        self.assertEqual(page_kind('문서명', 1), 'cover')
        self.assertEqual(page_kind('1. 제목 . . . . . . 4\n2. 제목 . . . . . . 8', 3), 'table_of_contents')
        self.assertEqual(page_kind('1. 제목 . . . . . . 4\n본문\n제741조\n내용', 5), 'content')
        self.assertEqual(page_kind('이 정보는 기준으로 작성된 것입니다. 법적 효력', 2), 'publication_notice')

    def test_headings_have_exact_page_offsets(self):
        text = '1. 큰 제목\n설명\n1.2.3. 작은 제목\n본문'
        found = headings(text)
        self.assertEqual(len(found), 2)
        for h in found:
            self.assertEqual(text[h['start']:h['end']], h['text'])

    def test_output_cannot_write_into_raw(self):
        with self.assertRaises(ValueError):
            run(INPUT_DIR, INPUT_DIR / 'processed')
        with self.assertRaises(ValueError):
            run(Path('tmp/guide_fixture'), Path('tmp/guide_fixture/output'))

    def test_saved_documents_match_pdf_hashes_and_page_manifest(self):
        expected = {'[pdf]인터넷 쇼핑.pdf': 35, '[pdf]주택임대차.pdf': 87,
                    '[pdf]중고거래 피해자.pdf': 38, '[pdf]해고근로자.pdf': 102, '[pdf]택배.pdf':32}
        report = json.loads((OUTPUT_DIR / 'guide_preprocessing_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['failures'], [])
        self.assertEqual({e['source_file'] for e in report['saved']}, set(expected))
        for entry in report['saved']:
            doc = json.loads((OUTPUT_DIR / entry['file']).read_text(encoding='utf-8'))
            self.assertEqual(doc['raw_sha256'], sha256(INPUT_DIR / doc['source_file']))
            self.assertEqual(entry['raw_sha256'], doc['raw_sha256'])
            self.assertEqual(doc['page_count'], expected[doc['source_file']])
            self.assertEqual([p['page_number'] for p in doc['pages']], list(range(1, doc['page_count'] + 1)))
            self.assertEqual(doc['information_as_of'], '2026-09-15')
            self.assertIsNone(doc['source_url'])
            self.assertIsNone(doc['downloaded_at'])
            self.assertEqual(entry['retrieval_page_count'], sum(p['embedding_candidate'] for p in doc['pages']))
            for page in doc['pages']:
                self.assertEqual((page['text'], page['removed_lines']),
                                 clean_page(page['raw_text'], page['page_number'], doc['page_count']))
                self.assertEqual(page['headings'], headings(page['text']))
                if page['embedding_candidate']:
                    self.assertEqual(page['page_kind'], 'content')
                    self.assertFalse(page['needs_manual_review'])
                    self.assertTrue(page['text'])
                if page['images'] and page['page_kind'] == 'content':
                    self.assertFalse(page['visual_review_required'])
                    self.assertFalse(page['needs_manual_review'])
                    self.assertTrue(page['embedding_candidate'])
                    self.assertEqual(len(page['visual_summaries']), len(page['images']))
                for table in page['tables']:
                    self.assertEqual(len(table['bbox']), 4)
                    self.assertIsInstance(table['rows'], list)
                    self.assertTrue(table['rows'])

    def test_representative_pdf_pages_and_tables_match_extraction(self):
        import pdfplumber

        report = json.loads((OUTPUT_DIR / 'guide_preprocessing_report.json').read_text(encoding='utf-8'))
        for entry in report['saved']:
            doc = json.loads((OUTPUT_DIR / entry['file']).read_text(encoding='utf-8'))
            with pdfplumber.open(INPUT_DIR / doc['source_file']) as pdf:
                table_page = next(p['page_number'] - 1 for p in doc['pages'] if p['tables'])
                for index in {0, 1, table_page, len(pdf.pages) - 1}:
                    self.assertEqual(doc['pages'][index]['raw_text'], pdf.pages[index].extract_text() or '')
                self.assertEqual(doc['pages'][table_page]['tables'],
                                 [{'bbox': list(t.bbox), 'rows': t.extract()} for t in pdf.pages[table_page].find_tables()])
                source = pdf.pages[table_page]
                body = source.filter(lambda obj: outside_tables(obj, doc['pages'][table_page]['tables'])).extract_text() or ''
                self.assertEqual(doc['pages'][table_page]['body_text'], clean_page(body, table_page+1, len(pdf.pages))[0])

    def test_stale_or_missing_review_does_not_release_image_page(self):
        manifest = json.loads((OUTPUT_DIR / 'guide_visual_review.json').read_text(encoding='utf-8'))
        doc = json.loads((OUTPUT_DIR / 'guide_23ffe4c8fd5715cb.json').read_text(encoding='utf-8'))
        for field in ['raw_sha256', 'stream_sha256', 'status', 'summary']:
            changed = copy.deepcopy(manifest)
            changed['reviews'][0][field] = ''
            revised = apply_visual_review(copy.deepcopy(doc), changed)
            page = revised['pages'][68]
            self.assertTrue(page['needs_manual_review'])
            self.assertFalse(page['embedding_candidate'])
            self.assertEqual(page['visual_summaries'], [])
        revised = apply_visual_review(copy.deepcopy(doc), {'reviews': []})
        self.assertFalse(revised['pages'][68]['embedding_candidate'])

    def test_exported_images_match_review_source_hashes(self):
        import hashlib
        import pdfplumber
        from src.scripts.preprocess_guides import ROOT
        report = json.loads((OUTPUT_DIR / 'guide_preprocessing_report.json').read_text(encoding='utf-8'))
        total = 0
        for entry in report['saved']:
            doc = json.loads((OUTPUT_DIR / entry['file']).read_text(encoding='utf-8'))
            with pdfplumber.open(INPUT_DIR / doc['source_file']) as pdf:
                for page in doc['pages']:
                    for summary in page.get('visual_summaries', []):
                        image = page['images'][summary['image_index']]
                        source = pdf.pages[page['page_number']-1].images[summary['image_index']]
                        self.assertEqual(summary['stream_sha256'], hashlib.sha256(source['stream'].get_data()).hexdigest())
                        self.assertEqual(image['image_sha256'], sha256(ROOT / image['image_path']))
                        self.assertEqual(summary['raw_sha256'], doc['raw_sha256'])
                        total += 1
        self.assertEqual(total, 21)

    def test_every_current_raw_pdf_has_a_processing_record(self):
        report=json.loads((OUTPUT_DIR/'guide_preprocessing_report.json').read_text(encoding='utf-8'))
        self.assertTrue(validate_source_coverage(report))
        self.assertEqual({e['source_file'] for e in report['saved']+report['failures']},
                         {p.name for p in INPUT_DIR.glob('*.pdf')})

    def test_coverage_detects_new_removed_changed_and_duplicate_sources(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            folder=Path(directory)
            original=folder/'original.pdf';original.write_bytes(b'original')
            report={'saved':[{'source_file':'original.pdf'}],'failures':[],
                    'input_files':[{'source_file':'original.pdf','raw_sha256':sha256(original)}]}
            self.assertTrue(validate_source_coverage(report,folder))
            added=folder/'new.pdf';added.write_bytes(b'new')
            with self.assertRaisesRegex(ValueError,'미기록'):
                validate_source_coverage(report,folder)
            added.unlink()
            original.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'지문 변경'):
                validate_source_coverage(report,folder)
            original.write_bytes(b'original')
            duplicate=copy.deepcopy(report);duplicate['saved']*=2
            with self.assertRaises(ValueError):validate_source_coverage(duplicate,folder)
            original.unlink()
            with self.assertRaisesRegex(ValueError,'원본 없음'):
                validate_source_coverage(report,folder)

    def test_parcel_guide_images_are_reviewed_and_only_front_matter_is_excluded(self):
        doc=json.loads((OUTPUT_DIR/'guide_94e9d4bd30eadf0b.json').read_text(encoding='utf-8'))
        self.assertEqual(doc['page_count'],32)
        self.assertEqual([p['page_number'] for p in doc['pages'] if not p['embedding_candidate']],[1,2,3])
        for number in [9,29,31]:
            page=doc['pages'][number-1]
            self.assertTrue(page['embedding_candidate'])
            self.assertEqual(len(page['visual_summaries']),1)
            self.assertFalse(page['needs_manual_review'])


if __name__ == '__main__':
    unittest.main()
