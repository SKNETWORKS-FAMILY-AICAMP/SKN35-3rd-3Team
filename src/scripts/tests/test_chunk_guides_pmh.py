import copy
import json
import unittest
import tempfile
from pathlib import Path
from collections import Counter

from src.Scripts.chunk_guides import INPUT_DIR, OUTPUT_DIR, make_chunks, run
from src.Scripts.preprocess_guides import sha256, ROOT


class GuideChunkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((OUTPUT_DIR / 'guide_chunking_report.json').read_text(encoding='utf-8'))
        cls.documents = []
        for entry in cls.report['saved']:
            payload = json.loads((OUTPUT_DIR / entry['file']).read_text(encoding='utf-8'))
            source = INPUT_DIR / (payload['guide_id'] + '.json')
            cls.documents.append((entry, json.loads(source.read_text(encoding='utf-8')), payload['chunks']))
            if sha256(source) != entry['processed_sha256']:
                raise AssertionError('정제 입력 지문 불일치')

    def test_all_eligible_pages_and_only_eligible_pages_represented(self):
        for entry, doc, chunks in self.documents:
            expected = {p['page_number'] for p in doc['pages'] if p['embedding_candidate']}
            self.assertEqual({c['page_number'] for c in chunks}, expected)
            self.assertEqual(entry['represented_pages'], sorted(expected))
        self.assertEqual(sum(len(e['represented_pages']) for e in self.report['saved']), 277)

    def test_body_windows_cover_every_character_with_correct_offsets(self):
        for entry, doc, chunks in self.documents:
            for page in doc['pages']:
                if not page['embedding_candidate']:
                    continue
                text = page['body_text']
                cursor = 0
                for c in [c for c in chunks if c['page_number'] == page['page_number'] and c['content_type'] == 'body']:
                    self.assertLessEqual(c['start'], cursor)
                    self.assertGreater(c['end'], cursor)
                    self.assertEqual(c['text'], text[c['start']:c['end']])
                    self.assertLessEqual(len(c['text']), self.report['size'])
                    cursor = c['end']
                self.assertEqual(cursor, len(text))

    def test_tables_preserve_all_rows_and_nulls_without_fragmenting(self):
        for entry, doc, chunks in self.documents:
            actual = [c for c in chunks if c['content_type'] == 'table']
            expected = [(p['page_number'], t) for p in doc['pages'] if p['embedding_candidate'] for t in p['tables']]
            self.assertEqual(len(actual), len(expected))
            for c, (page, table) in zip(actual, expected):
                self.assertEqual(c['page_number'], page)
                self.assertEqual(c['table_rows'], table['rows'])
                self.assertEqual([json.loads(line) for line in c['text'].splitlines()], table['rows'])
                self.assertTrue(c['atomic'])

    def test_visual_summaries_are_explicit_and_match_reviewed_records(self):
        count = 0
        for entry, doc, chunks in self.documents:
            for c in chunks:
                if c['content_type'] == 'image_summary':
                    self.assertEqual(c['text'], c['visual_review']['summary'])
                    self.assertEqual(c['extraction_method'], 'manual_visual_summary')
                    self.assertTrue(c['not_full_transcription'])
                    self.assertIn('전문 전사 아님', c['embedding_text'])
                    count += 1
        self.assertEqual(count, 21)

    def test_parcel_guide_chunks_keep_all_body_pages_tables_and_reviewed_images(self):
        entry,doc,chunks=next(x for x in self.documents if x[1]['source_file']=='[pdf]택배.pdf')
        self.assertEqual(entry['represented_pages'],list(range(4,33)))
        self.assertEqual(entry['excluded_pages'],[1,2,3])
        self.assertEqual([p['page_kind'] for p in entry['exclusion_details']],['cover','publication_notice','table_of_contents'])
        self.assertEqual(sum(c['content_type']=='table' for c in chunks),7)
        self.assertEqual({c['page_number'] for c in chunks if c['content_type']=='image_summary'},{9,29,31})

    def test_unrecorded_raw_source_blocks_chunk_writes(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            root=Path(directory);raw=root/'raw';processed=root/'processed';output=root/'chunks'
            raw.mkdir();processed.mkdir()
            first=raw/'first.pdf';first.write_bytes(b'first')
            report={'raw_input_dir':str(raw),'saved':[{'source_file':'first.pdf'}],'failures':[],
                    'input_files':[{'source_file':'first.pdf','raw_sha256':sha256(first)}]}
            (processed/'guide_preprocessing_report.json').write_text(json.dumps(report),encoding='utf-8')
            (raw/'new.pdf').write_bytes(b'new')
            with self.assertRaisesRegex(ValueError,'미기록'):
                run(processed,output)
            self.assertFalse(output.exists())

    def test_ids_metadata_and_report_counts(self):
        all_chunks = []
        for entry, doc, chunks in self.documents:
            self.assertEqual(entry['chunk_count'], len(chunks))
            self.assertEqual(entry['counts'], dict(Counter(c['content_type'] for c in chunks)))
            for c in chunks:
                self.assertEqual(c['raw_sha256'], doc['raw_sha256'])
                self.assertEqual(c['information_as_of'], doc['information_as_of'])
                self.assertIn(str(c['page_number'])+'페이지', c['citation'])
                self.assertTrue(c['text'].strip())
                self.assertTrue(c['embedding_text'].endswith(c['text']))
            all_chunks.extend(chunks)
        self.assertEqual(len({c['chunk_id'] for c in all_chunks}), len(all_chunks))
        self.assertEqual(self.report['chunk_count'], len(all_chunks))
        self.assertEqual(self.report['counts'], dict(Counter(c['content_type'] for c in all_chunks)))

    def test_pending_page_is_excluded_and_output_is_deterministic(self):
        entry, doc, chunks = self.documents[0]
        self.assertEqual(make_chunks(doc), chunks)
        changed = copy.deepcopy(doc)
        page = next(p for p in changed['pages'] if p['embedding_candidate'])
        page['needs_manual_review'] = True
        self.assertFalse(any(c['page_number'] == page['page_number'] for c in make_chunks(changed)))

    def test_invalid_parameters_rejected(self):
        doc = self.documents[0][1]
        for size, overlap in [(0, 0), (3, 3), (5, -1), (True, 0), (1200, 1.5)]:
            with self.assertRaises(ValueError):
                make_chunks(doc, size, overlap)

    def test_cannot_write_to_raw_or_processed(self):
        for output in [INPUT_DIR, INPUT_DIR / 'output', INPUT_DIR.parent.parent / 'raw/guides']:
            with self.assertRaises(ValueError):
                run(INPUT_DIR, output)


if __name__ == '__main__':
    unittest.main()
