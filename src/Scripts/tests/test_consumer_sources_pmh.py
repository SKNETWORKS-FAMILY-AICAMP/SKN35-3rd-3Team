import copy
import json
import tempfile
import unittest
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from src.Scripts.collect_consumer_sources import ROOT, MANIFEST, digest, preserve, safe_url
from src.Scripts.preprocess_consumer_sources import (OUTPUT_DIR, PARCEL_VERSIONS, extract_case,
    extract_admin, preprocess, hwp_paragraph_text, compare_parcel_versions, refund_table_rows)
from src.Scripts.chunk_consumer_sources import OUTPUT_DIR as CHUNK_DIR, make_chunks


class ConsumerSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((OUTPUT_DIR/'consumer_preprocessing_report.json').read_text(encoding='utf-8'))
        cls.documents=[json.loads((OUTPUT_DIR/e['file']).read_text(encoding='utf-8')) for e in cls.report['saved']]
        cls.chunks=[json.loads(line) for line in (CHUNK_DIR/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_collection_succeeded_and_raw_hashes_unchanged(self):
        report=json.loads((OUTPUT_DIR/'consumer_collection_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['failures'],[])
        self.assertEqual(self.report['failures'],[])
        for record in report['saved']:
            self.assertEqual(record['raw_sha256'],digest((ROOT/record['raw_file']).read_bytes()))
            self.assertNotIn('OC=',record['source_url'])

    def test_preserve_is_immutable_and_rejects_path_escape(self):
        (ROOT/'tmp').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'tmp') as directory:
            first=preserve(b'original',{'source_id':'fixture'},'html',directory)
            path=Path(first['raw_file'])
            before=path.stat().st_mtime_ns
            self.assertEqual(first['raw_file'],preserve(b'original',{'source_id':'fixture'},'html',directory)['raw_file'])
            self.assertEqual(before,path.stat().st_mtime_ns)
            second=preserve(b'changed',{'source_id':'fixture'},'html',directory)
            self.assertNotEqual(first['raw_file'],second['raw_file'])
            self.assertEqual(path.read_bytes(),b'original')
            with self.assertRaises(ValueError):
                preserve(b'x',{'source_id':'../escape'},'html',directory)

    def test_authentication_is_removed_from_persisted_url(self):
        self.assertEqual(safe_url('https://www.law.go.kr/a?OC=secret&ID=1'),'https://www.law.go.kr/a?ID=1')

    def test_cases_keep_facts_claims_and_reasoning_separate(self):
        cases=[d for d in self.documents if d['document_type']=='mediation_case']
        self.assertEqual(len(cases),19)
        for doc in cases:
            roles={u['role'] for u in doc['units']}
            self.assertTrue({'facts','reasoning'}.issubset(roles))
            self.assertEqual('party_claims' not in roles,'party_claims' in doc['source_missing_fields'])
            self.assertIsNone(doc['decision_date'])
            self.assertTrue(doc['updated_date'])
            self.assertNotIn('조회수','\n'.join(u['text'] for u in doc['units']))
        with self.assertRaises(ValueError):
            extract_case(b'<h4>fake</h4><p>error page</p>')

    def test_admin_sections_exactly_match_official_xml_and_keep_entire_tables(self):
        doc=next(d for d in self.documents if d['document_type']=='administrative_rule')
        root=ET.fromstring((ROOT/doc['raw_file']).read_bytes())
        text=root.find('.//별표단위[별표번호="0002"]/별표내용').text
        units=[u for u in doc['units'] if u['role']=='industry_table']
        self.assertEqual({u['domain'] for u in units},{'fitness_refund','parcel_loss','shopping_return'})
        for u in units:
            self.assertEqual(u['text'],text[u['start']:u['end']])
            self.assertIn('비',u['text'])
            self.assertTrue(u['atomic'])
        self.assertEqual(doc['effective_date'],root.findtext('.//시행일자'))

    def test_additional_appendices_preserve_complete_official_text_and_provenance(self):
        doc=next(d for d in self.documents if d['document_type']=='administrative_rule')
        root=ET.fromstring((ROOT/doc['raw_file']).read_bytes())
        units=[u for u in doc['units'] if u['role']=='appendix_table']
        self.assertEqual({u['appendix_number'] for u in units},{'0003','0004'})
        for u in units:
            appendix=root.find(f'.//별표단위[별표번호="{u["appendix_number"]}"]')
            body=appendix.findtext('별표내용')
            self.assertEqual(u['text'],body)
            self.assertEqual(u['appendix_title'],appendix.findtext('별표제목'))
            self.assertEqual((u['start'],u['end']),(0,len(body)))
            found=[c for c in self.chunks if c['source_id']==doc['source_id'] and c['unit_id']==u['unit_id']]
            self.assertEqual(len(found),1)
            c=found[0]
            self.assertEqual(c['text'],body)
            self.assertEqual(c['source_offset_basis'],u['offset_basis'])
            self.assertEqual(c['source_end'],len(body))
            self.assertIn(u['appendix_title'],c['embedding_text'])
            self.assertIn('consumer_general',c['domains'])
            self.assertTrue(c['atomic'])
            self.assertTrue(c['oversize_atomic_unit'])
            self.assertTrue(c['warnings'])

    def test_product_tables_keep_continuations_notes_and_exact_official_offsets(self):
        doc=next(d for d in self.documents if d['document_type']=='administrative_rule')
        root=ET.fromstring((ROOT/doc['raw_file']).read_bytes())
        text=root.find('.//별표단위[별표번호="0002"]/별표내용').text
        units={u['unit_id']:u for u in doc['units'] if u['role']=='product_table'}
        self.assertEqual(set(units),{'product_jewelry','product_electronics_office'})
        for u in units.values():
            self.assertEqual(u['text'],text[u['start']:u['end']])
            self.assertTrue(u['text'].startswith('┏'))
            self.assertIn('┛',u['text'])
            self.assertIn('비',u['text'])
            chunks=[c for c in self.chunks if c['unit_id']==u['unit_id']]
            self.assertEqual(len(chunks),1)
            self.assertEqual(chunks[0]['text'],u['text'])
            self.assertEqual(chunks[0]['product_scope'],u['product_scope'])
            self.assertEqual(chunks[0]['source_start'],u['start'])
            self.assertIn('consumer_general',chunks[0]['domains'])
        electronics=units['product_electronics_office']['text']
        self.assertIn('(2 - 1)',electronics)
        self.assertIn('(2 - 2)',electronics)
        self.assertIn('동일하자',electronics)
        self.assertNotIn('②전기통신기자재',electronics)
        self.assertNotIn('②악',units['product_jewelry']['text'])
        for label in ['귀금속','전자제품']:
            changed=copy.deepcopy(root)
            appendix=changed.find('.//별표단위[별표번호="0002"]/별표내용')
            appendix.text=appendix.text.replace(label,'누락된품목')
            with self.assertRaises(ValueError):
                extract_admin(ET.tostring(changed,encoding='utf-8'))

    def test_refund_rows_resolve_only_source_merges_and_keep_common_conditions(self):
        count=0
        for doc in self.documents:
            for u in doc['units']:
                for table in u.get('refund_tables',[]):
                    count+=1
                    cells=table['cells']
                    self.assertEqual(cells,json.loads(u['text'][table['start']:table['end']]))
                    rows=table['rows']
                    self.assertEqual([r['row_index'] for r in rows],[1,2,3,4])
                    for row in rows:
                        for col,index in enumerate(row['column_cell_indices']):
                            cell=cells[index]
                            self.assertLessEqual(int(cell['address']['rowAddr']),row['row_index'])
                            self.assertLess(row['row_index'],int(cell['address']['rowAddr'])+int(cell['span']['rowSpan']))
                            self.assertLessEqual(int(cell['address']['colAddr']),col)
                            self.assertLess(col,int(cell['address']['colAddr'])+int(cell['span']['colSpan']))
                        self.assertEqual(row['condition_texts'],[cells[i]['text'] for i in row['column_cell_indices'][:2]])
                        self.assertEqual(row['refund_text'],cells[row['column_cell_indices'][2]]['text'])
                        self.assertEqual(row['common_texts'],[cells[i]['text'] for i in row['common_cell_indices']])
                        self.assertTrue(any('위약금' in s and '이용료' in s for s in row['common_texts']))
                    self.assertEqual(rows[0]['condition_texts'][0],rows[1]['condition_texts'][0])
                    self.assertEqual(rows[2]['condition_texts'][0],rows[3]['condition_texts'][0])
                    self.assertIn('이전',rows[0]['condition_texts'][1])
                    self.assertIn('이후',rows[1]['condition_texts'][1])
                    for problem in ['overlap','missing','invalid_span']:
                        bad=copy.deepcopy(cells)
                        if problem=='overlap':bad.append(copy.deepcopy(cells[0]))
                        elif problem=='missing':bad.pop(rows[1]['column_cell_indices'][1])
                        else:bad[0]['span']['rowSpan']='0'
                        with self.subTest(problem=problem), self.assertRaises(ValueError):
                            refund_table_rows(bad)
        self.assertEqual(count,3)

    def test_additional_appendices_reject_missing_empty_duplicate_and_wrong_title(self):
        doc=next(d for d in self.documents if d['document_type']=='administrative_rule')
        original=ET.fromstring((ROOT/doc['raw_file']).read_bytes())
        for number in ['0003','0004']:
            for problem in ['missing','empty','duplicate','title']:
                root=copy.deepcopy(original)
                target=root.find(f'.//별표단위[별표번호="{number}"]')
                if problem=='missing':
                    parent=next(n for n in root.iter() if target in list(n))
                    parent.remove(target)
                elif problem=='empty':
                    target.find('별표내용').text='  '
                elif problem=='duplicate':
                    root.append(copy.deepcopy(target))
                else:
                    target.find('별표제목').text='wrong'
                with self.subTest(number=number,problem=problem), self.assertRaises(ValueError):
                    extract_admin(ET.tostring(root,encoding='utf-8'))

    def test_hwpx_equations_cells_and_merged_spans_are_preserved(self):
        for doc in self.documents:
            if doc['format']!='hwpx':
                continue
            with zipfile.ZipFile(ROOT/doc['raw_file']) as archive:
                root=ET.fromstring(archive.read('Contents/section0.xml'))
            equations=Counter(n.text for n in root.iter() if n.tag.endswith('}script'))
            text=doc['extracted_text']
            self.assertEqual(text.count('〔수식 원문: '),sum(equations.values()))
            for formula,count in equations.items():
                # JSON 표 문자열에는 수식이 그대로 포함된다.
                self.assertGreaterEqual(text.count(formula),count)
            self.assertEqual(''.join(u['text'] for u in doc['units']),text)
            for cells in doc['tables']:
                for cell in cells:
                    self.assertIn('rowAddr',cell['address'])
                    self.assertIn('colAddr',cell['address'])
                    self.assertIn('rowSpan',cell['span'])

    def test_pending_and_reference_documents_never_enter_chunks(self):
        pending={d['source_id'] for d in self.documents if not d['embedding_candidate']}
        self.assertTrue(pending)
        self.assertFalse(pending.intersection(c['source_id'] for c in self.chunks))
        doc=copy.deepcopy(next(d for d in self.documents if d['embedding_candidate']))
        doc['needs_manual_review']=True
        self.assertEqual(make_chunks(doc),[])

    def test_chunk_offsets_cover_all_units_and_keep_atomic_units_whole(self):
        for doc in self.documents:
            if not doc['embedding_candidate']:
                continue
            chunks=make_chunks(doc)
            self.assertEqual(chunks,[c for c in self.chunks if c['source_id']==doc['source_id']])
            for u in doc['units']:
                found=[c for c in chunks if c['unit_id']==u['unit_id']]
                cursor=0
                for c in found:
                    self.assertLessEqual(c['start'],cursor)
                    self.assertGreater(c['end'],cursor)
                    self.assertEqual(c['text'],u['text'][c['start']:c['end']])
                    self.assertEqual(c['raw_sha256'],doc['raw_sha256'])
                    cursor=c['end']
                self.assertEqual(cursor,len(u['text']))
                if u.get('atomic'):
                    self.assertEqual(len(found),1)

    def test_chunk_manifest_and_processed_fingerprints_match(self):
        report=json.loads((CHUNK_DIR/'consumer_chunking_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['chunks_sha256'],digest((CHUNK_DIR/'chunks.jsonl').read_bytes()))
        self.assertEqual(report['chunk_count'],len(self.chunks))
        self.assertEqual(report['document_count'],23)
        self.assertEqual(len({c['chunk_id'] for c in self.chunks}),len(self.chunks))
        for e in report['inputs']:
            self.assertEqual(e['processed_sha256'],digest((OUTPUT_DIR/e['file']).read_bytes()))

    def test_invalid_chunk_sizes_fail(self):
        for size,overlap in [(0,0),(5,5),(False,0),(5,-1)]:
            with self.assertRaises(ValueError):
                make_chunks(self.documents[0],size,overlap)

    def test_search_passages_cover_original_body_and_table_cell_texts_without_inference(self):
        for c in self.chunks:
            passages=c['search_passages']
            covered=set()
            for p in passages:
                self.assertEqual(p['text'],c['text'][p['start']:p['end']])
                self.assertTrue(0<=p['start']<p['end']<=len(c['text']))
                if 'search_text' in p:
                    cells=json.loads(p['text'])
                    if p['derivation']=='table_cell_texts_in_source_order_no_merge_inference':
                        self.assertEqual(p['search_text'],'\n'.join(cell['text'] for cell in cells))
                    else:
                        self.assertEqual(p['derivation'],'validated_refund_row_with_common_notes_and_retrieval_aliases')
                        table=c['refund_tables'][p['table_index']]
                        row=next(r for r in table['rows'] if r['row_index']==p['row_index'])
                        expected='\n'.join([c['text'].splitlines()[0],*row['condition_texts'],row['refund_text'],*row['common_texts']])
                        self.assertEqual(p['search_text'],expected+'\n[검색용 별칭] '+' '.join(p['search_aliases']))
                        self.assertEqual(cells,table['cells'])
                else:
                    covered.update(range(p['start'],p['end']))
            self.assertEqual(len(covered),len(c['text']))
            self.assertEqual(c['evidence_role']=='party_claim',c['role']=='party_claims')

    def test_parcel_versions_preserve_complete_articles_pages_and_conversion_hashes(self):
        for doc in self.documents:
            if doc['source_id'] not in PARCEL_VERSIONS:
                continue
            _, revision, count, role = PARCEL_VERSIONS[doc['source_id']]
            self.assertEqual(doc['terms_revision_date'],revision)
            self.assertEqual(doc['version_role'],role)
            self.assertEqual([u['article_number'] for u in doc['units'][1:]],list(range(1,count+1)))
            self.assertEqual(''.join(u['text'] for u in doc['units']),doc['extracted_text'])
            self.assertTrue(doc['conversion_validation']['body_character_counts_match'])
            self.assertTrue(doc['conversion_validation']['article_character_counts_match'])
            self.assertEqual(doc['derived_pdf_sha256'],digest((ROOT/doc['derived_pdf_file']).read_bytes()))
            pages=set()
            for u in doc['units']:
                self.assertEqual(u['text'],doc['extracted_text'][u['start']:u['end']])
                pages.update(u['page_numbers'])
            self.assertEqual(pages,set(range(1,doc['pdf_page_count']+1)))
            changed=copy.deepcopy(doc)
            changed['raw_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'지문'):
                preprocess(changed)

    def test_historical_parcel_version_is_not_in_default_search(self):
        historical=next(d for d in self.documents if d['source_id']=='ftc_terms_11125_file_e27d5f98')
        preferred=next(d for d in self.documents if d['source_id']=='ftc_terms_11185_file_a321590d')
        self.assertTrue(historical['historical_reference'])
        self.assertEqual(make_chunks(historical),[])
        self.assertEqual(len(make_chunks(preferred)),27)
        for chunk in make_chunks(preferred):
            self.assertEqual(chunk['terms_revision_date'],'2020-06-05')
            self.assertTrue(chunk['page_numbers'])
            self.assertIn('2020-06-05',chunk['embedding_text'])

    def test_revision_comparison_aligns_renumbered_articles_and_keeps_changes(self):
        report=json.loads((OUTPUT_DIR/'parcel_terms_revision_comparison.json').read_text(encoding='utf-8'))
        self.assertEqual(report,compare_parcel_versions(self.documents))
        self.assertEqual(report['old_article_count'],23)
        self.assertEqual(report['new_article_count'],26)
        added=[a['article_title'] for a in report['articles'] if a['status']=='added']
        self.assertEqual(set(added),{'사업자의 의무','송화인의 의무','분쟁해결'})
        damages=next(a for a in report['articles'] if a['article_title']=='손해배상')
        self.assertEqual((damages['old_article_number'],damages['new_article_number']),(20,22))
        self.assertNotIn('30일 이내',damages['old_text'])
        self.assertIn('30일 이내',damages['new_text'])
        self.assertTrue(damages['diff'])

    def test_original_ambiguous_rate_is_preserved_and_warning_is_searchable(self):
        doc=next(d for d in self.documents if d['source_id']=='ftc_terms_11185_file_a321590d')
        unit=next(u for u in doc['units'] if u['article_number']==22)
        self.assertIn('운송장기재운임액의 200\n',unit['text'])
        self.assertNotIn('운송장기재운임액의 200%',unit['text'])
        chunk=next(c for c in make_chunks(doc) if c['article_number']==22)
        self.assertEqual(chunk['source_anomaly'],'missing_rate_unit_in_original')
        self.assertIn('단위가 없음',chunk['embedding_text'])
        self.assertIn('계산하지 말 것',chunk['embedding_text'])
        self.assertEqual(chunk['calculation_restricted_clause'],'제22조 제2항 제3호 나목')

    def test_hwp_reader_fails_closed_on_unsupported_or_truncated_input(self):
        with self.assertRaises(ValueError):
            hwp_paragraph_text(b'not a compound file')
        doc=next(d for d in self.documents if d['source_id'] in PARCEL_VERSIONS)
        data=(ROOT/doc['raw_file']).read_bytes()
        with self.assertRaises(ValueError):
            hwp_paragraph_text(data[:512])


if __name__=='__main__':
    unittest.main()
