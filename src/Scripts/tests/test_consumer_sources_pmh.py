import copy
import json
import re
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from src.Scripts.collect_consumer_sources import ROOT, MANIFEST, digest, preserve, safe_url, collect_admin, fetch
from src.Scripts.preprocess_consumer_sources import (OUTPUT_DIR, PARCEL_VERSIONS, extract_case,
    extract_admin, preprocess, hwp_paragraph_text, compare_parcel_versions, refund_table_rows, extract_labor_guide, extract_overseas_guide, extract_fitness_closure_guide, installment_context,
    CIVIL_RECOVERY_GUIDES, HOUSING_GUIDES, extract_civil_recovery_guide,
    CONSUMER_REDRESS_GUIDES, extract_consumer_redress_guide)
from src.Scripts.chunk_consumer_sources import OUTPUT_DIR as CHUNK_DIR, make_chunks


class ConsumerSourceTests(unittest.TestCase):
    def test_consumer_redress_units_reproduce_sources_and_keep_warnings_and_law_links(self):
        by_id={d['source_id']:d for d in self.documents}
        law_ids={json.loads(line)['chunk_id'] for line in (ROOT/'data/chunk/laws/chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()}
        for identifier,count in [('kca_redress_requirements',2),('kca_redress_submission',5),('kca_mediation_procedure',7)]:
            d=by_id[identifier]
            self.assertEqual(len(d['units']),count)
            self.assertIsNone(d['information_as_of'])
            self.assertIsNone(d['effective_date'])
            self.assertEqual(preprocess(d)['units'],d['units'])
            chunks=make_chunks(d)
            self.assertEqual(chunks,[c for c in self.chunks if c['source_id']==identifier])
            self.assertEqual(len(chunks),count)
            for c in chunks:
                self.assertEqual(c['domains'],['consumer_general'])
                self.assertTrue(c['atomic'])
                self.assertTrue(c['required_context_chunk_ids'])
                self.assertTrue(set(c['required_context_chunk_ids']) <= law_ids | {'kca_damage_redress_procedure_redress_exclusions_0000'})
                for warning in c['warnings']:
                    self.assertIn(warning,c['embedding_text'])

    def test_consumer_redress_does_not_generalize_specialized_forms_online_time_or_specific_event_popup(self):
        docs={d['source_id']:d for d in self.documents}
        specialized=next(u for u in docs['kca_redress_requirements']['units'] if u['unit_id']=='specialized_forms')
        self.assertIn('일반분야를 제외한',specialized['text'])
        self.assertIn('일반분야의 모든 신청',specialized['warnings'][0])
        online=next(u for u in docs['kca_redress_submission']['units'] if u['unit_id']=='online')
        self.assertIn('6개월 이내',online['text'])
        self.assertIn('소멸시효',online['warnings'][0])
        body=' '.join(u['text'] for u in docs['kca_redress_submission']['units'])
        for excluded in ['알렛츠','인터스텔라','티몬','해피머니','구매자 계정']:
            self.assertNotIn(excluded,body)
        self.assertIn('특정 사건',str(docs['kca_redress_submission']['scope_excluded']))

    def test_consumer_mediation_acceptance_and_no_response_are_connected_to_article_67(self):
        d=next(d for d in self.documents if d['source_id']=='kca_mediation_procedure')
        by_id={c['chunk_id']:c for c in self.chunks}
        definition=by_id['kca_mediation_procedure_definition_0000']
        self.assertIn('불수락',definition['embedding_text'])
        acceptance=by_id['kca_mediation_procedure_acceptance_0000']
        for phrase in ['통지를 받은 날부터 15일','의사표시가 없는','재판상 화해']:
            self.assertIn(phrase,acceptance['text'])
        laws=[json.loads(line) for line in (ROOT/'data/chunk/laws/chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        actual=[c for c in laws if c['chunk_id'] in acceptance['required_context_chunk_ids'] and c.get('article_label')=='제67조']
        self.assertEqual(len(actual),1)
        self.assertEqual(actual[0]['law_name'],'소비자기본법')
        self.assertIn('수락한 것으로 본다',actual[0]['text'])

    def test_consumer_redress_missing_conditions_and_block_pages_fail_closed(self):
        from src.Scripts.collect_consumer_sources import decode_html
        for identifier,phrase in [('kca_redress_requirements','일반분야를 제외한'),
                                  ('kca_redress_submission','6개월 이내'),
                                  ('kca_mediation_procedure','의사표시가 없는')]:
            d=next(d for d in self.documents if d['source_id']==identifier)
            html=decode_html((ROOT/d['raw_file']).read_bytes())
            html=re.sub(r'(?i)(charset\s*=\s*[\"\']?)[\w-]+',r'\1utf-8',html)
            for damaged in ['blocked',html.replace(phrase,'')]:
                with self.assertRaises(ValueError):
                    extract_consumer_redress_guide(d,damaged.encode('utf-8'))

    def test_general_warranty_rule_preserves_unknown_purchase_date_exception_and_pdf_comparison(self):
        laws=[json.loads(p.read_text(encoding='utf-8')) for p in (ROOT/'data/processed/laws').glob('[0-9]*.json')]
        d=next(d for d in laws if d['law_name']=='소비자기본법 시행령')
        unit=next(u for u in d['appendices'] if u['appendix_key']=='000100E')
        compact=re.sub(r'\s+','',unit['body_text'])
        for phrase in ['영수증을받지아니하거나분실한경우','제조일이나수입통관일부터3월이',
                       '사업자가그판매일자를입증하여야한다','계약일과인도일','교환받은날부터']:
            self.assertIn(phrase,compact)
        self.assertTrue(unit['pdf_comparison']['character_inventory_match'])
        chunks=[json.loads(line) for line in (ROOT/'data/chunk/laws/chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        selected=[c for c in chunks if c.get('unit_id')==unit['unit_id']]
        self.assertTrue(selected)
        self.assertTrue(any('판매일자를' in c['text'] for c in selected))

    def test_housing_guides_keep_conditions_exceptions_provenance_and_law_links(self):
        from src.Scripts.collect_consumer_sources import HTMLTree,decode_html
        docs={d['source_id']:d for d in self.documents}
        law_ids={json.loads(line)['chunk_id'] for line in (ROOT/'data/chunk/laws/chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()}
        for identifier,scope in HOUSING_GUIDES.items():
            d=docs[identifier]
            self.assertEqual(d['information_as_of'],'2026-09-15')
            self.assertIsNone(d['effective_date'])
            self.assertEqual(len(d['units']),len(scope))
            self.assertEqual(preprocess(d)['units'],d['units'])
            tree=HTMLTree(decode_html((ROOT/d['raw_file']).read_bytes())).root
            nodes={n.attrs['id']:n for n in tree.find(lambda n:bool(n.attrs.get('id')))}
            for u in d['units']:
                expected='\n'.join(' '.join(nodes[i].text().split()) for i in u['source_node_ids'] if ' '.join(nodes[i].text().split()))
                self.assertEqual(u['text'],expected)
            generated=make_chunks(d)
            self.assertEqual(generated,[c for c in self.chunks if c['source_id']==identifier])
            for c in generated:
                self.assertTrue(c['atomic'])
                self.assertEqual(c['domains'],['housing_lease'])
                self.assertTrue(c['required_context_chunk_ids'])
                self.assertTrue(set(c['required_context_chunk_ids']) <= law_ids)
                for warning in d['warnings']:
                    self.assertIn(warning,c['embedding_text'])
        distribution=docs['easylaw_housing_distribution']
        text=' '.join(u['text'] for u in distribution['units'])
        for phrase in ['배당요구를 하지 않아도','경매개시결정 전에','경매개시결정 후에','서면','종기']:
            self.assertIn(phrase,text)
        self.assertNotIn('토지초과이득세',text)
        self.assertIn('조세법',str(distribution['scope_excluded']))
        self.assertEqual(len(distribution['citation_discrepancies']),2)
        for c in make_chunks(distribution):
            self.assertIn('원문 인용 조문 불일치',c['embedding_text'])
            self.assertIn('제148조 제3호',c['embedding_text'])

    def test_housing_missing_exception_or_block_page_cannot_be_indexed(self):
        from src.Scripts.collect_consumer_sources import decode_html
        d=next(d for d in self.documents if d['source_id']=='easylaw_housing_distribution')
        html=decode_html((ROOT/d['raw_file']).read_bytes())
        html=re.sub(r'(?i)(charset\s*=\s*[\"\']?)[\w-]+',r'\1utf-8',html)
        for damaged in [html.replace('배당요구를 하지 않아도',''),'blocked']:
            with self.assertRaises(ValueError):
                extract_civil_recovery_guide(d,damaged.encode('utf-8'))

    @classmethod
    def setUpClass(cls):
        cls.report=json.loads((OUTPUT_DIR/'consumer_preprocessing_report.json').read_text(encoding='utf-8'))
        cls.documents=[json.loads((OUTPUT_DIR/e['file']).read_text(encoding='utf-8')) for e in cls.report['saved']]
        cls.chunks=[json.loads(line) for line in (CHUNK_DIR/'chunks.jsonl').read_text(encoding='utf-8').splitlines()]

    def test_installment_law_context_covers_definition_exceptions_notice_remaining_amount_and_threshold(self):
        records=[json.loads(line) for line in (ROOT/'data/chunk/laws/chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        by_id={c['chunk_id']:c for c in records}
        context=installment_context()
        referenced=[by_id[i] for i in context['required_context_chunk_ids']]
        for name,labels in [('할부거래에 관한 법률',{'제2조','제3조','제4조','제16조'}),
                            ('할부거래에 관한 법률 시행령',{'제2조','제4조','제11조'})]:
            self.assertEqual({c['article_label'] for c in referenced if c['law_name']==name},labels)
            self.assertEqual(len({c['document_id'] for c in referenced if c['law_name']==name}),1)
        definition=' '.join(c['text'] for c in referenced if c['law_name']=='할부거래에 관한 법률' and c['article_label']=='제2조')
        self.assertIn('2개월 이상의 기간에 걸쳐 3회 이상',definition)
        self.assertIn('간접할부계약',definition)
        objection=' '.join(c['text'] for c in referenced if c['law_name']=='할부거래에 관한 법률' and c['article_label']=='제16조')
        for phrase in ['의사를 통지한 후','지급하지 아니한 나머지 할부금','서면을 발송한 날','5영업일','7영업일']:
            self.assertIn(phrase,objection)
        threshold=' '.join(c['text'] for c in referenced if c['law_name']=='할부거래에 관한 법률 시행령' and c['article_label']=='제11조')
        self.assertIn('10만원',threshold)
        self.assertIn('20만원',threshold)
        for doc in [d for d in self.documents if d['source_id'] in {'kca_fitness_closure_gym','kca_fitness_closure_pilates'}]:
            self.assertEqual(doc['legal_basis'],context)
            for c in [c for c in make_chunks(doc) if c['role'] in {'payment_and_evidence','card_objection','closure_action','closure_evidence'}]:
                self.assertEqual(c['required_context_chunk_ids'],context['required_context_chunk_ids'])
                self.assertIn('2개월 이상에 걸쳐 3회 이상',c['embedding_text'])
                self.assertIn('2024/2025 안내 당시 구법',c['embedding_text'])
                self.assertIn('3개월',c['text'])

    def test_civil_recovery_conditions_original_nodes_dates_and_update_notices_are_preserved(self):
        from src.Scripts.collect_consumer_sources import HTMLTree, decode_html
        docs={d['source_id']:d for d in self.documents}
        for identifier,scope in CIVIL_RECOVERY_GUIDES.items():
            doc=docs[identifier]
            tree=HTMLTree(decode_html((ROOT/doc['raw_file']).read_bytes())).root
            nodes={n.attrs['id']:n for n in tree.find(lambda n:bool(n.attrs.get('id')))}
            self.assertEqual(doc['information_as_of'],'2026-09-15')
            self.assertIsNone(doc['effective_date'])
            self.assertEqual(len(doc['units']),len(scope))
            self.assertTrue(doc['embedding_candidate'])
            self.assertEqual(preprocess(doc)['units'],doc['units'])
            for u in doc['units']:
                expected='\n'.join(' '.join(nodes[i].text().split()) for i in u['source_node_ids']
                                   if ' '.join(nodes[i].text().split()))
                self.assertEqual(u['text'],expected)
            chunks=make_chunks(doc)
            self.assertEqual(chunks,[c for c in self.chunks if c['source_id']==identifier])
            for c in chunks:
                self.assertTrue(c['atomic'])
                self.assertEqual(c['information_as_of'],doc['information_as_of'])
                self.assertIn('실제 회수를 보장하지',c['embedding_text'])
                self.assertIn('fitness_refund',c['domains'])
            if doc['source_update_notices']:
                for c in chunks:
                    self.assertIn(doc['source_update_notices'][0],c['embedding_text'])
        self.assertTrue(docs['easylaw_payment_order_application']['source_update_notices'])
        execution=next(u for u in docs['easylaw_civil_enforcement']['units'] if u['unit_id']=='execution_copy')
        for phrase in ['집행문을 부여받지 않아도','다만','승계','조건']:
            self.assertIn(phrase,execution['text'])

    def test_civil_recovery_missing_condition_duplicate_heading_and_block_page_fail_closed(self):
        from src.Scripts.collect_consumer_sources import decode_html
        docs={d['source_id']:d for d in self.documents}
        for identifier,phrase in [('easylaw_consumer_small_claim','분할'),
                                  ('easylaw_consumer_payment_order','이의신청한 범위'),
                                  ('easylaw_payment_order_application','공시송달'),
                                  ('easylaw_civil_enforcement','집행문을 부여받지 않아도')]:
            doc=docs[identifier]
            html=decode_html((ROOT/doc['raw_file']).read_bytes())
            html=re.sub(r'(?i)(charset\s*=\s*[\"\']?)[\w-]+',r'\1utf-8',html)
            self.assertIn(phrase,html)
            for damaged in [html.replace(phrase,''),'<h2>Web firewall blocked</h2>',
                            html.replace('class="ovDivbox"','class="missing"')]:
                with self.subTest(identifier=identifier),self.assertRaises(ValueError):
                    extract_civil_recovery_guide(doc,damaged.encode('utf-8'))
        doc=docs['easylaw_consumer_small_claim']
        html=decode_html((ROOT/doc['raw_file']).read_bytes())
        html=html.replace('class="ovDivbox"','class="ovDivbox"><div class="plv2a">소액사건심판의 대상</div><div')
        with self.assertRaises(ValueError):
            extract_civil_recovery_guide(doc,html.encode('utf-8'))

    def test_civil_collection_rejects_success_status_with_blocked_html(self):
        url='https://www.easylaw.go.kr/CSP/CnpClsMain.laf'
        response=SimpleNamespace(url=url,content=b'<h2>Blocked</h2>',raise_for_status=lambda:None)
        with self.assertRaises(ValueError):
            fetch(SimpleNamespace(get=lambda *a,**k:response),url)

    def test_closure_guides_link_civil_recovery_separately_from_installment_objection(self):
        by_id={c['chunk_id']:c for c in self.chunks}
        expected={identifier+'_'+key+'_0000' for identifier,scope in CIVIL_RECOVERY_GUIDES.items() for key,_,_ in scope}
        for identifier in ['kca_fitness_closure_gym_closure_risk_0000','kca_fitness_closure_pilates_consultation_0000']:
            chunk=by_id[identifier]
            self.assertEqual(set(chunk['required_context_chunk_ids']),expected)
            self.assertTrue(expected <= by_id.keys())
            self.assertIn('현금·일시불·기지급',chunk['embedding_text'])
        self.assertFalse(expected & set(by_id['kca_fitness_closure_pilates_card_objection_0000']['required_context_chunk_ids']))

    def test_installment_link_rejects_missing_article_and_mixed_version(self):
        data=(ROOT/'data/chunk/laws/chunks.jsonl').read_text(encoding='utf-8')
        records=[json.loads(line) for line in data.splitlines() if line.strip()]
        missing=[c for c in records if not (c.get('law_name')=='할부거래에 관한 법률 시행령' and c.get('article_label')=='제11조')]
        mixed=copy.deepcopy(records)
        next(c for c in mixed if c.get('law_name')=='할부거래에 관한 법률' and c.get('article_label')=='제16조')['document_id']='law:other_version'
        for damaged in [missing,mixed]:
            encoded='\n'.join(json.dumps(c,ensure_ascii=False) for c in damaged)
            with patch('src.Scripts.preprocess_consumer_sources.Path.read_text',return_value=encoded),self.assertRaises(ValueError):
                installment_context()

    def test_fitness_closure_guides_preserve_conditions_evidence_limits_and_dates(self):
        docs={d['source_id']:d for d in self.documents}
        gym=docs['kca_fitness_closure_gym']
        pilates=docs['kca_fitness_closure_pilates']
        redress=docs['kca_damage_redress_procedure']
        self.assertEqual(gym['publication_date'],'2024-12-11')
        self.assertEqual(pilates['publication_period'],'2025-05')
        self.assertIsNone(pilates['publication_date'])
        text=' '.join(u['text'] for u in pilates['units'])
        for phrase in ['잔여할부금','3개월','20만 원 이상','잔여 횟수','계약해지일','내용증명','1372']:
            self.assertIn(phrase,text)
        units={u['unit_id']:u for u in redress['units']}
        for phrase in ['폐업','연락이 불가능','소재파악']:
            self.assertIn(phrase,units['redress_exclusions']['text'])
        for phrase in ['먼저 상담','30일','90일','토·일요일','합의권고','소비자분쟁조정위원회']:
            self.assertIn(phrase,units['redress_procedure']['text'])
        for doc in [gym,pilates,redress]:
            self.assertTrue(doc['embedding_candidate'])
            self.assertTrue(doc['scope_excluded'])
            self.assertIsNone(doc['effective_date'])
            self.assertEqual(preprocess(doc)['units'],doc['units'])
            self.assertEqual(make_chunks(doc),[c for c in self.chunks if c['source_id']==doc['source_id']])
            self.assertNotIn('티몬',' '.join(u['text'] for u in doc['units']))
            self.assertNotIn('공지사항',' '.join(u['text'] for u in doc['units']))

    def test_fitness_closure_missing_card_conditions_and_redress_exclusion_fail_closed(self):
        from src.Scripts.collect_consumer_sources import decode_html
        report=json.loads((OUTPUT_DIR/'consumer_collection_report.json').read_text(encoding='utf-8'))
        for source in [s for s in report['saved'] if s['source_id'] in {'kca_fitness_closure_gym','kca_fitness_closure_pilates','kca_damage_redress_procedure'}]:
            html=decode_html((ROOT/source['raw_file']).read_bytes())
            html=re.sub(r'(?i)(charset\s*=\s*[\"\']?)[\w-]+',r'\1utf-8',html)
            phrase={'kca_fitness_closure_gym':'3개월 이상','kca_fitness_closure_pilates':'잔여할부금','kca_damage_redress_procedure':'소재파악'}[source['source_id']]
            self.assertIn(phrase,html)
            with self.subTest(source_id=source['source_id']),self.assertRaises(ValueError):
                extract_fitness_closure_guide(source,html.replace(phrase,'').encode('utf-8'))

    def test_redress_limit_is_linked_and_embedded_with_card_caveat(self):
        by_id={c['chunk_id']:c for c in self.chunks}
        c=by_id['kca_damage_redress_procedure_redress_procedure_0000']
        self.assertEqual(c['required_context_chunk_ids'],['kca_damage_redress_procedure_redress_exclusions_0000'])
        self.assertTrue(all(i in by_id for i in c['required_context_chunk_ids']))
        self.assertIn('피해구제 제외',c['embedding_text'])
        self.assertIn('환급이나 강제집행을 보장하지 않음',c['embedding_text'])
        card=by_id['kca_fitness_closure_pilates_card_objection_0000']
        self.assertIn('이미 낸 이용료의 자동 환급',card['embedding_text'])
        self.assertIn('일시불·현금',card['embedding_text'])
        table=by_id['kca_fitness_closure_pilates_evidence_table_0000']
        self.assertTrue(table['atomic'])
        self.assertTrue(table['official_table']['rows'])

    def test_overseas_guide_types_table_conditions_and_raw_scope_are_preserved(self):
        docs={d['source_id']:d for d in self.documents}
        definition=docs['kca_overseas_types']
        table=next(u for u in definition['units'] if u['unit_id']=='comparison')['official_table']
        self.assertEqual(table['headers'],['유형','이런 점이 좋아요!','이런 점은 유의하세요!'])
        self.assertEqual([r[0]['text'] for r in table['rows'][1:]],['직접배송','배송대행','구매대행'])
        self.assertEqual(len(table['rows']),4)
        for row in table['rows']:
            self.assertEqual(len(row),3)
            self.assertTrue(all(c['rowspan']==c['colspan']==1 for c in row))
        texts={u['unit_id']:u['text'] for u in definition['units']}
        self.assertIn('쇼핑몰형',texts['type_purchase_agent'])
        self.assertIn('위임형',texts['type_purchase_agent'])
        for kind in ['direct_delivery','forwarding','purchase_agent']:
            self.assertIn('process_'+kind,texts)
        checklist=docs['kca_overseas_checklist']
        self.assertIn('국내소재의 구매대행은 국내법',next(u['text'] for u in checklist['units'] if u['unit_id']=='purchase_agent'))
        self.assertNotIn('token=', ' '.join(u['text'] for u in checklist['units']))
        self.assertNotIn('150달러',' '.join(u['text'] for u in checklist['units']))
        returns=docs['kca_overseas_returns']['units'][0]['text']
        for word in ['제품하자','국내에서 배송대행지까지','단, 배송대행지에서 국내로 오는 과정','보상을 요구','단순변심']:
            self.assertIn(word,returns)
        for identifier in ['kca_overseas_types','kca_overseas_checklist','kca_overseas_returns']:
            doc=docs[identifier]
            self.assertTrue(doc['embedding_candidate'])
            self.assertTrue(doc['scope_excluded'])
            self.assertEqual(preprocess(doc)['units'],doc['units'])
            self.assertEqual(make_chunks(doc),[c for c in self.chunks if c['source_id']==identifier])
            for c in make_chunks(doc):
                self.assertEqual(c['domains'],['shopping_return'])
                self.assertIsNone(c['dates']['effective_date'])
                self.assertIsNone(c['dates']['information_as_of'])
                self.assertNotIn('서비스 접근 대기',c['text'])

    def test_overseas_html_missing_conditions_or_changed_table_fail_closed(self):
        from src.Scripts.collect_consumer_sources import decode_html
        report=json.loads((OUTPUT_DIR/'consumer_collection_report.json').read_text(encoding='utf-8'))
        for source in [s for s in report['saved'] if s['source_id'].startswith('kca_overseas_')]:
            html=decode_html((ROOT/source['raw_file']).read_bytes())
            html=re.sub(r'(?i)(charset\s*=\s*[\"\']?)[\w-]+',r'\1utf-8',html)
            mutations=[html.replace('id="contents"','id="missing-contents"')]
            if source['source_id']=='kca_overseas_types':
                mutations.append(html.replace('이런 점이 좋아요!','changed-header'))
            elif source['source_id']=='kca_overseas_checklist':
                mutations.append(html.replace('국내소재의 구매대행은 국내법',''))
            else:
                mutations.append(html.replace('국내에서 배송대행지까지',''))
            for changed in mutations:
                with self.subTest(source_id=source['source_id']),self.assertRaises(ValueError):
                    extract_overseas_guide(source,changed.encode('utf-8'))

    def test_purchase_agent_caveat_and_related_law_chunks_survive_chunking(self):
        laws=[json.loads(line) for line in (ROOT/'data/chunk/laws/chunks.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        by_id={c['chunk_id']:c for c in laws}
        c=next(c for c in self.chunks if c['chunk_id']=='kca_overseas_checklist_purchase_agent_0000')
        self.assertEqual(c['trade_types'],['purchase_agent'])
        self.assertIn('예외·비용 조건',c['embedding_text'])
        self.assertTrue(c['required_context_chunk_ids'])
        self.assertEqual({by_id[i]['article_label'] for i in c['required_context_chunk_ids']},{'제17조','제18조'})
        self.assertTrue(all(by_id[i]['law_name']=='전자상거래 등에서의 소비자보호에 관한 법률' for i in c['required_context_chunk_ids']))
        table=next(c for c in self.chunks if c['chunk_id']=='kca_overseas_types_comparison_0000')
        self.assertTrue(table['atomic'])
        self.assertEqual(len(table['official_table']['rows']),4)
        self.assertIn('유형: 배송대행',table['text'])

    def test_labor_guides_preserve_procedure_evidence_dates_and_excluded_scope(self):
        docs={d['source_id']:d for d in self.documents}
        procedure=docs['moel_wage_procedure']
        evidence=docs['moel_wage_evidence_1000323154']
        self.assertEqual(len(procedure['units']),5)
        text=' '.join(u['text'] for u in procedure['units'])
        for word in ['요구(진정)', '요구(고소)', '사업장 소재지 관할', '25일', '토요일', '2차', '검찰']:
            self.assertIn(word,text)
        self.assertNotIn('간이대지급금',text)
        self.assertTrue(procedure['scope_excluded'])
        u=evidence['units'][0]
        self.assertEqual(evidence['normalized_answer_text'][u['start']:u['end']],u['text'])
        for word in ['근로계약서','급여자료','근로시간','별도 증빙이 어려운 경우','대질 조사']:
            self.assertIn(word,u['text'])
        self.assertNotIn('지연이자',u['text'])
        self.assertEqual(evidence['information_as_of'],'2026-05-14')
        for doc in [procedure,evidence]:
            self.assertIsNone(doc['effective_date'])
            self.assertTrue(doc['embedding_candidate'])
            regenerated=preprocess(doc)
            self.assertEqual(regenerated['units'],doc['units'])
            chunks=make_chunks(doc)
            self.assertEqual(chunks,[c for c in self.chunks if c.get('source_id')==doc['source_id']])
            for c in chunks:
                self.assertEqual(c['domains'],['labor_wage'])
                self.assertIsNone(c['dates']['effective_date'])
                self.assertEqual(c['dates']['information_as_of'],doc.get('information_as_of'))

    def test_changed_labor_html_fails_instead_of_indexing_navigation_or_partial_evidence(self):
        report=json.loads((OUTPUT_DIR/'consumer_collection_report.json').read_text(encoding='utf-8'))
        for source in [s for s in report['saved'] if s['source_id'].startswith('moel_wage')]:
            from src.Scripts.collect_consumer_sources import decode_html
            html=decode_html((ROOT/source['raw_file']).read_bytes())
            html=re.sub(r'(?i)(charset\s*=\s*[\"\']?)[\w-]+',r'\1utf-8',html)
            mutations=(html.replace('tabpanel01','unknown-panel'),html.replace('처리절차','changed-heading')) if source['source_id']=='moel_wage_procedure' else (html.replace('별도 증빙이 어려운 경우',''),html.replace('답변일','unknown-date'))
            for changed in mutations:
                with self.subTest(source_id=source['source_id']),self.assertRaises(ValueError):
                    extract_labor_guide(source,changed.encode('utf-8').replace(b'charset=euc-kr',b'charset=utf-8'))
        with self.assertRaises(ValueError):
            extract_labor_guide({'source_id':'unknown'},b'<nav>menu</nav>')

    def test_labor_collection_rejects_non_official_redirect(self):
        client=SimpleNamespace(get=lambda *a,**kw:SimpleNamespace(url='https://example.com/private',raise_for_status=lambda:None))
        for url in ['http://labor.moel.go.kr/', 'https://labor.moel.go.kr.example.com/', 'https://example.com/']:
            with self.assertRaises(ValueError):
                fetch(client,url)
        with self.assertRaises(ValueError):
            fetch(client,'https://labor.moel.go.kr/minwonSysInfo/wagesolway.do')

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
        self.assertEqual(report['document_count'],sum(d['embedding_candidate'] and not d['needs_manual_review'] for d in self.documents))
        self.assertEqual(len({c['chunk_id'] for c in self.chunks}),len(self.chunks))
        for e in report['inputs']:
            self.assertEqual(e['processed_sha256'],digest((OUTPUT_DIR/e['file']).read_bytes()))

    def test_notice_collection_rejects_wrong_title_future_date_and_fixed_version(self):
        source={'source_id':'test_notice_pmh','title':'2026년 적용 최저임금 고시',
                'administrative_id':'2100000262710','expected_effective_date':'20260101'}
        for title,identifier,effective in [('다른 고시','2100000262710','20260101'),
                                           (source['title'],'999','20260101'),
                                           (source['title'],'2100000262710','20990101'),
                                           (source['title'],'2100000262710','20270101')]:
            xml=('<행정규칙><행정규칙기본정보><행정규칙명>'+title+'</행정규칙명>'
                 '<행정규칙일련번호>'+identifier+'</행정규칙일련번호><시행일자>'+effective+
                 '</시행일자></행정규칙기본정보></행정규칙>').encode()
            with patch('src.Scripts.collect_consumer_sources.load_dotenv'), patch.dict('os.environ',{'LAW_API_OC':'test_pmh'}), \
                 patch('src.Scripts.collect_consumer_sources.fetch',return_value=SimpleNamespace(content=xml)), \
                 patch('src.Scripts.collect_consumer_sources.preserve') as saving:
                with self.assertRaises(ValueError):collect_admin(object(),source)
                saving.assert_not_called()

    def test_added_notices_preserve_all_clauses_tables_and_pdf_period(self):
        documents={d['source_id']:d for d in self.documents}
        d=documents['continual_contract_refund']
        root=ET.fromstring((ROOT/d['raw_file']).read_bytes())
        original=[n.text for n in root.findall('조문내용') if (n.text or '').strip()]+[n.findtext('별표내용') for n in root.findall('.//별표단위')]
        self.assertEqual([u['text'] for u in d['units']],original)
        self.assertIn('부칙',d['additional_materials'])
        self.assertTrue(all(u['atomic'] for u in d['units']))
        m=documents['minimum_wage_2026']
        self.assertEqual(m['valid_from'],'2026-01-01')
        self.assertEqual(m['valid_until'],'2026-12-31')
        self.assertEqual(digest((ROOT/m['derived_pdf_file']).read_bytes()),m['derived_pdf_sha256'])
        self.assertEqual(m['units'][0]['text'],'\n'.join(p['text'] for p in m['pdf_pages']))
        self.assertIn('10,320원',m['units'][0]['text'])
        self.assertIn('209시간',m['units'][0]['text'])
        self.assertTrue(m['pdf_pages'][0]['tables'])
        c=next(c for c in self.chunks if c.get('source_id')==m['source_id'])
        self.assertEqual(c['valid_until'],m['valid_until'])
        self.assertEqual(c['page_numbers'],[1])
        bad=copy.deepcopy(m)
        bad['expected_effective_date']='20270101'
        with self.assertRaises(ValueError):preprocess(bad)
        bad=copy.deepcopy(m)
        bad['body_pdf_sha256']='invalid'
        with self.assertRaises(ValueError):preprocess(bad)

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
                        self.assertEqual(p['derivation'],'compact_refund_row_with_retrieval_aliases')
                        table=c['refund_tables'][p['table_index']]
                        row=next(r for r in table['rows'] if r['row_index']==p['row_index'])
                        expected='\n'.join([c['text'].splitlines()[0],*row['condition_texts'],row['refund_text']])
                        self.assertEqual(p['search_text'],expected+'\n[검색용 별칭] '+' '.join(p['search_aliases']))
                        self.assertEqual(cells,table['cells'])
                        self.assertTrue({'환불 계산','환급금 계산'}.issubset(p['search_aliases']))
                        for definition in row['common_texts']:
                            self.assertIn(definition,[cell['text'] for cell in cells])
                        self.assertIn(p['text'],c['embedding_text'])
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
