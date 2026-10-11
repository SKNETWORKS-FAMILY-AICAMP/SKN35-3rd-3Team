"""기존 법령·안내서·판례와 보강 소비자 자료의 로컬 BM25 비교.

uv run python -m src.Scripts.tests.test_consumer_coverage_pmh
LLM 답변 정확도 또는 벡터 검색 평가가 아니다.
"""
import json
import unittest
from pathlib import Path

from src.Scripts.collect_consumer_sources import ROOT, write_json
from src.Scripts.tests.test_precedent_retrieval_pmh import BM25

SEARCH_OPTIONS={'max_per_document':2,'repeat_penalty':1.0}
SEARCH_POLICY={'evidence_mode':True,**SEARCH_OPTIONS,'claim_weight':0.6,'header_weight':0.5,'search_representation_version':3,
               'query_expansions':[['헬스장','체력단련장'],['환불','환급']],
               'note':'검색 점수용 구간과 용어 확장. 전체 원문·역할·경고를 반환하며 법률상 동등성이나 계산 가능성 판정이 아님.'}
SELECTED_CHUNKS={
    'gym_refund':['ftc_terms_46118_file_5b139f26_terms_article_7_0000','consumer_dispute_standard_industry_fitness_refund_0000'],
    'pilates_before_start':['ftc_terms_47762_file_05d51a47_terms_article_8_0000'],
    'pilates_cancelled_class':['kca_case_1003739114_case_reasoning_0000','kca_case_1003739114_case_reasoning_0001'],
    'pilates_transfer':['kca_case_1003640779_case_reasoning_0000','kca_case_1003640779_case_decision_0000'],
    'parcel_loss':['ftc_terms_11185_file_a321590d_terms_article_22_0000'],
    'shopping_custom':['kca_case_1003164054_case_reasoning_0000'],
    'parcel_compensation_deadline':['ftc_terms_11185_file_a321590d_terms_article_22_0000'],
}

QUESTIONS=[
    {'id':'gym_refund','query':'헬스장 이용권을 중도해지하면 환불은 얼마나 받나요?',
     'expected_ids':['consumer_dispute_standard','ftc_terms_46118_file_5b139f26']},
    {'id':'pilates_before_start','query':'필라테스 등록기간 개시일 이전에 취소하면 위약금과 이용료는 어떻게 환급하나요?',
     'expected_ids':['ftc_terms_47762_file_05d51a47']},
    {'id':'pilates_cancelled_class','query':'그룹필라테스가 폐강됐어요. 잔여 이용료 환급을 받을 수 있나요?',
     'expected_ids':['kca_case_1003739114']},
    {'id':'pilates_transfer','query':'양도받은 필라테스 강습계약을 해지하면 환급받을 수 있나요?',
     'expected_ids':['kca_case_1003640779']},
    {'id':'parcel_loss','query':'택배가 분실됐는데 누가 배상해야 하나요?',
     'expected_ids':['consumer_dispute_standard','kca_parcel_1003817349','kca_case_1002730975','kca_case_1001243565']},
    {'id':'parcel_receipt','query':'택배 운송장에는 30만원으로 적었는데 공기청정기 구매 영수증은 489000원이에요. 분실 배상은 얼마인가요?',
     'expected_ids':['kca_parcel_1003817349']},
    {'id':'half_price_parcel','query':'반값택배 이용 중 제품이 분실됐는데 배상받을 수 있나요?',
     'expected_ids':['kca_case_1003475989']},
    {'id':'shopping_simple_change','query':'인터넷 쇼핑몰에서 옷을 샀는데 단순변심 반품 배송비는 누가 부담하나요?',
     'expected_ids':['guide_383b62bce56bf1ce','law_009318']},
    {'id':'shopping_custom','query':'인터넷에서 산 한복을 맞춤제작이라는 이유로 반품 거절당했어요. 청약철회할 수 있나요?',
     'expected_ids':['kca_case_1003164054']},
    {'id':'parcel_compensation_deadline','query':'택배 분실 손해입증서류 영수증을 제출했어요. 사업자는 배상요청 후 30일 이내에 우선 배상해야 하나요?',
     'expected_ids':['ftc_terms_11185_file_a321590d']},
    {'id':'parcel_absent_receiver','query':'택배 수화인 부재 시 인도확인은 어떻게 하고 고객과 합의한 장소에 보관할 수 있나요?',
     'expected_ids':['ftc_terms_11185_file_a321590d']},
    {'id':'parcel_shipper_duties','query':'택배 송화인의 의무로 운송장에 주소 전화번호 성명 품명 표준가액을 정확하게 작성해야 하나요?',
     'expected_ids':['ftc_terms_11185_file_a321590d']},
]


def normalize(chunk):
    identifier=chunk.get('source_id') or chunk.get('guide_id') or ('law_'+chunk['law_id'] if chunk.get('law_id') else chunk['precedent_id'])
    return {**chunk,'precedent_id':identifier,'text':chunk['text'],
            'embedding_text':chunk.get('embedding_text',chunk['text'])}


def load_chunks(include_consumer=True):
    chunks=[]
    for folder in ['laws','precedents']+(['consumer_sources'] if include_consumer else []):
        path=ROOT/'data/chunk'/folder/'chunks.jsonl'
        if not path.exists():
            raise ValueError('청크 파일 누락: '+str(path))
        chunks.extend(normalize(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line)
    path=ROOT/'data/chunk/guides'
    manifest=json.loads((path/'guide_chunking_report.json').read_text(encoding='utf-8'))
    for entry in manifest['saved']:
        chunks.extend(normalize(c) for c in json.loads((path/entry['file']).read_text(encoding='utf-8'))['chunks'])
    return chunks


def compact(hit):
    return {k:hit.get(k) for k in ['chunk_id','precedent_id','document_id','source_url','source_path','title','role','unit_id','score','raw_score','matched_span','warnings','source_anomaly','calculation_restricted_clause']}


def evaluate():
    base=load_chunks(False)
    added=[normalize(json.loads(line)) for line in (ROOT/'data/chunk/consumer_sources/chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    before,legacy,after=BM25(base),BM25(base+added),BM25(base+added,evidence_mode=True)
    records=[]
    for question in QUESTIONS:
        results={}
        selected=SELECTED_CHUNKS.get(question['id'],[])
        for label,index in [('before',before),('legacy_after',legacy),('after',after)]:
            hits=index.search(question['query'],top_k=5,**(SEARCH_OPTIONS if label=='after' else {}))
            rank=next((i for i,h in enumerate(hits,1) if h['precedent_id'] in question['expected_ids']),None)
            results[label]={'expected_source_hit_at_3':rank is not None and rank<=3,
                            'expected_source_hit_at_5':rank is not None,'rank':rank,
                            'expected_chunk_hit_at_5':any(h['chunk_id'] in selected for h in hits) if selected else None,
                            'results':[compact(h) for h in hits]}
        records.append({**question,'selected_chunk_ids':selected,**results})
    report={'evaluation_type':'global_char_ngram_bm25_before_after_and_search_policy_comparison','search_policy':SEARCH_POLICY,'baseline_chunk_count':len(base),
            'added_chunk_count':len(added),'question_count':len(records),'questions':records,
            'limitations':['수동 질문으로 기대 출처의 검색 여부를 평가. 답변·환불 계산·현행 적용의 정확성은 검증하지 않음.',
                           '기존 UNCOVERED는 판례만 평가. 이번 before는 기존 모든 자료 유형을 사용.',
                           'legacy_after와 after는 같은 전체 데이터로 기존 검색과 개선 검색을 비교. before는 소비자 자료 추가 전 기존 방식.',
                           '선정 청크 진단은 일부 질문만 수행하며 출처 목록과 독립적임. 출처 적중은 답변 근거 적중을 보장하지 않음.',
                           'LLM·임베딩·Qdrant 평가가 아님. 기대 출처는 모든 정답의 목록이 아님.']}
    for label in ['before','legacy_after','after']:
        report[label]={f'expected_source_hit_at_{k}':sum(r[label][f'expected_source_hit_at_{k}'] for r in records)/len(records) for k in [3,5]}
        reviewed=[r for r in records if r[label]['expected_chunk_hit_at_5'] is not None]
        report[label]['selected_chunk_hits']=sum(r[label]['expected_chunk_hit_at_5'] for r in reviewed)
        report[label]['questions_with_selected_chunks']=len(reviewed)
    report['search_changes']={'source_improved':[r['id'] for r in records if not r['legacy_after']['expected_source_hit_at_5'] and r['after']['expected_source_hit_at_5']],
                              'source_regressed':[r['id'] for r in records if r['legacy_after']['expected_source_hit_at_5'] and not r['after']['expected_source_hit_at_5']]}
    path=ROOT/'data/processed/retrieval_evaluation/consumer_coverage_report.json'
    saved=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    frozen=saved.get('preprocessing_baseline') or {
        'added_chunk_count':saved.get('added_chunk_count'),
        'questions':[{ 'id':r['id'], 'selected_chunk_ids':r.get('selected_chunk_ids',[]),
                       'source_hit':r['after']['expected_source_hit_at_5'],
                       'chunk_hit':r['after'].get('expected_chunk_hit_at_5')}
                      for r in saved.get('questions',[])]}
    current={r['id']:r for r in records}
    comparison=[]
    for old in frozen['questions']:
        now=current[old['id']]
        comparison.append({'id':old['id'],'source_before':old['source_hit'],
                           'source_after':now['after']['expected_source_hit_at_5'],
                           'chunk_before':old['chunk_hit'],
                           'chunk_after':any(h['chunk_id'] in old['selected_chunk_ids'] for h in now['after']['results']) if old['selected_chunk_ids'] else None})
    report['preprocessing_baseline']=frozen
    report['preprocessing_comparison']={
        'basis':'수정 전 선정 청크 ID를 고정한 누적 비교. 현재 검색 정책도 변경되어 전처리 단독 효과가 아님',
        'source_before':sum(r['source_before'] for r in comparison),
        'source_after':sum(r['source_after'] for r in comparison),
        'chunk_before':sum(r['chunk_before'] is True for r in comparison),
        'chunk_after':sum(r['chunk_after'] is True for r in comparison),
        'source_regressed':[r['id'] for r in comparison if r['source_before'] and not r['source_after']],
        'chunk_regressed':[r['id'] for r in comparison if r['chunk_before'] and not r['chunk_after']],
        'questions':comparison}
    refinement=saved.get('search_refinement_baseline') or {
        'added_chunk_count':saved.get('added_chunk_count'),'search_policy':saved.get('search_policy'),
        'questions':[{'id':r['id'],'selected_chunk_ids':r.get('selected_chunk_ids',[]),
                      'source_hit':r['after']['expected_source_hit_at_5'],
                      'chunk_hit':r['after'].get('expected_chunk_hit_at_5')} for r in saved.get('questions',[])]}
    refined=[]
    for old in refinement['questions']:
        now=current[old['id']]['after']
        refined.append({'id':old['id'],'source_before':old['source_hit'],
                        'source_after':now['expected_source_hit_at_5'],'chunk_before':old['chunk_hit'],
                        'chunk_after':any(h['chunk_id'] in old['selected_chunk_ids'] for h in now['results']) if old['selected_chunk_ids'] else None})
    report['search_refinement_baseline']=refinement
    report['search_refinement_comparison']={
        'source_before':sum(r['source_before'] for r in refined),
        'source_after':sum(r['source_after'] for r in refined),
        'chunk_before':sum(r['chunk_before'] is True for r in refined),
        'chunk_after':sum(r['chunk_after'] is True for r in refined),
        'source_regressed':[r['id'] for r in refined if r['source_before'] and not r['source_after']],
        'chunk_regressed':[r['id'] for r in refined if r['chunk_before'] and not r['chunk_after']],
        'questions':refined}
    write_json(path,report)
    print(json.dumps({k:report[k] for k in ['question_count','baseline_chunk_count','added_chunk_count','before','after']},ensure_ascii=False))
    return report


class CoverageTests(unittest.TestCase):
    def test_combined_corpus_has_all_source_types_and_unique_ids(self):
        chunks=load_chunks()
        self.assertEqual(len(chunks),len({c['chunk_id'] for c in chunks}))
        types={c.get('document_type') for c in chunks}
        self.assertTrue({'official_guide','law','mediation_case','standard_terms','administrative_rule'}.issubset(types))

    def test_new_sources_match_manually_selected_expected_documents(self):
        ids={c['precedent_id'] for c in load_chunks()}
        for question in QUESTIONS:
            self.assertTrue(ids.intersection(question['expected_ids']),question['id'])

    def test_saved_evaluation_counts_and_boolean_metrics(self):
        path=ROOT/'data/processed/retrieval_evaluation/consumer_coverage_report.json'
        report=json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(report['question_count'],len(QUESTIONS))
        self.assertEqual(report['baseline_chunk_count'],len(load_chunks(False)))
        self.assertEqual(report['added_chunk_count'],len(load_chunks())-len(load_chunks(False)))
        for label in ['before','legacy_after','after']:
            for k in [3,5]:
                key=f'expected_source_hit_at_{k}'
                self.assertEqual(report[label][key],sum(r[label][key] for r in report['questions'])/len(QUESTIONS))

    def test_parcel_search_preserves_pdf_citations_and_excludes_old_terms(self):
        chunks=load_chunks()
        self.assertNotIn('ftc_terms_11125_file_e27d5f98',{c['precedent_id'] for c in chunks})
        index=BM25(chunks,evidence_mode=True)
        for q in [q for q in QUESTIONS if q['id'] in {'parcel_compensation_deadline','parcel_absent_receiver','parcel_shipper_duties'}]:
            hits=[h for h in index.search(q['query'],top_k=5,**SEARCH_OPTIONS) if h['precedent_id']=='ftc_terms_11185_file_a321590d']
            self.assertTrue(hits,q['id'])
            for hit in hits:
                self.assertTrue(hit['page_numbers'])
                self.assertEqual(hit['terms_revision_date'],'2020-06-05')
                self.assertTrue(hit['derived_pdf_sha256'])

    def test_saved_policy_and_selected_chunk_diagnostics_are_independent_of_source_hits(self):
        chunks={c['chunk_id']:c for c in load_chunks()}
        report=json.loads((ROOT/'data/processed/retrieval_evaluation/consumer_coverage_report.json').read_text(encoding='utf-8'))
        self.assertEqual(report['search_policy'],SEARCH_POLICY)
        for row in report['questions']:
            for identifier in row['selected_chunk_ids']:
                self.assertIn(identifier,chunks)
            for label in ['before','legacy_after','after']:
                hits=row[label]['results']
                expected=any(h['chunk_id'] in row['selected_chunk_ids'] for h in hits) if row['selected_chunk_ids'] else None
                self.assertEqual(row[label]['expected_chunk_hit_at_5'],expected)
                for hit in hits:
                    original=chunks[hit['chunk_id']]
                    self.assertEqual(hit['warnings'],original.get('warnings'))
                    self.assertEqual(hit['role'],original.get('role'))
                    if label=='after':
                        span=hit['matched_span']
                        self.assertEqual(span['text'],original['text'][span['start']:span['end']])

    def test_search_refinement_comparison_keeps_selected_ids_and_source_metrics_separate(self):
        report=json.loads((ROOT/'data/processed/retrieval_evaluation/consumer_coverage_report.json').read_text(encoding='utf-8'))
        comparison=report['search_refinement_comparison']
        self.assertEqual(comparison['chunk_after'],sum(r['chunk_after'] is True for r in comparison['questions']))
        self.assertEqual(comparison['source_after'],sum(r['source_after'] for r in comparison['questions']))
        self.assertFalse(comparison['source_regressed'])
        self.assertFalse(comparison['chunk_regressed'])


if __name__=='__main__':
    evaluate()
