"""기존 법령·안내서·판례와 보강 소비자 자료의 로컬 BM25 비교.

uv run python -m src.Scripts.tests.test_consumer_coverage_pmh
LLM 답변 정확도 또는 벡터 검색 평가가 아니다.
"""
import json
import unittest
from pathlib import Path

from src.Scripts.collect_consumer_sources import ROOT, write_json
from src.Scripts.tests.test_precedent_retrieval_pmh import BM25

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
    return {k:hit.get(k) for k in ['chunk_id','precedent_id','source_url','title','role','unit_id','score']}


def evaluate():
    base=load_chunks(False)
    added=[normalize(json.loads(line)) for line in (ROOT/'data/chunk/consumer_sources/chunks.jsonl').read_text(encoding='utf-8').splitlines()]
    before,after=BM25(base),BM25(base+added)
    records=[]
    for question in QUESTIONS:
        results={}
        for label,index in [('before',before),('after',after)]:
            hits=index.search(question['query'],top_k=5)
            rank=next((i for i,h in enumerate(hits,1) if h['precedent_id'] in question['expected_ids']),None)
            results[label]={'expected_source_hit_at_3':rank is not None and rank<=3,
                            'expected_source_hit_at_5':rank is not None,'rank':rank,
                            'results':[compact(h) for h in hits]}
        records.append({**question,**results})
    report={'evaluation_type':'global_char_ngram_bm25_before_after','baseline_chunk_count':len(base),
            'added_chunk_count':len(added),'question_count':len(records),'questions':records,
            'limitations':['수동 질문으로 기대 출처의 검색 여부를 평가. 답변·환불 계산·현행 적용의 정확성은 검증하지 않음.',
                           '기존 UNCOVERED는 판례만 평가. 이번 before는 기존 모든 자료 유형을 사용.',
                           'LLM·임베딩·Qdrant 평가가 아님. 기대 출처는 모든 정답의 목록이 아님.']}
    for label in ['before','after']:
        report[label]={f'expected_source_hit_at_{k}':sum(r[label][f'expected_source_hit_at_{k}'] for r in records)/len(records) for k in [3,5]}
    write_json(ROOT/'data/processed/retrieval_evaluation/consumer_coverage_report.json',report)
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
        for label in ['before','after']:
            for k in [3,5]:
                key=f'expected_source_hit_at_{k}'
                self.assertEqual(report[label][key],sum(r[label][key] for r in report['questions'])/len(QUESTIONS))

    def test_parcel_search_preserves_pdf_citations_and_excludes_old_terms(self):
        chunks=load_chunks()
        self.assertNotIn('ftc_terms_11125_file_e27d5f98',{c['precedent_id'] for c in chunks})
        index=BM25(chunks)
        for q in [q for q in QUESTIONS if q['id'] in {'parcel_compensation_deadline','parcel_absent_receiver','parcel_shipper_duties'}]:
            hits=[h for h in index.search(q['query'],top_k=5) if h['precedent_id']=='ftc_terms_11185_file_a321590d']
            self.assertTrue(hits,q['id'])
            for hit in hits:
                self.assertTrue(hit['page_numbers'])
                self.assertEqual(hit['terms_revision_date'],'2020-06-05')
                self.assertTrue(hit['derived_pdf_sha256'])


if __name__=='__main__':
    evaluate()
