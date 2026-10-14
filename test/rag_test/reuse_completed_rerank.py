"""완료된 실제 모델 점수로 고정된 20개 질문의 지표를 재계산합니다."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from evaluation.artifacts import load_artifact
from evaluation.data import digest, load_cases
from evaluation.metrics import quality


ROOT = Path(__file__).resolve().parents[2]


def main():
    source = ROOT / 'test/rerank/docs/rerank_scores.json'
    dataset_path = ROOT / 'rag_evaluation_cases_list.json'
    raw = dataset_path.read_bytes()
    saved = json.loads(source.read_text(encoding='utf-8'))
    dataset, cases = load_cases(dataset_path, allow_ai_reviewed=True)
    previous = load_artifact(ROOT / 'reports/evaluation/direct_20_recall/evaluation_results.json')
    if saved['status'] != 'completed' or saved['dataset_sha256'] != hashlib.sha256(raw).hexdigest():
        raise ValueError('완료 상태 또는 정답 데이터 해시가 일치하지 않습니다.')
    if previous['settings']['dataset_sha256'] != digest(dataset):
        raise ValueError('기존 20개 평가와 현재 정답 데이터가 다릅니다.')
    if saved['model']['name'] != 'BAAI/bge-reranker-v2-m3' or saved['model']['score_method'] != 'max_window':
        raise ValueError('모델 또는 점수 집계 방식이 다릅니다.')
    indexed = {c['case_id']: c for c in cases}
    originals = {c['case_id']: c for c in dataset['cases']}
    completed = {c['case_id']: c for c in saved['cases']}
    ids = list(dict.fromkeys(row['case_id'] for row in previous['rows']))
    rows = []
    details = []
    for case_id in ids:
        case, stored = indexed[case_id], completed[case_id]
        if stored['query'] != case['query']:
            raise ValueError('질문 본문이 다릅니다: ' + case_id)
        pool = sorted(stored['candidate_pools']['50'], key=lambda item: item['rank'])
        before = [d['document_id'] for d in pool]
        documents = [d for d in saved['documents'] if d['case_id'] == case_id and d['candidate_k'] == 50]
        if len(documents) != len(before) or {d['document_id'] for d in documents} != set(before):
            raise ValueError('저장 점수와 후보 ID가 일치하지 않습니다: ' + case_id)
        source_documents = {d['document_id']: d for d in originals[case_id]['candidate_documents']}
        for document in documents:
            identifier = document['document_id']
            if (not math.isfinite(document['rerank_score'])
                    or document['relevance_grade'] != case['relevance_grades'][identifier]
                    or document['text_sha256'] != digest(source_documents[identifier]['text'])):
                raise ValueError('점수·라벨·본문 해시 검증 실패: ' + case_id)
        after = [d['document_id'] for d in sorted(documents, key=lambda item: item['rerank_rank'])]
        details.append({'case_id': case_id, 'evaluation_split': stored['evaluation_split'],
                        'before_ids': before, 'after_ids': after, 'scores': documents})
        for stage, ranked in (('rrf', before), ('reranking', after)):
            for k in (1, 3, 5, 10, 20):
                rows.append({'case_id': case_id, 'stage': stage, 'k': k, 'candidate_k': 50,
                             'evaluation_split': stored['evaluation_split'], **quality(ranked, case, k)})
    summaries = []
    for stage in ('rrf', 'reranking'):
        for k in (1, 3, 5, 10, 20):
            selected = [r for r in rows if r['stage'] == stage and r['k'] == k and r['quality_status'] == 'evaluated']
            summaries.append({'stage': stage, 'k': k, 'candidate_k': 50, 'query_count': len(ids),
                              'quality_query_count': len(selected),
                              **{m: mean(r[m] for r in selected) for m in ('hit_rate', 'recall', 'precision', 'mrr', 'ndcg')}})
    result = {'status': 'completed', 'execution_mode': 'reused_completed_real_inference_no_new_search',
              'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'dataset_sha256': digest(dataset), 'model': saved['model'], 'case_ids': ids,
              'candidate_scope': 'saved 70-case run RRF50 pools, not the separate direct_20_recall pools',
              'validation_scope': '20-case descriptive summary only; no threshold selection or parameter tuning',
              'summary': summaries, 'rows': rows, 'details': details}
    target = ROOT / 'reports/evaluation/reused_rerank50/evaluation_results.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for row in summaries:
        if row['k'] in (5, 10):
            print(row)
    print(target)


if __name__ == '__main__':
    main()
