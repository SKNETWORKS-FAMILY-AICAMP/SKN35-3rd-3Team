"""STEP 2/3 only: frozen RRF pools, real raw-logit reranking, no threshold selection."""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def validate(data):
    seen, queries, groups = set(), set(), {}
    for case in data['cases']:
        cid = case['case_id']
        if cid in seen or case['query'] in queries:
            raise ValueError('Duplicate case/query: ' + cid)
        seen.add(cid)
        queries.add(case['query'])
        group, split = case['question_group_id'], case['evaluation_split']
        if group in groups and groups[group] != split:
            raise ValueError('Group leakage: ' + group)
        groups[group] = split
        docs = {d['document_id']: d for d in case['candidate_documents']}
        labels = {l['document_id']: l['relevance_grade'] for l in case['relevance_labels']}
        if len(docs) != len(case['candidate_documents']) or len(labels) != len(case['relevance_labels']):
            raise ValueError('Duplicate document/label: ' + cid)
        if any(g is not None and (type(g) is not int or g not in (0, 1, 2)) for g in labels.values()):
            raise ValueError('Invalid relevance grade: ' + cid)
        if set(case['relevant_ids']) != {key for key, g in labels.items() if g == 2}:
            raise ValueError('Core evidence/grade mismatch: ' + cid)
        for d in docs.values():
            if not d['text'].strip() or digest(d['text']) != d['text_sha256']:
                raise ValueError('Stored text hash mismatch: ' + d['document_id'])
        for n in (20, 50):
            rows = case['rrf_rankings'][f'top{n}']
            if len(rows) != n or [r['rank'] for r in rows] != list(range(1, n + 1)):
                raise ValueError('Incomplete saved ranking: ' + cid)
            ids = [r['document_id'] for r in rows]
            if len(set(ids)) != n or not set(ids) <= docs.keys() or not set(ids) <= labels.keys():
                raise ValueError('Missing/duplicate ranked document: ' + cid)
    return data['cases']


def ranking_metrics(ids, grades, k):
    top = list(dict.fromkeys(ids))[:k]
    positives = {key for key, grade in grades.items() if grade in (1, 2)}
    judged = all(grades.get(key) is not None for key in top)
    hits = sum(grades.get(key) in (1, 2) for key in top)
    rr = next((1 / rank for rank, key in enumerate(top, 1) if grades.get(key) in (1, 2)), 0.0)
    result = {'Recall': hits / len(positives) if positives else None,
              'Precision': hits / k if judged else None,
              'Hit Rate': float(hits > 0), 'MRR': rr,
              'nDCG': None, 'graded nDCG': None,
              'unjudged_count': sum(grades.get(key) is None for key in top)}
    if positives and judged:
        for name, gain in [('nDCG', lambda g: float(g > 0)), ('graded nDCG', lambda g: 2 ** g - 1)]:
            dcg = sum(gain(grades[key]) / math.log2(rank + 1) for rank, key in enumerate(top, 1))
            ideal = sum(gain(g) / math.log2(rank + 1) for rank, g in enumerate(
                sorted((g for g in grades.values() if g is not None and g > 0), reverse=True)[:k], 1))
            result[name] = dcg / ideal
    return result


def summarize_scores(values):
    if not values:
        return {'count': 0}
    a = np.asarray(values, dtype=float)
    return {'count': len(a), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'std_population': float(a.std()), 'min': float(a.min()), 'max': float(a.max()),
            'quantiles': {str(q): float(np.quantile(a, q)) for q in (.05, .25, .5, .75, .95)}}


def score_distribution(rows):
    # A Top20/50 duplicate contributes once. Validation is reserved for STEP 5.
    unique = {(r['case_id'], r['document_id']): r for r in rows if r['evaluation_split'] == 'development'}
    negatives = [r['rerank_score'] for r in unique.values() if r['relevance_grade'] == 0]
    positives = [r['rerank_score'] for r in unique.values() if r['relevance_grade'] in (1, 2)]
    result = {'scope': 'development_only_unique_case_document_pairs',
              'grade0': summarize_scores(negatives), 'grade1_2': summarize_scores(positives),
              'unjudged_count': sum(r['relevance_grade'] is None for r in unique.values())}
    if negatives and positives:
        lo = max(float(np.quantile(negatives, .05)), float(np.quantile(positives, .05)))
        hi = min(float(np.quantile(negatives, .95)), float(np.quantile(positives, .95)))
        result['central_90_percent_overlap'] = [lo, hi] if lo <= hi else None
        edges = np.histogram_bin_edges(negatives + positives, bins=40)
        a, _ = np.histogram(negatives, bins=edges)
        b, _ = np.histogram(positives, bins=edges)
        result['histogram_overlap_coefficient'] = float(np.minimum(a / a.sum(), b / b.sum()).sum())
        result['histogram_bins'] = edges.tolist()
    return result, negatives, positives


def save_json(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def export(result, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    rows = result['quality_rows']
    measured = {c['case_id']: c['timings'] for c in result['cases']}
    for row in rows:
        row['shared_union_processing_ms'] = measured[row['case_id']]['processing_seconds'] * 1000
        row['shared_union_inference_ms'] = measured[row['case_id']]['inference_seconds'] * 1000
        row['timing_scope'] = 'one measured Top20/50 union call; shared scores, not per-pool timing'
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (report_dir / 'rag_rerank_quality.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    distribution, negative, positive = score_distribution(result['documents'])
    result['score_distribution'] = distribution
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = distribution.get('histogram_bins', 40)
    for values, label, color in [(negative, 'Grade 0', '#e76f51'), (positive, 'Grade 1 / 2', '#277da8')]:
        if values:
            ax.hist(values, bins=bins, weights=np.ones(len(values)) / len(values), alpha=.55, label=label, color=color)
    overlap = distribution.get('central_90_percent_overlap')
    if overlap:
        ax.axvspan(*overlap, alpha=.08, color='black', label='Central 90% interval overlap')
    ax.set(xlabel='Raw reranker logit (max window)', ylabel='Fraction per bin',
           title='Development score distribution; unique question-document pairs')
    ax.legend()
    fig.tight_layout()
    fig.savefig(report_dir / 'rag_rerank_score_distribution.png', dpi=160)
    plt.close(fig)
    planned = result.get('planned_case_count')
    if planned is None:
        planned = len(json.loads(Path(result['dataset_path']).read_text(encoding='utf-8-sig'))['cases'])
    lines = ['# RAG STEP 2·3 평가', '',
             f"실행 상태: `{result['status']}`. 실제 추론 완료 {len(result['cases'])}/{planned}개 질문. running 상태의 결과는 중간 결과다.",
             f"모델: `{result['model']['name']}`. 데이터셋 SHA256: `{result['dataset_sha256']}`.",
             '저장된 실제 RRF 후보를 사용했다. 질문마다 Top20·50 합집합을 한 번 추론하여 두 풀에서 같은 실제 문서 점수를 재사용했다.',
             '품질 평균과 분포 분석은 Development에 한정했다. Validation은 모델 점수만 저장했으며 STEP 4·5용으로 품질 분석·임계값 선택에 사용하지 않았다.',
             '원시 logit에 Identity activation을 사용했다. 기존 rerank.py의 400토큰·50토큰 overlap 분할 및 max-window 집계는 그대로 사용했다. 확률·0~1 정규화가 아니다.',
             '기본 로더의 activation은 Sigmoid이지만 이번 원시 점수는 Identity이다. 후속 단계에서도 JSON에 기록된 점수 산출 설정을 유지해야 한다.', '']
    for n in (20, 50):
        for k in (5, 10, 20):
            lines += [f'## 후보 {n}, K={k}', '',
                      '| 후보 | K | 지표 | RRF | Rerank | 유효 질문 수 |', '|---:|---:|---|---:|---:|---:|']
            selected = [r for r in rows if r['candidate_k'] == n and r['k'] == k and r['evaluation_category'] == 'answerable']
            for metric in ('Recall', 'Precision', 'Hit Rate', 'MRR', 'nDCG', 'graded nDCG'):
                pairs = [(r['rrf_' + metric], r['rerank_' + metric]) for r in selected
                         if r['rrf_' + metric] is not None and r['rerank_' + metric] is not None]
                if pairs:
                    a, b = np.mean(pairs, axis=0)
                    lines.append(f'| {n} | {k} | {metric} | {a:.4f} | {b:.4f} | {len(pairs)} |')
            changes = Counter(r['change'] for r in selected)
            lines.append(f"\n후보 {n}, K={k}: `{dict(changes)}`\n")
    lines += ['', '## 정답 없는 질문', '',
              'Unanswerable과 uncertain은 검색 정답 지표의 전체 평균에서 제외하고 CSV에 별도 범주로 기록했다. 알려진 양성이 없으면 Recall/nDCG는 정의되지 않으며 null이다. 분기 성공률은 임계값을 결정하지 않아 계산하지 않았다.',
              '', '| 질문 | 범주 | 후보 | 최고 실제 점수 | 양성 후보 수 |', '|---|---|---:|---:|---:|']
    for case in result['cases']:
        if case['evaluation_split'] != 'development' or case['answerability'] == 'answerable':
            continue
        for n in (20, 50):
            documents = [r for r in result['documents'] if r['case_id'] == case['case_id'] and r['candidate_k'] == n]
            lines.append(f"| {case['case_id']} | {case['answerability']} | {n} | {max(r['rerank_score'] for r in documents):.4f} | {sum(r['relevance_grade'] in (1,2) for r in documents)} |")
    lines += [
              '', '## 실제 처리 시간', '',
              f"모델 로딩: {result['model']['load_seconds']:.3f}초. 준비 실행: {result['model']['warmup_seconds']:.3f}초.",
              f"질문별 합집합 처리 시간(ms): `{summarize_scores([c['timings']['processing_seconds'] * 1000 for c in result['cases']])}`",
              'Top20 전용 처리 시간은 별도로 측정하지 않았다. Top50 합집합 추론의 점수를 재사용하므로 풀별 시간 비교를 주장하지 않는다.',
              '', '## 점수 분포', '', '```json', json.dumps(distribution, ensure_ascii=False, indent=2), '```',
              '', '![Development score distribution](rag_rerank_score_distribution.png)',
              '', '## 한계 및 후속 전달', '',
              '- AI 잠정 라벨이며 사람 정답 검수가 아니다. 일부 라벨 오류 가능성이 남아 있다.',
              '- Recall 분모는 후보 밖을 포함한 저장된 전체 Grade 1·2 참조 문서다. 코퍼스 전체 정답 목록은 완전하지 않으며 풀 내 Recall로 분모를 축소하지 않았다.',
              '- CSV의 reference_positive_count와 outside_candidate_positive_count로 후보 밖 정답 수를 확인할 수 있다.',
              '- rerank_scores.json에는 양쪽 분할의 원본 질문·후보별 순위·실제 점수·라벨·필수 근거 조건을 저장했다. Top20/50 중복은 분포 분석에서 한 번만 계산했다.',
              '- 표본은 질문 변형을 포함하며 단일 추론 실행의 시간이다. 반복 시간·신뢰구간은 측정하지 않았다.',
              '- STEP 4·5, 임계값 탐색, LangGraph 분기 구현은 수행하지 않았다.', '']
    (report_dir / 'rag_rerank_report.md').write_text('\n'.join(lines), encoding='utf-8')


def run(dataset_path, output, report_dir, batch_size=8, resume=False):
    from src.const.model import create_rerank_model, RERANK_MODEL
    from src.rag.rerank import rerank
    import torch
    raw = dataset_path.read_bytes()
    data = json.loads(raw.decode('utf-8-sig'))
    cases = validate(data)
    previous = json.loads(output.read_text(encoding='utf-8')) if resume and output.exists() else None
    if previous is not None:
        if (previous['dataset_sha256'] != hashlib.sha256(raw).hexdigest()
                or previous['model']['name'] != RERANK_MODEL
                or previous['model']['activation'] != 'Identity (raw logits)'
                or previous['model']['batch_size'] != batch_size):
            raise ValueError('Resume dataset/model/score settings differ')
        if previous['status'] == 'completed':
            export(previous, report_dir)
            save_json(output, previous)
            return previous
    started = perf_counter()
    model = create_rerank_model()
    load_seconds = perf_counter() - started
    # Only this evaluation model instance changes; repository rerank implementation is unchanged.
    previous_activation = str(model.activation_fn)
    model.activation_fn = torch.nn.Identity()
    model.eval()
    result = {'schema_version': 1, 'steps': [2, 3], 'status': 'running',
              'planned_case_count': len(cases),
              'created_at': datetime.now(timezone.utc).isoformat(),
              'dataset_path': str(dataset_path), 'dataset_sha256': hashlib.sha256(raw).hexdigest(),
              'collection': data['collection'], 'collection_snapshot_id': data['collection_snapshot_id'],
              'analysis_split': 'development', 'validation_use': 'inference_export_only_reserved_for_step5',
              'model': {'name': RERANK_MODEL, 'device': str(model.device), 'batch_size': batch_size,
                        'max_seq_length': model.max_seq_length, 'activation': 'Identity (raw logits)',
                        'loader_default_activation': previous_activation, 'score_method': 'max_window',
                        'score_space': 'raw_logits_not_probabilities',
                        'window_tokens': 400, 'overlap_tokens': 50, 'load_seconds': load_seconds},
              'cases': [], 'documents': [], 'quality_rows': []}
    if previous is not None:
        result = previous
        result['model'].setdefault('resume_load_seconds', []).append(load_seconds)
    first = cases[0]['candidate_documents'][0]
    started = perf_counter()
    with torch.inference_mode():
        rerank({'origin_query': cases[0]['query'], 'documents': [first]}, top_k=1, model=model, batch_size=batch_size, verbose=False)
    warmup = perf_counter() - started
    if previous is None:
        result['model']['warmup_seconds'] = warmup
    else:
        result['model'].setdefault('resume_warmup_seconds', []).append(warmup)
    completed_ids = {c['case_id'] for c in result['cases']}
    for index, case in enumerate(cases, 1):
        if case['case_id'] in completed_ids:
            continue
        source = {d['document_id']: d for d in case['candidate_documents']}
        grades = {l['document_id']: l['relevance_grade'] for l in case['relevance_labels']}
        union = list(dict.fromkeys(r['document_id'] for n in (50, 20) for r in case['rrf_rankings'][f'top{n}']))
        documents = [dict(source[key]) for key in union]
        timings = {}
        with torch.inference_mode():
            ranked = rerank({'origin_query': case['query'], 'documents': documents}, top_k=len(documents),
                            model=model, batch_size=batch_size, timings=timings, verbose=False)['documents']
        scores = {d['document_id']: float(d['score']) for d in ranked}
        if set(scores) != set(union) or len(ranked) != len(union) or any(not math.isfinite(s) for s in scores.values()):
            raise ValueError('Real rerank result integrity failure: ' + case['case_id'])
        if any(d['text'] != source[d['document_id']]['text'] for d in ranked):
            raise ValueError('Rerank changed source text')
        result['cases'].append({'case_id': case['case_id'], 'query': case['query'],
                                'evaluation_split': case['evaluation_split'], 'answerability': case['answerability'],
                                'no_answer_reason': case['no_answer_reason'], 'question_group_id': case['question_group_id'],
                                'answer_requirements': case['answer_requirements'], 'reference_relevant_ids': [k for k,g in grades.items() if g in (1,2)],
                                'candidate_pools': {str(n): case['rrf_rankings'][f'top{n}'] for n in (20,50)},
                                'timings': timings, 'unique_inferred_documents': len(union)})
        for n in (20, 50):
            rows = case['rrf_rankings'][f'top{n}']
            ids = [r['document_id'] for r in rows]
            after = sorted(ids, key=lambda key: (-scores[key], ids.index(key)))
            ranks = {key: rank for rank,key in enumerate(after,1)}
            result['documents'].extend({'case_id': case['case_id'], 'document_id': row['document_id'],
                                        'relevance_grade': grades[row['document_id']], 'rerank_score': scores[row['document_id']],
                                        'evaluation_split': case['evaluation_split'], 'candidate_k': n,
                                        'rrf_rank': row['rank'], 'rerank_rank': ranks[row['document_id']],
                                        'text_sha256': source[row['document_id']]['text_sha256']}
                                       for row in rows)
            if case['evaluation_split'] != 'development':
                continue
            for k in (5, 10, 20):
                a, b = ranking_metrics(ids, grades, k), ranking_metrics(after, grades, k)
                delta = b['graded nDCG'] - a['graded nDCG'] if a['graded nDCG'] is not None and b['graded nDCG'] is not None else None
                result['quality_rows'].append({'case_id': case['case_id'], 'query': case['query'],
                    'evaluation_split': case['evaluation_split'], 'question_type': case['question_type'],
                    'evaluation_category': case['answerability'], 'candidate_k': n, 'k': k,
                    'reference_positive_count': sum(g in (1,2) for g in grades.values()),
                    'outside_candidate_positive_count': sum(g in (1,2) and key not in ids for key,g in grades.items()),
                    'change': 'undefined' if delta is None else 'improved' if delta > 1e-12 else 'worsened' if delta < -1e-12 else 'unchanged',
                    **{'rrf_'+key:value for key,value in a.items()}, **{'rerank_'+key:value for key,value in b.items()}})
        save_json(output, result)
        print(f'[{index}/{len(cases)}] {case["case_id"]}: {len(union)} documents, {timings["inference_seconds"]:.2f}s inference', flush=True)
    if dataset_path.read_bytes() != raw:
        raise ValueError('Dataset changed during evaluation')
    result['status'] = 'completed'
    export(result, report_dir)
    save_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=Path, default=ROOT / 'rag_evaluation_cases_list.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'test' / 'rerank' / 'docs' / 'rerank_scores.json')
    parser.add_argument('--report-dir', type=Path, default=ROOT / 'test' / 'rerank' / 'docs')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--resume', action='store_true', help='같은 데이터·모델 설정의 완료 질문은 실제 저장 점수를 재사용')
    args = parser.parse_args()
    run(args.dataset, args.output, args.report_dir, args.batch_size, args.resume)
