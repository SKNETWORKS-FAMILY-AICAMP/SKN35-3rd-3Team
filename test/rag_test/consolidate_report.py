"""완료된 실제 측정 자료를 하나의 한글 보고서로 정리합니다."""
from pathlib import Path

from evaluation.artifacts import load_artifact


ROOT = Path(__file__).resolve().parents[2] / 'reports/evaluation'
LABELS = {'dense': 'Dense 검색', 'sparse': 'Sparse 검색', 'hybrid': 'Hybrid 검색',
          'query_rewriting_original': '질문 재작성 전', 'query_rewriting': '질문 재작성 후',
          'single_query': '단일 질문', 'multi_query': 'Multi Query', 'rrf': 'RRF',
          'reranking_before': '리랭킹 전', 'reranking': '리랭킹 후'}
METRICS = ('hit_rate', 'recall', 'precision', 'mrr', 'ndcg')


def table(rows):
    lines = ['| 단계 | K | Hit@K | Recall@K | Precision@K | MRR@K | nDCG@K |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = [LABELS[row['stage']], str(row['k']), *[f'{row[m]:.4f}' for m in METRICS]]
        lines.append('| ' + ' | '.join(values) + ' |')
    return lines


def detailed_main():
    sources = [ROOT / folder / 'evaluation_results.json'
               for folder in ('direct_20_fast_c5', 'direct_20_recall')]
    results = [load_artifact(path) for path in sources]
    cached_path = ROOT / 'reused_rerank50/evaluation_results.json'
    cached = None
    if cached_path.exists():
        import json
        cached = json.loads(cached_path.read_text(encoding='utf-8'))
        if cached['status'] != 'completed' or cached['dataset_sha256'] != results[0]['settings']['dataset_sha256']:
            raise ValueError('재사용 리랭킹 결과의 상태 또는 데이터가 다릅니다.')
        if set(cached['case_ids']) != {r['case_id'] for r in results[0]['rows']}:
            raise ValueError('재사용 리랭킹 결과의 질문 집합이 다릅니다.')
    if any(result['status'] != 'completed' for result in results):
        raise ValueError('완료된 실제 평가 결과만 통합할 수 있습니다.')
    if results[0]['settings']['dataset_sha256'] != results[1]['settings']['dataset_sha256']:
        raise ValueError('서로 다른 정답 데이터의 결과는 통합할 수 없습니다.')
    if {r['case_id'] for r in results[0]['rows']} != {r['case_id'] for r in results[1]['rows']}:
        raise ValueError('같은 질문 집합의 결과만 통합할 수 있습니다.')
    lines = ['# RAG 통합 평가 보고서', '', '## 평가 기준: K=5', '',
             '**프로젝트 초기 제안 기준입니다. 보편적 표준이 아니며, 코드의 자동 판정 기준으로 적용된 것은 아닙니다.** 모든 지표는 높을수록 좋고, 합격은 아래 다섯 기준을 모두 충족하는 것으로 정의합니다.', '',
             '| 지표 | 점수 범위 | 합격 기준 | 목표(우수) |',
             '|---|---|---:|---:|',
             '| Hit@5 | 0~1 | ≥ 0.90 | ≥ 0.95 |',
             '| Recall@5 | 정의상 0~1; 현재 20개 질문의 최대 평균 0.4418 | ≥ 0.3534 | ≥ 0.3976 |',
             '| Precision@5 | 0~1 | ≥ 0.60 | ≥ 0.80 |',
             '| MRR@5 | 0~1 | ≥ 0.70 | ≥ 0.85 |',
             '| nDCG@5 | 0~1 | ≥ 0.75 | ≥ 0.85 |', '',
             '- Recall 합격선은 가능한 최대 평균의 80%, 우수 목표는 90%입니다. 각각 0.4418 × 0.8 및 0.4418 × 0.9를 반올림한 값입니다. 데이터셋이나 K가 바뀌면 다시 계산해야 합니다.',
             '- Precision 합격선 0.60은 상위 5개 중 평균 3개 이상, 우수 목표 0.80은 평균 4개 이상이 관련 근거라는 뜻입니다. 이번에 추가한 초기 목표입니다.',
             '- K=1/3/10/20/50은 보조 진단입니다. 위 K=5 합격선을 그대로 적용하지 않습니다.',
             '- AI 잠정 라벨이므로 기준 충족은 참고 판정이며, 독립적인 사람 검수 완료를 뜻하지 않습니다.', '',
             f"### 현재 리랭킹 후 결과: 후보 {50 if cached else 5}개, K=5", '',
             '| 지표 | 실제 점수 | 합격 기준 | 참고 판정 |',
             '|---|---:|---:|---|']
    final = next(row for row in (cached or results[0])['summary'] if row['stage'] == 'reranking' and row['k'] == 5)
    thresholds = {'hit_rate': 0.90, 'recall': 0.4418 * 0.8, 'precision': 0.60, 'mrr': 0.70, 'ndcg': 0.75}
    names = {'hit_rate': 'Hit@5', 'recall': 'Recall@5', 'precision': 'Precision@5', 'mrr': 'MRR@5', 'ndcg': 'nDCG@5'}
    for metric in METRICS:
        status = '충족' if final[metric] >= thresholds[metric] else '미달'
        lines.append(f'| {names[metric]} | {final[metric]:.4f} | ≥ {thresholds[metric]:.4f} | {status} |')
    passed = sum(final[m] >= thresholds[m] for m in METRICS)
    lines += ['', f'**현재 {passed}개 충족·{5-passed}개 미달입니다.**', '']
    if cached:
        lines += ['## 주 결과: 실제 후보 50개 리랭킹 전후', '',
                  '완료된 70개 실제 추론 결과에서 기존과 동일한 20개 질문을 추출했습니다. 아래 RRF와 리랭킹은 동일한 저장 후보를 사용한 전후 비교입니다. 후보 50개를 실제 모델로 재정렬한 후 최종 5개 또는 10개를 평가한 결과입니다.', '']
        lines += table([r for r in cached['summary'] if r['k'] in (5, 10)])
        lines += ['', '**이 표를 현재 리랭킹 성능 판단의 주 결과로 봅니다.** 기존 후보 5개 실험은 아래에 과거 참고 결과로 보존했습니다.', '',
                  '- 모델: BAAI/bge-reranker-v2-m3, CPU 배치 8, 400토큰 구간·50토큰 겹침, max-window 집계.',
                  '- 저장 점수는 Identity activation의 원시 logit입니다. 앞선 Sigmoid 점수와 임계값을 직접 혼용하지 않습니다.',
                  '- 원본 데이터 파일 해시, 질문, 후보 ID, 관련도 라벨, 근거 본문 해시를 검증했습니다. 저장된 실제 점수를 재사용했으며 모델을 다시 추론하지 않았습니다.',
                  '- 이 후보 풀은 별도 확대 검색 실행의 후보와 다릅니다. 그 실행의 RRF 점수와 이 리랭킹 점수를 전후 비교로 연결하지 않습니다.',
                  '- 20개 요약은 기존 질문 집합의 기술적 비교입니다. 검증 분할 결과를 임계값 선택이나 추가 파라미터 튜닝에 사용하지 않았습니다.', '']
    lines += ['## 과거 참고: 후보 5개의 다섯 단계 비교', '',
              '동일한 실제 질문 20개와 AI 잠정 관련도 라벨을 사용했습니다. 후보 5개로 실제 검색과 리랭킹을 수행했습니다. 측정 1회이며, 원본 측정 값을 그대로 정리했습니다.', '']
    lines += table([row for row in results[0]['summary'] if row['k'] == 5])
    lines += ['', '리랭킹은 MRR@5와 nDCG@5를 개선했습니다. 동일 후보 5개를 모두 유지하므로 Hit@5·Recall@5·Precision@5는 같습니다. Multi Query와 RRF는 현재 구현에서 동일한 융합 결과를 평가합니다.', '',
              '## 후보 확대 검색 비교', '',
              '후보 50개를 실제 검색한 별도 실행입니다. 이 실행 자체에는 리랭킹 결과가 없습니다. 주 결과의 저장 후보와 구성이 다르므로 서로 연결해 전후 비교하지 않습니다.', '']
    lines += table([row for row in results[1]['summary'] if row['stage'] in ('dense', 'sparse', 'hybrid', 'rrf') and row['k'] in (5, 20, 50)])
    lines += ['', '## 평가 방법과 한계', '',
              '- 관련도 0=무관, 1=부분 관련, 2=직접 관련. Hit·Recall·Precision·MRR은 관련도 1 이상을 정답으로 계산합니다. nDCG는 0/1/2 등급을 반영합니다.',
              '- 질문별 점수의 평균입니다. 미검수 청크의 관련성은 unknown이며, 등록된 참고 정답 기준의 평가입니다.',
              '- 후보 50개 리랭킹 결과는 주 결과에 표시했습니다. 임계값·분기 평가는 test/rerank/docs/report.md의 별도 개발/검증 분할 결과를 확인해야 합니다.' if cached else '- 후보 50개 → 리랭킹 → 최종 5개 평가는 아직 없습니다.', '',
              '## 후보 5개 조건의 모든 K 결과', '']
    lines += table(results[0]['summary'])
    lines += ['', '## 후보 50개 검색 조건의 모든 K 결과', '']
    lines += table(results[1]['summary'])
    lines += ['', '## 원본 측정 자료', '', *[f'- `{path.relative_to(ROOT)}`' for path in sources]]
    if cached:
        lines += [f'- `{cached_path.relative_to(ROOT)}`', '- 원본 실제 추론: `test/rerank/docs/rerank_scores.json`', '- 원본 70개 평가: `test/rerank/docs/rag_rerank_report.md`', '- 임계값·분기 평가: `test/rerank/docs/report.md`']
    output = ROOT / 'rag_report.md'
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    for source in sources:
        for name in ('scores.md', 'evaluation_summary.md'):
            source.with_name(name).write_text('# 통합 보고서 안내\n\n평가 결과는 [RAG 통합 평가 보고서](../rag_report.md) 하나로 통합했습니다.\n\n원본 측정 JSON과 CSV는 이 폴더에 보존되어 있습니다.\n', encoding='utf-8')
    return output


def main():
    """네 지표만 포함한 단계별 요약을 기존 통합 보고서에 저장합니다."""
    import json
    from statistics import mean

    search_path = ROOT / 'direct_20_recall/evaluation_results.json'
    rerank_path = ROOT / 'reused_rerank50/evaluation_results.json'
    search = load_artifact(search_path)
    rerank = json.loads(rerank_path.read_text(encoding='utf-8'))
    if search['status'] != 'completed' or rerank['status'] != 'completed':
        raise ValueError('완료된 실제 결과만 표시합니다.')
    if search['settings']['dataset_sha256'] != rerank['dataset_sha256']:
        raise ValueError('정답 데이터가 다릅니다.')
    identifiers = {row['case_id'] for row in search['rows']}
    if identifiers != set(rerank['case_ids']) or len(identifiers) != 20:
        raise ValueError('동일한 20개 질문이 아닙니다.')
    keys = ('hit_rate', 'recall', 'mrr', 'ndcg')
    for result in (search, rerank):
        for summary in result['summary']:
            if summary['k'] not in (5, 10):
                continue
            rows = [row for row in result['rows'] if row['stage'] == summary['stage']
                    and row['k'] == summary['k'] and row['quality_status'] == 'evaluated']
            if len(rows) != 20:
                raise ValueError('평가 질문 수가 다릅니다.')
            for metric in keys:
                if abs(summary[metric] - mean(row[metric] for row in rows)) > 1e-12:
                    raise ValueError('질문별 실제 점수와 요약이 다릅니다.')
    lines = ['# RAG 단계별 평가 결과', '',
             '**질문 20개 · 검색 후보 50개 · 실제 검색/모델 결과 · AI 잠정 정답 라벨.** 요청한 네 지표를 @5와 @10으로 정리했습니다. 저장된 실제 결과를 재사용했으며, 이번 보고서 정리에 새 검색·모델 추론은 없습니다.', '',
             '**비교 범위:** 앞 네 단계는 확대 검색 실행의 결과입니다. 리랭킹은 완료된 70개 실행에서 동일한 20개 질문을 추출했습니다. 리랭킹 전후는 그 실행의 동일한 후보 50개를 비교합니다. 두 실행의 후보가 다르므로 전체를 하나의 연속 실행으로 해석하지 않습니다.', '',
             '## K=5', '',
             '| 단계·조건 | Hit@5 | Recall@5 | MRR@5 | nDCG@5 |',
             '|---|---:|---:|---:|---:|']
    stage_order = ('query_rewriting_original', 'query_rewriting', 'single_query', 'multi_query',
                   'dense', 'sparse', 'hybrid', 'rrf')
    for k in (5, 10):
        if k == 10:
            lines += ['', '## K=10', '', '| 단계·조건 | Hit@10 | Recall@10 | MRR@10 | nDCG@10 |',
                      '|---|---:|---:|---:|---:|']
        for stage in stage_order:
            row = next(r for r in search['summary'] if r['stage'] == stage and r['k'] == k)
            lines.append('| ' + ' | '.join([LABELS[stage], *[f'{row[m]:.4f}' for m in keys]]) + ' |')
        for stage, name in (('rrf', '리랭킹 전: 저장된 RRF50'), ('reranking', '리랭킹 후: 같은 후보 50개')):
            row = next(r for r in rerank['summary'] if r['stage'] == stage and r['k'] == k)
            lines.append('| ' + ' | '.join([name, *[f'{row[m]:.4f}' for m in keys]]) + ' |')
    lines += ['', '## K=5 기준과 현재 리랭킹 후 판정', '',
              '프로젝트 제안 기준입니다. 네 기준을 모두 충족해야 합격으로 봅니다. AI 잠정 라벨 기반의 참고 판정입니다.', '',
              '| 지표 | 점수 범위 | 합격 기준 | 목표(우수) | 실제 점수 | 판정 |',
              '|---|---|---:|---:|---:|---|']
    final = next(r for r in rerank['summary'] if r['stage'] == 'reranking' and r['k'] == 5)
    criteria = [('hit_rate', 'Hit@5', '0~1', .90, .95),
                ('recall', 'Recall@5', '0~1; 현재 최대 평균 0.4418', .3534, .3976),
                ('mrr', 'MRR@5', '0~1', .70, .85), ('ndcg', 'nDCG@5', '0~1', .75, .85)]
    for key, name, extent, target, excellent in criteria:
        status = '충족' if final[key] >= target else '미달'
        lines.append(f'| {name} | {extent} | ≥ {target:.4f} | ≥ {excellent:.4f} | {final[key]:.4f} | {status} |')
    lines += ['', '**동일 후보 리랭킹 전후에는 네 지표가 모두 개선됐습니다.** 다만 Recall@5와 nDCG@5는 제안 합격 기준에 미달합니다.', '',
              '- Multi Query와 RRF는 현재 구현에서 같은 융합 결과를 평가하므로 점수가 같습니다.',
              '- nDCG는 모든 표에서 0/1/2 등급 기준입니다. 원본 70개 보고서의 이진 nDCG와 혼용하지 않았습니다.',
              '- 저장 리랭커 점수는 원시 logit입니다. Sigmoid 점수와 임계값을 직접 혼용하지 않습니다.',
              '- 20개 요약에 포함된 검증 분할 결과는 이번 임계값 선택·튜닝에 사용하지 않았습니다.',
              '- Recall 기준은 현재 20개 질문의 가능한 최대 평균의 80%, 우수 목표는 90%입니다. @10에는 이 기준을 적용하지 않습니다.', '',
              '## 측정 출처', '',
              '- 앞 네 단계: `direct_20_recall/evaluation_results.json`',
              '- 리랭킹 전후 20개 재계산: `reused_rerank50/evaluation_results.json`',
              '- 완료된 실제 추론 원본: `test/rerank/docs/rerank_scores.json`']
    output = ROOT / 'rag_report.md'
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output


if __name__ == '__main__':
    print(main())
