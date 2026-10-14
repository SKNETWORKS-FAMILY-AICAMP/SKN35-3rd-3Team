"""재검색 없이 완료된 실제 측정 결과의 보고서 갱신·단계별 분리."""
import argparse
import csv
import json
from pathlib import Path

from evaluation.report import save_report
from evaluation.artifacts import load_artifact
from evaluation.metadata import load_question_metadata, annotate_records
from evaluation.data import digest, load_cases
from evaluation.metrics import quality
from evaluation.metrics import aggregate


def save_scores(output, rows):
    """검색 품질 다섯 지표를 K별로 표시. 실제 완료 결과의 질문 평균을 사용."""
    output = Path(output)
    scores = [{'stage': row['stage'], 'K': row['k'], 'questions': row['quality_query_count'],
               'Hit@K': row['hit_rate'], 'Recall@K': row['recall'], 'Precision@K': row['precision'],
               'MRR@K': row['mrr'], 'nDCG@K': row['ndcg']} for row in aggregate(rows)]
    if not scores:
        return
    with (output / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(scores[0]))
        writer.writeheader()
        writer.writerows(scores)
    lines = ['# 실제 검색 평가 점수', '',
             'AI 잠정 라벨 기반 참고 평가. 조건과 실행 범위는 evaluation_summary.md를 참고하세요.', '',
             '| Stage | K | 질문 수 | Hit@K | Recall@K | Precision@K | MRR@K | nDCG@K |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in scores:
        values = [row['stage'], str(row['K']), str(row['questions'])]
        values.extend(f'{row[name]:.4f}' if row[name] is not None else '제외'
                      for name in ('Hit@K', 'Recall@K', 'Precision@K', 'MRR@K', 'nDCG@K'))
        lines.append('| ' + ' | '.join(values) + ' |')
    (output / 'scores.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def add_quality_diagnostics(result):
    dataset_path = Path(result['settings']['dataset_path'])
    dataset = json.loads(dataset_path.read_text(encoding='utf-8-sig'))
    if digest(dataset) != result['settings']['dataset_sha256']:
        raise ValueError('측정에 사용한 정답 파일이 변경됐습니다. 같은 라벨로만 진단을 보충할 수 있습니다.')
    _, normalized_cases = load_cases(dataset_path, allow_ai_reviewed=True)
    cases = {c['case_id']: c for c in normalized_cases}
    indexed = {(d['stage'], d['case_id'], d.get('candidate_k'), d['repeat']): d for d in result['details']}
    for row in result['rows']:
        detail = indexed[row['stage'], row['case_id'], row.get('candidate_k'), row['repeat']]
        scores = quality([hit['evaluation_id'] for hit in detail['results']], cases[row['case_id']], row['k'])
        for metric in ('recall', 'precision', 'mrr', 'hit_rate', 'ndcg'):
            if row[metric] is not None and abs(row[metric] - scores[metric]) > 1e-12:
                raise ValueError('기존 지표와 재계산 지표가 다릅니다. 측정값을 임의로 바꾸지 않습니다.')
        row.update(scores)
    return result

GROUPS = {
    'query_rewriting': {'query_rewriting_original', 'query_rewriting'},
    'multi_query': {'single_query', 'multi_query'},
    'hybrid': {'dense', 'sparse', 'hybrid'},
    'rrf': {'single_query', 'rrf'},
    'reranking': {'reranking_before', 'reranking'},
}


def refresh_reports(source, output_root, question_metadata=None):
    source = Path(source)
    result = load_artifact(source)
    if result['status'] != 'completed':
        raise ValueError('단계별 결과 분리는 완료된 실평가만 허용합니다.')
    add_quality_diagnostics(result)
    if question_metadata:
        metadata = load_question_metadata(question_metadata, result['settings']['dataset_sha256'])
        annotate_records(result['rows'], metadata)
        annotate_records(result['details'], metadata)
        result['settings'].update(question_metadata_file=str(question_metadata), question_metadata_sha256=digest(metadata))
    save_report(source.parent, result['rows'], result['details'], result['settings'])
    save_scores(source.parent, result['rows'])
    exported = []
    for name, stages in GROUPS.items():
        rows = [row for row in result['rows'] if row['stage'] in stages]
        if not any(row['stage'] == name or name == 'hybrid' and row['stage'] == 'hybrid' for row in rows):
            continue
        settings = {**result['settings'], 'selected_stages': [name], 'source_actual_result': str(source),
                    'execution_mode': 'stage functions executed in shared actual run; no fresh search during export'}
        details = [detail for detail in result['details'] if detail['stage'] in stages]
        path = Path(output_root) / name
        save_report(path, rows, details, settings)
        save_scores(path, rows)
        exported.append(str(path))
    return exported


def save_single_report(source, output):
    """test/rag_test 실제 단일 실행 결과만 하나의 Markdown에 저장합니다."""
    result = load_artifact(source)
    if result['status'] != 'completed':
        raise ValueError('완료된 실제 평가만 보고서로 만들 수 있습니다.')
    add_quality_diagnostics(result)
    summary = aggregate(result['rows'])
    cases = {row['case_id'] for row in result['rows']}
    names = {'query_rewriting_original': '질문 재작성 전', 'query_rewriting': '질문 재작성 후',
             'single_query': '단일 질문', 'multi_query': 'Multi Query', 'dense': 'Dense 검색',
             'sparse': 'Sparse 검색', 'hybrid': 'Hybrid 검색', 'rrf': 'RRF',
             'reranking_before': '리랭킹 전', 'reranking': '리랭킹 후'}
    settings = result['settings']
    lines = ['# test/rag_test 단계별 실제 평가', '',
             f"**질문 {len(cases)}개 · 후보 {settings['candidate_ks']}개 · 측정 {settings['repeats']}회 · 리랭커 배치 {settings['batch_size']}**", '',
             '`test/rag_test`의 다섯 단계 함수를 공유 실행한 기존 실제 결과만 사용했습니다. `test/rerank`의 별도 70개 실험과 후보 50개 리랭킹 재사용 결과는 포함하지 않습니다. 이번에는 보고서만 재생성했으며 검색·모델 추론을 다시 실행하지 않았습니다.', '',
             '## 평가 기준: K=5', '',
             '프로젝트 제안 기준이며 AI 잠정 라벨 기반 참고 판정입니다. 다섯 기준을 모두 충족해야 합격으로 봅니다.', '',
             '| 지표 | 점수 범위 | 합격 기준 | 목표(우수) | 실제 점수 | 판정 |',
             '|---|---|---:|---:|---:|---|']
    final = next(r for r in summary if r['stage'] == 'reranking' and r['k'] == 5)
    criteria = [('hit_rate', 'Hit@5', '0~1', .90, .95),
                ('recall', 'Recall@5', '0~1; 현재 최대 평균 0.4418', .3534, .3976),
                ('precision', 'Precision@5', '0~1', .60, .80),
                ('mrr', 'MRR@5', '0~1', .70, .85), ('ndcg', 'nDCG@5', '0~1', .75, .85)]
    for metric, name, extent, target, excellent in criteria:
        lines.append(f"| {name} | {extent} | ≥ {target:.4f} | ≥ {excellent:.4f} | {final[metric]:.4f} | {'충족' if final[metric] >= target else '미달'} |")
    lines += ['', 'Recall 기준은 현재 질문 집합의 최대 평균에 대한 80%/90% 목표입니다. 다른 질문 집합·K에는 그대로 적용하지 않습니다.', '']
    for k in (5, 1, 3):
        lines += [f'## K={k} 단계별 결과', '',
                  f'| 단계 | Hit@{k} | Recall@{k} | Precision@{k} | MRR@{k} | nDCG@{k} |',
                  '|---|---:|---:|---:|---:|---:|']
        for row in summary:
            if row['k'] == k:
                lines.append('| ' + ' | '.join([names[row['stage']],
                    *[f'{row[m]:.4f}' for m in ('hit_rate', 'recall', 'precision', 'mrr', 'ndcg')]]) + ' |')
        lines.append('')
    lines += ['## 해석', '',
              '- 리랭킹은 같은 후보 5개를 재정렬했습니다. 따라서 @5의 Hit·Recall·Precision은 전후 동일하며 MRR·nDCG에서 순위 개선을 확인할 수 있습니다.',
              '- Multi Query는 이미 RRF 융합까지 포함하므로 Multi Query와 RRF의 품질 점수는 같습니다. 두 단계의 독립적인 개선 효과로 중복 계산하지 않습니다.',
              '- 후보 5개 실험이라 리랭킹 후 @10 결과는 없습니다.',
              '- 현재 20개는 모두 답변 가능한 질문입니다. 무정답 질문의 분기 성능은 이 평가에 포함되지 않습니다.',
              '- 관련도 0=무관, 1=부분 관련, 2=직접 관련. 1·2점을 정답으로 사용하고 nDCG는 등급을 반영합니다.',
              '- 잠정 AI 라벨과 불완전한 정답 범위의 영향을 받습니다. 사람 검수 완료 결과는 아닙니다.', '',
              '## 원본 실제 측정 자료', '', f'- `{source}`',
              f"- 실제 평가 질문 ID: {', '.join(sorted(cases))}"]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description='완료된 실제 측정 결과를 단계별 CSV/JSON/Markdown으로 분리')
    parser.add_argument('source', type=Path)
    parser.add_argument('--output-root', type=Path, default=Path('reports/evaluation'))
    parser.add_argument('--question-metadata', type=Path)
    parser.add_argument('--single-report', type=Path, help='단계별 파일을 추가 생성하지 않고 하나의 Markdown만 저장')
    args = parser.parse_args(argv)
    if args.single_report:
        print(save_single_report(args.source, args.single_report))
        return
    for path in refresh_reports(args.source, args.output_root, args.question_metadata):
        print(path)


if __name__ == '__main__':
    main()
