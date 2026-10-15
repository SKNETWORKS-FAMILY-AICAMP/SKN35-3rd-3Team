"""실제 JSON의 공통 20개 질문으로 다섯 단계 평가 및 단계별 보고서 생성."""
import argparse
import json
from pathlib import Path

from evaluation.run import ROOT, main as run_evaluation
from test.rag_test.report_results import refresh_reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=20)
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/evaluation/direct_20_fast_c5')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if args.limit < 1:
        parser.error('--limit은 양수여야 합니다.')
    source = ROOT / 'rag_evaluation_cases_list.json'
    dataset = json.loads(source.read_text(encoding='utf-8-sig'))
    identifiers = [case['case_id'] for case in dataset['cases'][:args.limit]]
    code = run_evaluation(['--stage', 'all', '--stages', 'query_rewriting', 'multi_query',
        'hybrid', 'rrf', 'reranking', '--config', str(Path(__file__).with_name('fast_config.json')),
        '--evaluation-file', str(source), '--allow-ai-reviewed', '--output', str(args.output),
        '--case-ids', *identifiers, *(['--resume'] if args.resume else [])])
    if code:
        return code
    refresh_reports(args.output / 'evaluation_results.json', args.output / 'stages')
    print(f'실제 {len(identifiers)}개 질문의 다섯 단계 보고서 완료: {args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
