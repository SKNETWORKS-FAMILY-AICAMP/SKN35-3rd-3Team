"""Hand-built fixtures only; never actual evaluation artifacts."""
import json
import csv
import math
import tempfile
import unittest
from pathlib import Path
from test.rerank.threshold_evaluation import evaluate, load_inputs, metrics, run

def test_hand_calculated_metrics():
    m = metrics([(True, True)] * 2 + [(False, False)] * 3 + [(False, True)] + [(True, False)] * 2)
    assert [m[k] for k in ('TP', 'TN', 'FP', 'FN')] == [2, 3, 1, 2]
    for k, expected in {'precision': 2 / 3, 'recall': 0.5, 'f1': 4 / 7, 'accuracy': 5 / 8, 'mcc': 4 / math.sqrt(240), 'fpr': 0.25, 'fnr': 0.5, 'error_rate': 3 / 8}.items():
        assert math.isclose(m[k], expected, rel_tol=0.0, abs_tol=1e-07)
    assert metrics([])['accuracy'] is None
    assert metrics([(False, False)])['mcc'] is None

def test_question_vs_document_and_missing_gold():
    cases = {'a': {'answerability': 'answerable'}, 'b': {'answerability': 'unanswerable'}, 'c': {'answerability': 'answerable'}, 'd': {'answerability': 'uncertain'}}
    rows = [{'case_id': c, 'document_id': str(i), 'relevance_grade': g, 'rerank_score': s} for i, (c, g, s) in enumerate([('a', 2, 0.9), ('a', 0, 0.8), ('b', 0, 0.7), ('c', 0, 0.1), ('d', 0, 0.1)])]
    r = evaluate(rows, cases, 0.5, 'higher')
    assert r['document']['FP'] == 2
    assert r['question_pool']['FP'] == 1
    assert r['question_pool']['FN'] == 0
    assert r['question_pool']['n'] == 4
    assert r['question_gold_known_only']['FN'] == 1
    assert r['question_gold_known_only']['n'] == 3
    assert r['question_strata']['gold_missing_from_pool']['n'] == 1

def fixture(root):
    cases, rows = ([], [])
    for split in ('development', 'validation'):
        for suffix, grade, score in [('yes', 2, 4.0), ('no', 0, -2.0)]:
            cid = split + suffix
            cases.append({'case_id': cid, 'evaluation_split': split, 'question_group_id': cid, 'answerability': 'answerable' if grade else 'unanswerable', 'relevance_labels': [{'document_id': 'doc', 'relevance_grade': grade}]})
            rows.append(dict(case_id=cid, document_id='doc', relevance_grade=grade, rerank_score=score, evaluation_split=split, candidate_k=1, rrf_rank=1, rerank_rank=1))
    cp, sp = (root / 'cases.json', root / 'scores.json')
    cp.write_text(json.dumps({'cases': cases}), encoding='utf-8')
    sp.write_text(json.dumps(rows), encoding='utf-8')
    return (cp, sp, cases, rows)

def test_fixed_threshold_and_no_validation_tuning():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cp, sp, _, rows = fixture(root)
        first = run(cp, sp, root / 'out')['results']['1']
        assert first['threshold'] == 4.0
        assert first['validation']['document']['accuracy'] == 1.0
        for r in rows:
            if r['evaluation_split'] == 'validation':
                r['rerank_score'] *= -10
        sp.write_text(json.dumps(rows), encoding='utf-8')
        second = run(cp, sp, root / 'out')['results']['1']
        assert (first['threshold'], first['direction']) == (second['threshold'], second['direction'])
        assert second['validation']['document']['accuracy'] == 0.0
        assert (root / 'out' / 'threshold_comparison.csv').is_file()
        assert second['validation_threshold_frozen']
        with (root / 'out' / 'split_metrics.csv').open(encoding='utf-8-sig', newline='') as f:
            exported = list(csv.DictReader(f))
        assert len(exported) == 6
        assert {r['threshold'] for r in exported} == {'4.0'}
        val_doc = next((r for r in exported if r['step'] == '5' and r['unit'] == 'document'))
        assert [int(val_doc[k]) for k in ('TP', 'TN', 'FP', 'FN')] == [0, 0, 1, 1]
        assert float(val_doc['accuracy']) == 0.0
        assert float(val_doc['error_rate']) == 1.0

def test_lower_score_direction():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cp, sp, _, rows = fixture(root)
        for r in rows:
            r['rerank_score'] *= -1
        sp.write_text(json.dumps(rows), encoding='utf-8')
        r = run(cp, sp, root / 'out')['results']['1']
        assert r['direction'] == 'lower'
        assert r['validation']['document']['accuracy'] == 1.0

def test_input_rejections():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cp, sp, cases, rows = fixture(root)
        for mutated in (rows + [rows[0]], rows[:-1], [{**r, 'rerank_score': float('nan')} for r in rows], [{**r, 'evaluation_split': 'validation'} for r in rows], [{**r, 'relevance_grade': 1} for r in rows]):
            sp.write_text(json.dumps(mutated), encoding='utf-8')
            try:
                load_inputs(cp, sp)
            except ValueError:
                pass
            else:
                raise AssertionError('Expected ValueError')
        sp.write_text(json.dumps(rows), encoding='utf-8')
        cases[-1]['question_group_id'] = cases[0]['question_group_id']
        cp.write_text(json.dumps({'cases': cases}), encoding='utf-8')
        try:
            load_inputs(cp, sp)
        except ValueError:
            pass
        else:
            raise AssertionError('Expected ValueError')


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(func) for name, func in globals().items()
                              if name.startswith('test_') and callable(func))


if __name__ == '__main__':
    unittest.main()
