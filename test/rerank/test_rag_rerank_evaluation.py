import math
import unittest
from test.rerank.rag_rerank_evaluation import ranking_metrics, score_distribution

def test_positives_include_partial_and_outside_pool():
    result = ranking_metrics(['partial', 'negative'], {'direct': 2, 'partial': 1, 'negative': 0}, 2)
    assert result['Recall'] == 0.5
    assert result['Precision'] == 0.5
    assert result['MRR'] == 1
    assert math.isclose(result['graded nDCG'], 1 / (3 + 1 / 1.584962500721156), rel_tol=0.0, abs_tol=1e-07)

def test_unjudged_is_not_zero():
    result = ranking_metrics(['unknown', 'a'], {'a': 2}, 2)
    assert result['Precision'] is None
    assert result['nDCG'] is None
    assert result['unjudged_count'] == 1

def test_no_answer_is_separate():
    result = ranking_metrics(['n'], {'n': 0}, 5)
    assert result['Recall'] is None
    assert result['graded nDCG'] is None
    assert result['Precision'] == 0

def test_distribution_excludes_validation_and_pool_duplicates():
    rows = [{'case_id': 'a', 'document_id': 'x', 'relevance_grade': 0, 'rerank_score': -4, 'evaluation_split': 'development'}, {'case_id': 'a', 'document_id': 'y', 'relevance_grade': 1, 'rerank_score': 3, 'evaluation_split': 'development'}, {'case_id': 'v', 'document_id': 'z', 'relevance_grade': 2, 'rerank_score': 100, 'evaluation_split': 'validation'}]
    dist, negative, positive = score_distribution(rows + rows)
    assert negative == [-4]
    assert positive == [3]
    assert dist['grade1_2']['count'] == 1


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(func) for name, func in globals().items()
                              if name.startswith('test_') and callable(func))


if __name__ == '__main__':
    unittest.main()
