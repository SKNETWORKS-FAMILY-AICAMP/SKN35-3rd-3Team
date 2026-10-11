"""대체 점수 모델로 재정렬/출처 보존/실패 처리를 검증합니다."""
import unittest
from types import SimpleNamespace

from src.rerank import CrossEncoderScorer, RerankingRetriever
from src.retrieval_evaluation import EvaluationCase, compare_retrievers
from src.vectorstore import VectorSearchHit


class RerankTests(unittest.TestCase):
    def setUp(self):
        self.docs = [VectorSearchHit("a",0.04,"A",{"page":1}),VectorSearchHit("b",0.03,"B",{"page":2})]
        self.calls = []
        self.base = SimpleNamespace(search_documents=lambda query,**kwargs:self.calls.append((query,kwargs)) or self.docs)
        self.scorer = SimpleNamespace(score_pairs=lambda query,texts:[0.1,0.9])

    def test_reordering_and_provenance(self):
        r = RerankingRetriever(self.base,self.scorer)
        hits = r.search_documents(" 질문 ",top_k=1,filters={"document_type":"law"})
        self.assertEqual(hits[0].point_id,"b")
        self.assertEqual(hits[0].metadata["page"],2)
        self.assertEqual(hits[0].metadata["before_rerank"]["rank"],2)
        self.assertEqual(hits[0].metadata["before_rerank"]["score"],0.03)
        self.assertEqual(self.calls[0][0],"질문")
        self.assertEqual(self.calls[0][1]["top_k"],10)
        self.assertNotIn("before_rerank",self.docs[1].metadata)

    def test_ties_empty_threshold(self):
        self.scorer.score_pairs = lambda query,texts:[0.5,0.5]
        r = RerankingRetriever(self.base,self.scorer)
        self.assertEqual([hit.point_id for hit in r.search_documents("질문")],["a","b"])
        self.assertEqual(r.search_documents("질문",score_threshold=0.6),())
        self.base.search_documents = lambda *a,**k:[]
        self.scorer.score_pairs = lambda *a:self.fail("빈 후보에 모델 호출")
        self.assertEqual(r.search_documents("질문"),())

    def test_bad_scores_and_input(self):
        r = RerankingRetriever(self.base,self.scorer)
        for scores in ([1], [float("nan"),1], [True,1]):
            self.scorer.score_pairs = lambda query,texts,s=scores:s
            with self.assertRaises(ValueError):
                r.search_documents("질문")
        self.calls.clear()
        with self.assertRaises(ValueError):
            r.search_documents("질문",top_k=0)
        self.assertEqual(self.calls,[])

    def test_model_errors_propagate(self):
        def fail(*args): raise RuntimeError("model failed")
        self.scorer.score_pairs = fail
        with self.assertRaisesRegex(RuntimeError,"model failed"):
            RerankingRetriever(self.base,self.scorer).search_documents("질문")

    def test_cross_encoder_adapter(self):
        captured = []
        model = SimpleNamespace(predict=lambda pairs,**kwargs:captured.append((pairs,kwargs)) or [0.2,0.8])
        self.assertEqual(CrossEncoderScorer(model).score_pairs("Q",["A","B"]),[0.2,0.8])
        self.assertEqual(captured[0][0],[("Q","A"),("Q","B")])
        self.assertEqual(CrossEncoderScorer(model).score_pairs("Q",[]),())

    def test_comparison_reports_before_and_after(self):
        # 모델 품질이 아니라 재정렬한 순위가 평가에 반영되는지 확인합니다.
        report = compare_retrievers(
            {"before":self.base,"after":RerankingRetriever(self.base,self.scorer)},
            [EvaluationCase("질문",("b",))], k=2,
        )
        self.assertEqual(report["results"]["before"]["mrr"],0.5)
        self.assertEqual(report["results"]["after"]["mrr"],1.0)


if __name__ == "__main__":
    unittest.main()
