"""서버/키 없이 실제 로컬 Qdrant와 검색 파이프라인을 검증합니다.

여기 문서/벡터는 테스트 전용이며 실제 법률 검색 품질을 대표하지 않습니다.
"""
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from src.multi_query import LLMQueryExpander, MultiQueryRetriever
from src.retrieval_evaluation import EvaluationCase, compare_retrievers, evaluate_ranking
from src.retriever import BM25Retriever, DocumentRetriever, HybridRetriever
from src.rrf import reciprocal_rank_fusion
from src.vector_db.qdrant_store import QdrantVectorStore
from src.vector_db.search_bootstrap import load_keyword_documents, create_team_hybrid_retriever
from src.vectorstore import VectorSearchHit


def make_fixture():
    # 실제 API 임베딩이 아닌 2차원 고정 벡터: 연결/계산 검증용입니다.
    client = QdrantClient(location=":memory:")
    client.create_collection("fixture", vectors_config=models.VectorParams(size=2, distance=models.Distance.COSINE))
    texts = ["보증금 반환", "택배 분실", "임대차 종료"]
    vectors = [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]]
    client.upsert("fixture", [
        models.PointStruct(
            id=i+1, vector=vector,
            payload={"text":text, "document_id":f"d{i+1}", "chunk_id":"c1", "page":i+1, "document_type":"law"},
        ) for i,(text,vector) in enumerate(zip(texts,vectors))
    ])
    settings = SimpleNamespace(collection=SimpleNamespace(name="fixture", vector_name=None))
    store = QdrantVectorStore(client, settings)
    def embed_query(query):
        return [0.0,1.0] if "택배" in query else [1.0,0.0]
    dense = DocumentRetriever(SimpleNamespace(embed_query=embed_query),store)
    bm25 = BM25Retriever(load_keyword_documents(store,page_size=1))
    return client, HybridRetriever(dense,bm25)


class SearchTests(unittest.TestCase):
    def test_metrics_hand_calculation(self):
        metrics = evaluate_ranking(["x","a","b","a"],["a","b","c"],k=3)
        self.assertAlmostEqual(metrics["recall"],2/3)
        self.assertAlmostEqual(metrics["precision"],2/3)
        self.assertEqual(metrics["hit"],1)
        self.assertEqual(metrics["rr"],0.5)
        self.assertEqual(evaluate_ranking([],["a"],k=5)["precision"],0)
        with self.assertRaises(ValueError):
            evaluate_ranking(["a"],[],k=5)

    def test_real_local_qdrant_hybrid_and_multi(self):
        client,hybrid = make_fixture()
        self.addCleanup(client.close)
        self.assertEqual(hybrid.dense.search_documents("보증금")[0].point_id,"1")
        self.assertEqual(len(hybrid.keyword.search_documents("보증금")),1)
        results = hybrid.search_documents("보증금",top_k=2,filters={"document_type":"law"})
        self.assertEqual(results[0].point_id,"1")
        self.assertEqual(results[0].metadata["page"],1)
        self.assertEqual(set(results[0].metadata["rrf_sources"]),{"dense","bm25"})
        calls = []
        def expand(query,count):
            calls.append((query,count))
            return [query," ","임대차 종료","임대차 종료","택배 분실"]
        multi = MultiQueryRetriever(hybrid,expand,max_queries=3)
        self.assertEqual(multi.generate_queries(" 보증금 "),("보증금","임대차 종료","택배 분실"))
        hits = multi.search_documents("보증금",top_k=3)
        self.assertEqual(len(hits),3)
        self.assertIn("query_search_sources",hits[0].metadata)
        self.assertEqual(multi.search_documents("보증금",filters={"document_type":"precedent"}),())
        before = len(calls)
        with self.assertRaises(ValueError):
            multi.search_documents("보증금",top_k=0)
        self.assertEqual(len(calls),before)
        one = MultiQueryRetriever(hybrid,lambda *args: self.fail("확장 호출 금지"),max_queries=1)
        self.assertEqual(one.generate_queries("보증금"),("보증금",))

    def test_rrf_duplicates_and_score(self):
        a = VectorSearchHit("a",0.9,"본문",{"document_id":"d","chunk_id":"a"})
        b = VectorSearchHit("b",2.0,"다른 본문",{"document_id":"d","chunk_id":"b"})
        fused = reciprocal_rank_fusion({"dense":[a,a,b],"bm25":[b,a]})
        self.assertEqual(len(fused),2)
        result = next(hit for hit in fused if hit.point_id=="a")
        self.assertAlmostEqual(result.score,1/61+1/62)
        self.assertNotIn("rrf_sources",a.metadata)

    def test_llm_parser(self):
        fence = chr(96) * 3
        llm = SimpleNamespace(invoke=lambda messages:SimpleNamespace(content=fence + 'json\n["검색 표현"]\n' + fence))
        self.assertEqual(LLMQueryExpander(llm)("질문",2),["검색 표현"])
        llm.invoke = lambda messages:SimpleNamespace(content='{"wrong":1}')
        with self.assertRaises(ValueError):
            LLMQueryExpander(llm)("질문",2)

    def test_bm25_formula_and_empty(self):
        docs = [VectorSearchHit("a",0,"반환 보증금"),VectorSearchHit("b",0,"택배 분실")]
        hit = BM25Retriever(docs).search_documents("보증금")[0]
        self.assertAlmostEqual(hit.score,math.log1p(1.5/1.5))
        self.assertEqual(BM25Retriever([]).search_documents("질문"),())
        self.assertEqual(BM25Retriever(docs).search_documents("환불"),())

    def test_compare_all_methods(self):
        client,hybrid = make_fixture()
        self.addCleanup(client.close)
        multi = MultiQueryRetriever(hybrid,lambda query,count:["임대차 종료"] if "보증금" in query else ["택배 분실"])
        report = compare_retrievers(
            {"dense":hybrid.dense,"bm25":hybrid.keyword,"hybrid":hybrid,"multi_query_hybrid":multi},
            [EvaluationCase("보증금 반환",("d1::c1",)),EvaluationCase("택배 분실",("d2::c1",))],k=3,
        )
        self.assertEqual(report["query_count"],2)
        self.assertEqual(set(report["results"]),{"dense","bm25","hybrid","multi_query_hybrid"})
        for result in report["results"].values():
            self.assertEqual(result["recall"],1)
            self.assertEqual(result["hit_rate"],1)
            self.assertAlmostEqual(result["precision"],1/3)

    def test_empty_remote_corpus_stops_before_embedding(self):
        # 빈 서버를 발견하면 임베딩 API를 생성/호출하지 않아야 합니다.
        settings = SimpleNamespace(validate_for_vector_work=lambda: None)
        store = SimpleNamespace(validate_collection_contract=lambda: None, client=SimpleNamespace(close=lambda: None))
        with patch("src.vector_db.search_bootstrap.load_project_env"), \
             patch("src.vector_db.search_bootstrap.load_qdrant_settings", return_value=settings), \
             patch("src.vector_db.search_bootstrap.QdrantVectorStore.from_settings", return_value=store), \
             patch("src.vector_db.search_bootstrap.load_keyword_documents", return_value=()), \
             patch("src.vector_db.search_bootstrap.create_embedding_provider") as embedding:
            with self.assertRaisesRegex(ValueError,"문서가 없습니다"):
                create_team_hybrid_retriever()
            embedding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
