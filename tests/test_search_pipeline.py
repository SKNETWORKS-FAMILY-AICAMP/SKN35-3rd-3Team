"""서버/키 없이 실제 로컬 Qdrant와 검색 파이프라인을 검증합니다.

여기 문서/벡터는 테스트 전용이며 실제 법률 검색 품질을 대표하지 않습니다.
"""
import math
import unittest
from importlib.util import find_spec
from types import SimpleNamespace
from unittest.mock import patch

from qdrant_client import QdrantClient, models

from src.retrieval_evaluation import EvaluationCase, compare_retrievers, evaluate_ranking
from src.retriever import BM25Retriever, DocumentRetriever, HybridRetriever
from src.rrf import reciprocal_rank_fusion
from src.vector_db.search_bootstrap import load_keyword_documents, create_team_hybrid_retriever
from src.vectorstore import VectorSearchHit

# 공용 DB 코드를 개인 브랜치에 복사하지 않습니다. dev에서는 실제로 검사합니다.
HAS_COMMON_DB = find_spec("src.vector_db.qdrant_store") is not None


def make_fixture():
    from src.vector_db.qdrant_store import QdrantVectorStore
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
    settings = SimpleNamespace(collection=SimpleNamespace(name="fixture", vector_name=None, sparse_vector_name=None, search_mode="dense"))
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


    def test_rrf_duplicates_and_score(self):
        a = VectorSearchHit("a",0.9,"본문",{"document_id":"d","chunk_id":"a"})
        b = VectorSearchHit("b",2.0,"다른 본문",{"document_id":"d","chunk_id":"b"})
        fused = reciprocal_rank_fusion({"dense":[a,a,b],"bm25":[b,a]})
        self.assertEqual(len(fused),2)
        result = next(hit for hit in fused if hit.point_id=="a")
        self.assertAlmostEqual(result.score,1/61+1/62)
        self.assertNotIn("rrf_sources",a.metadata)


    def test_bm25_formula_and_empty(self):
        docs = [VectorSearchHit("a",0,"반환 보증금"),VectorSearchHit("b",0,"택배 분실")]
        hit = BM25Retriever(docs).search_documents("보증금")[0]
        self.assertAlmostEqual(hit.score,math.log1p(1.5/1.5))
        self.assertEqual(BM25Retriever([]).search_documents("질문"),())
        self.assertEqual(BM25Retriever(docs).search_documents("환불"),())


    @unittest.skipUnless(HAS_COMMON_DB, "공용 Qdrant 연동 테스트는 dev에서 실행")
    def test_empty_remote_corpus_stops_before_embedding(self):
        # 빈 서버를 발견하면 임베딩 API를 생성/호출하지 않아야 합니다.
        settings = SimpleNamespace(validate_for_vector_work=lambda: None)
        store = SimpleNamespace(validate_collection_contract=lambda: None, client=SimpleNamespace(close=lambda: None))
        with patch("src.const.qdrant_config.load_qdrant_settings", return_value=settings), \
             patch("src.vector_db.qdrant_store.QdrantVectorStore.from_settings", return_value=store), \
             patch("src.vector_db.search_bootstrap.load_keyword_documents", return_value=()), \
             patch("src.embedder.create_embedding_provider") as embedding:
            with self.assertRaisesRegex(ValueError,"문서가 없습니다"):
                create_team_hybrid_retriever()
            embedding.assert_not_called()


if __name__ == "__main__":
    unittest.main()
