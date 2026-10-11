"""dev 공용 bootstrap/MCP가 전달하는 입력과 검색기 사이의 계약 검사."""
import unittest
from types import SimpleNamespace

from src.retriever import DocumentRetriever
from src.vectorstore import VectorSearchRequest, VectorDocument, VectorCollectionInfo


class DevCompatibilityTests(unittest.TestCase):
    def test_sparse_request_and_expected_dimension(self):
        requests = []
        store = SimpleNamespace(search=lambda request: requests.append(request) or ())
        dense = SimpleNamespace(embed_query=lambda query: [1.0, 0.0])
        sparse = SimpleNamespace(embed_query=lambda query: SimpleNamespace(indices=[3], values=[0.7]))
        searcher = DocumentRetriever(dense, store, expected_dimension=2, sparse_embedder=sparse)
        searcher.search_documents(" 질문 ")
        self.assertEqual(requests[0].query_text, "질문")
        self.assertEqual(requests[0].sparse_indices, (3,))
        self.assertEqual(requests[0].sparse_values, (0.7,))

    def test_lookup_does_not_embed(self):
        calls = []
        store = SimpleNamespace(lookup=lambda filters, limit: calls.append((filters, limit)) or ())
        embedder = SimpleNamespace(embed_query=lambda query: self.fail("문서 조회는 임베딩 금지"))
        DocumentRetriever(embedder, store, expected_dimension=2).lookup_document(" law:1 ", limit=3)
        self.assertEqual(calls, [({"document_id": "law:1"}, 3)])

    def test_common_response_shapes(self):
        self.assertEqual(VectorDocument("p1", "본문").to_document()["text"], "본문")
        info = VectorCollectionInfo("c", 2, "cosine", 1, "green", search_mode="hybrid", sparse_vector_name="bm25")
        self.assertEqual(info.to_dict()["sparse_vector_name"], "bm25")

    def test_incomplete_sparse_vector_rejected(self):
        with self.assertRaises(ValueError):
            VectorSearchRequest.create([1.0], sparse_indices=[1])


if __name__ == "__main__":
    unittest.main()
