"""dev 공용 bootstrap/MCP가 전달하는 입력과 검색기 사이의 계약 검사."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.const.config import CHUNK_DATA_PATH, ENV_PATH, PROJECT_PATH
from src.const.model import create_embedding, create_vector_embedding
from src.const.qdrant_config import QdrantSettings
from src.retriever import DocumentRetriever
from src.vectorstore import VectorSearchRequest, VectorDocument, VectorCollectionInfo


class DevCompatibilityTests(unittest.TestCase):
    def test_common_project_paths_have_one_root(self):
        self.assertEqual(CHUNK_DATA_PATH, PROJECT_PATH / "data" / "chunk")
        self.assertEqual(ENV_PATH, PROJECT_PATH / ".env")

    def test_openai_factory_uses_openai_key_and_model(self):
        environment = {
            "OPENAI_API_KEY": "openai-test-key",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            "src.const.model.load_project_environment"
        ), patch("src.const.model.OpenAIEmbeddings") as constructor:
            create_embedding()

        constructor.assert_called_once_with(
            model="text-embedding-3-small",
            api_key="openai-test-key",
        )

    def test_vector_factory_delegates_to_existing_embedder(self):
        embedding_settings = SimpleNamespace(provider="openai")
        expected = object()
        with patch(
            "src.embedder.create_embedding_provider",
            return_value=expected,
        ) as constructor:
            actual = create_vector_embedding(embedding_settings)

        self.assertIs(actual, expected)
        constructor.assert_called_once_with(embedding_settings)

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

    def test_openai_model_name_comes_from_provider_environment(self):
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_MODEL": "text-embedding-3-small",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = QdrantSettings.from_env()

        self.assertEqual(settings.embedding.provider, "openai")
        self.assertEqual(settings.embedding.model, "text-embedding-3-small")
        self.assertEqual(settings.embedding.dimension, 1536)
        self.assertEqual(
            settings.collection.name,
            "team_documents_openai_dev",
        )

    def test_nvidia_model_name_comes_from_provider_environment(self):
        environment = {
            "NVIDIA_API_KEY": "test-key",
            "NVIDIA_EMBEDDING_MODEL": "nvidia/nemotron-3-embed-1b",
        }
        with patch.dict(os.environ, environment, clear=True):
            settings = QdrantSettings.from_env()

        self.assertEqual(settings.embedding.provider, "nvidia")
        self.assertEqual(settings.embedding.dimension, 2048)
        self.assertEqual(
            settings.collection.name,
            "team_documents_nvidia_dev",
        )

    def test_custom_api_model_requires_separate_collection_contract(self):
        environment = {
            "OPENAI_API_KEY": "test-key",
            "OPENAI_EMBEDDING_MODEL": "different-model",
        }
        with patch.dict(os.environ, environment, clear=True):
            with self.assertRaisesRegex(ValueError, "EMBEDDING_DIMENSION"):
                QdrantSettings.from_env()


if __name__ == "__main__":
    unittest.main()
