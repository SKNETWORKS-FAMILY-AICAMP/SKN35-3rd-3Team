"""dev 공용 bootstrap/MCP가 전달하는 입력과 검색기 사이의 계약 검사."""
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.const.config import CHUNK_DATA_PATH, ENV_PATH, PROJECT_PATH
from src.const.qdrant_config import QdrantSettings
from src.retriever import DocumentRetriever
from src.scripts import mcp_server, qdrant_ingest, qdrant_setup
from src.vectorstore import VectorSearchRequest, VectorDocument, VectorCollectionInfo


class DevCompatibilityTests(unittest.TestCase):
    def test_default_profile_does_not_depend_on_legacy_model_variable(self):
        from src.const.qdrant_config import get_api_embedding_profile
        self.assertEqual(get_api_embedding_profile("openai").model, "text-embedding-3-small")
        with patch.dict(os.environ, {
            "EMBEDDING_PROVIDER": "openai", "EMBEDDING_MODEL": "text-embedding-3-small",
            "EMBEDDING_DIMENSION": "1536", "QDRANT_COLLECTION": "team_documents_openai_dev",
            "OPENAI_API_KEY": "test-key",
            "EMBEDDING_NORMALIZED": "false",
        }, clear=True):
            settings = QdrantSettings.from_env()
        self.assertEqual(settings.embedding.model, "text-embedding-3-small")
        self.assertFalse(settings.embedding.normalized)

    def test_common_project_paths_have_one_root(self):
        self.assertEqual(CHUNK_DATA_PATH, PROJECT_PATH / "data" / "chunk")
        self.assertEqual(ENV_PATH, PROJECT_PATH / ".env")

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

    def test_openai_profile_uses_default_model_when_model_env_is_missing(self):
        environment = {"OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, environment, clear=True):
            settings = QdrantSettings.from_env()

        self.assertEqual(settings.embedding.model, "text-embedding-3-small")
        self.assertEqual(settings.embedding.dimension, 1536)

    def test_embedding_normalized_uses_profile_default_or_env_override(self):
        default_environment = {"OPENAI_API_KEY": "test-key"}
        with patch.dict(os.environ, default_environment, clear=True):
            default_settings = QdrantSettings.from_env()

        override_environment = {
            "OPENAI_API_KEY": "test-key",
            "EMBEDDING_NORMALIZED": "false",
        }
        with patch.dict(os.environ, override_environment, clear=True):
            override_settings = QdrantSettings.from_env()

        self.assertTrue(default_settings.embedding.normalized)
        self.assertFalse(override_settings.embedding.normalized)

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

    def test_qdrant_ingest_main_passes_explicit_function_arguments(self):
        args = SimpleNamespace(
            command="ingest",
            data_root=Path("data/chunk"),
            input_report=Path("input.md"),
            ingestion_report=Path("ingestion.md"),
            env_file=Path("team.env"),
            embedding_batch_size=32,
            upsert_batch_size=16,
            parallel=3,
            max_retries=4,
        )
        parser = Mock()
        parser.parse_args.return_value = args

        with (
            patch.object(qdrant_ingest, "build_parser", return_value=parser),
            patch.object(qdrant_ingest, "ingest", return_value=0) as ingest,
        ):
            result = qdrant_ingest.main()

        self.assertEqual(result, 0)
        ingest.assert_called_once_with(
            data_root=args.data_root,
            input_report=args.input_report,
            ingestion_report=args.ingestion_report,
            env_file=args.env_file,
            embedding_batch_size=32,
            upsert_batch_size=16,
            parallel=3,
            max_retries=4,
        )

    def test_qdrant_setup_dispatches_command_and_env_file(self):
        env_file = Path("team.env")
        with patch.object(
            qdrant_setup,
            "check_connection",
            return_value=0,
        ) as check_connection:
            result = qdrant_setup.run_command("check", env_file)

        self.assertEqual(result, 0)
        check_connection.assert_called_once_with(env_file)

    def test_mcp_server_runner_keeps_transport_boundary(self):
        env_file = Path("team.env")
        server = Mock()
        with patch.object(
            mcp_server,
            "create_team_mcp_server",
            return_value=server,
        ) as create_server:
            mcp_server.run_server(transport="stdio", env_file=env_file)

        create_server.assert_called_once_with(env_file)
        server.run.assert_called_once_with()

        with self.assertRaises(SystemExit):
            mcp_server.run_server(
                transport="streamable-http",
                host="0.0.0.0",
            )


if __name__ == "__main__":
    unittest.main()
