import unittest
from unittest.mock import Mock

from src.graph.mcp_graph.mcp_graph import build_mcp_graph, route_after_mcp_validation
from src.mcp_tools.contracts import serialize_tool_response


class MCPGraphTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.service.health_check.return_value = serialize_tool_response(
            "success",
            healthy=True,
        )
        self.service.get_collection_info.return_value = serialize_tool_response(
            "success",
            collection={"name": "team_documents_openai_dev"},
        )

    def test_runs_four_tools_and_preserves_document_metadata(self):
        search_document = {
            "point_id": "point-1",
            "text": "검색 결과",
            "score": 0.91,
            "metadata": {
                "document_id": "law:1",
                "chunk_id": "law:1:1",
                "source_url": "https://example.com/law/1",
            },
        }
        lookup_document = {
            "point_id": "point-2",
            "text": "정확 조회 결과",
            "metadata": {
                "document_id": "law:1",
                "chunk_id": "law:1:2",
                "source_url": "https://example.com/law/1",
            },
        }
        self.service.search_documents.return_value = serialize_tool_response(
            "success",
            used_query="계약 해지",
            documents=[search_document],
        )
        self.service.lookup_document.return_value = serialize_tool_response(
            "success",
            document_id="law:1",
            documents=[lookup_document],
        )

        result = build_mcp_graph(self.service).invoke(
            {
                "origin_query": "계약을 해지하고 싶어요.",
                "retry_count": 0,
                "answer": "",
            }
        )

        self.assertEqual(result["mcp_health_status"], "success")
        self.assertEqual(result["mcp_collection_status"], "success")
        self.assertEqual(result["mcp_search_documents"], [search_document])
        self.assertEqual(result["mcp_document_id"], "law:1")
        self.assertEqual(result["mcp_lookup_documents"], [lookup_document])
        self.assertEqual(result["documents"], [search_document, lookup_document])
        self.assertTrue(result["is_valid"])
        self.assertFalse(result["mcp_should_end"])
        self.service.search_documents.assert_called_once_with(
            "계약을 해지하고 싶어요.",
            top_k=5,
            score_threshold=None,
            filters=None,
        )
        self.service.lookup_document.assert_called_once_with("law:1", limit=20)

    def test_empty_search_retries_then_ends_without_lookup(self):
        self.service.search_documents.return_value = serialize_tool_response(
            "empty_result",
            used_query="검색 결과 없는 질문",
            documents=[],
        )

        result = build_mcp_graph(self.service).invoke(
            {
                "origin_query": "검색 결과 없는 질문",
                "retry_count": 0,
                "answer": "",
            }
        )

        self.assertFalse(result["is_valid"])
        self.assertTrue(result["mcp_should_end"])
        self.assertEqual(result["retry_count"], 2)
        self.assertEqual(result["documents"], [])
        self.assertEqual(self.service.search_documents.call_count, 2)
        self.service.lookup_document.assert_not_called()

    def test_router_only_selects_next_destination(self):
        self.assertEqual(
            route_after_mcp_validation(
                {"mcp_should_end": True, "is_valid": False}
            ),
            "end",
        )
        self.assertEqual(
            route_after_mcp_validation(
                {"mcp_should_end": False, "is_valid": True}
            ),
            "lookup",
        )
        self.assertEqual(
            route_after_mcp_validation(
                {"mcp_should_end": False, "is_valid": False}
            ),
            "retry",
        )


if __name__ == "__main__":
    unittest.main()
