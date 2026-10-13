"""MCP Tool 서비스의 공통 오류 응답과 내부 로깅 검증."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.mcp_tools.contracts import parse_tool_response, serialize_tool_response
from src.mcp_tools.nodes import create_mcp_tool_nodes
from src.mcp_tools.server import MCPToolService


class MCPToolServiceTests(unittest.TestCase):
    def test_invalid_requests_share_the_same_external_status(self):
        backend = Mock()
        service = MCPToolService(backend)

        responses = (
            service.search_documents(""),
            service.lookup_document(""),
        )

        for response in responses:
            with self.subTest(response=response):
                parsed = parse_tool_response(response)
                self.assertEqual(parsed["status"], "invalid_request")
                self.assertIn("message", parsed["error"])

        backend.search_documents.assert_not_called()
        backend.lookup_document.assert_not_called()

    def test_unexpected_error_is_logged_but_not_exposed(self):
        backend = Mock()
        backend.search_documents.side_effect = RuntimeError(
            "외부 응답에 노출하면 안 되는 내부 정보"
        )
        service = MCPToolService(backend)

        with patch("src.mcp_tools.server.logger.exception") as log_error:
            parsed = parse_tool_response(service.search_documents("질문"))

        self.assertEqual(parsed["status"], "internal_error")
        self.assertNotIn("노출하면 안 되는", str(parsed))
        log_error.assert_called_once_with(
            "검색 처리 중 내부 오류가 발생했습니다."
        )


class MCPToolNodeTests(unittest.TestCase):
    def setUp(self):
        self.service = Mock()
        self.nodes = create_mcp_tool_nodes(self.service)

    def test_creates_one_node_for_each_mcp_tool(self):
        self.assertEqual(
            set(self.nodes),
            {
                "mcp_search_documents",
                "mcp_lookup_document",
                "mcp_health_check",
                "mcp_get_collection_info",
            },
        )

    def test_search_node_prefers_rewritten_query_and_returns_documents(self):
        documents = [{"text": "검색 결과", "document_id": "law:1"}]
        self.service.search_documents.return_value = serialize_tool_response(
            "success",
            used_query="재작성 질문",
            documents=documents,
        )

        result = self.nodes["mcp_search_documents"](
            {
                "origin_query": "원본 질문",
                "rewrite_query": "재작성 질문",
                "mcp_top_k": 3,
            }
        )

        self.assertEqual(result["mcp_status"], "success")
        self.assertEqual(result["documents"], documents)
        self.assertEqual(result["mcp_documents"], documents)
        self.service.search_documents.assert_called_once_with(
            "재작성 질문",
            top_k=3,
            score_threshold=None,
            filters=None,
        )
        self.service.lookup_document.assert_not_called()
        self.service.health_check.assert_not_called()
        self.service.get_collection_info.assert_not_called()

    def test_lookup_node_uses_document_id_and_limit(self):
        documents = [{"text": "정확 조회 결과", "document_id": "case:1"}]
        self.service.lookup_document.return_value = serialize_tool_response(
            "success",
            document_id="case:1",
            documents=documents,
        )

        result = self.nodes["mcp_lookup_document"](
            {"mcp_document_id": "case:1", "mcp_limit": 7}
        )

        self.assertEqual(result["documents"], documents)
        self.service.lookup_document.assert_called_once_with(
            "case:1",
            limit=7,
        )
        self.service.search_documents.assert_not_called()
        self.service.health_check.assert_not_called()
        self.service.get_collection_info.assert_not_called()

    def test_status_nodes_return_normalized_state(self):
        self.service.health_check.return_value = serialize_tool_response(
            "success",
            healthy=True,
        )
        self.service.get_collection_info.return_value = serialize_tool_response(
            "success",
            collection={"name": "team_documents_openai_dev"},
        )

        health_result = self.nodes["mcp_health_check"]({})
        self.service.health_check.assert_called_once_with()
        self.service.search_documents.assert_not_called()
        self.service.lookup_document.assert_not_called()
        self.service.get_collection_info.assert_not_called()

        collection_result = self.nodes["mcp_get_collection_info"]({})

        self.assertTrue(health_result["mcp_healthy"])
        self.assertEqual(
            collection_result["mcp_collection"],
            {"name": "team_documents_openai_dev"},
        )
        self.service.get_collection_info.assert_called_once_with()

    def test_node_converts_invalid_tool_response_to_safe_error(self):
        self.service.search_documents.return_value = "not-json"

        result = self.nodes["mcp_search_documents"](
            {"origin_query": "질문"}
        )

        self.assertEqual(result["mcp_status"], "invalid_request")
        self.assertNotIn("not-json", str(result))


if __name__ == "__main__":
    unittest.main()
