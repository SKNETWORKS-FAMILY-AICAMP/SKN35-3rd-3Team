"""MCP Tool 서비스의 공통 오류 응답과 내부 로깅 검증."""

from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from src.mcp_tools.contracts import parse_tool_response
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


if __name__ == "__main__":
    unittest.main()
