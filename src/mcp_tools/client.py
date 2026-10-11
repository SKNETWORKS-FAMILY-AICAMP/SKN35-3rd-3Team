"""MCP Tool의 문자열 JSON 응답을 검증하는 클라이언트 경계.

공식 MCP SDK의 Transport 연결과 Tool 호출은 ``src.mcp_tools.sdk_client``가
담당합니다. 이 모듈은 Transport와 무관하게 문자열 응답 형식을 검증하고
호출부가 사용할 사전으로 변환합니다.
"""

from src.mcp_tools.contracts import parse_tool_response


def decode_tool_response(response: str) -> dict[str, object]:
    """MCP Tool 문자열 응답을 검증된 사전으로 변환합니다."""

    return parse_tool_response(response)
