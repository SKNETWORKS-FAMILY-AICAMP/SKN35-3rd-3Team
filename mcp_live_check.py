import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def payload(result):
    text = "".join(
        item.text
        for item in result.content
        if isinstance(getattr(item, "text", None), str)
    )
    return json.loads(text)
