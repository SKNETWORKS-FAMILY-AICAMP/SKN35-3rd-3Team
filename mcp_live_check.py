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


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "src.scripts.mcp_server", "--transport", "stdio"],
    )
    async with stdio_client(server) as streams:
        async with ClientSession(*streams) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS:", [tool.name for tool in tools.tools])

            health = payload(await session.call_tool("health_check"))
            info = payload(await session.call_tool("get_collection_info"))
            search = payload(await session.call_tool(
                "search_documents",
                {"query": "계약 해지 시 환불 기준은 무엇인가요?", "top_k": 3},
            ))
            document_id = search["documents"][0]["metadata"]["document_id"]
            lookup = payload(await session.call_tool(
                "lookup_document",
                {"document_id": document_id, "limit": 20},
            ))

            print("health_check:", health["status"])
            print("get_collection_info:", info["status"])
            print("search_documents:", search["status"], len(search["documents"]))
            print("lookup_document:", lookup["status"], len(lookup["documents"]))


asyncio.run(main())