"""Round-trip through the MCP stdio transport: list tools, ingest, decide, inspect."""
import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


async def _roundtrip(tmp_path):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    env = {**os.environ, "REMEMBRO_HOME": str(tmp_path / "space"), "REMEMBRO_EXTRACTOR": "rules", "PYTHONPATH": str(ROOT / "src")}
    params = StdioServerParameters(command=sys.executable, args=["-m", "remembro.mcp_server"], env=env)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            assert {"remembro_ingest", "remembro_decide", "remembro_register", "remembro_inspect", "remembro_forget", "remembro_status", "remembro_dismiss", "remembro_exercise"} <= tools
            r = await session.call_tool("remembro_ingest", {"path": str(ROOT / "fixtures/delegation_policy_v1.md"), "effective_date": "2026-01-01"})
            ingest = json.loads(r.content[0].text)
            assert ingest["document_id"] == "delegation_policy_v1"
            r = await session.call_tool("remembro_decide", {"actor": "Bob Chen", "resource": "operational expenditure", "amount": 70000, "currency": "AUD", "on": "2026-09-15"})
            decision = json.loads(r.content[0].text)
            assert decision["decision"] == "DENY"  # revoked on 12 September
            assert decision["evidence"]
            r = await session.call_tool("remembro_inspect", {"what": "contradictions"})
            assert json.loads(r.content[0].text)["count"] >= 1


def test_mcp_roundtrip(tmp_path):
    asyncio.run(_roundtrip(tmp_path))
