"""
Thin async MCP client wrapper used by the agents that act on the simulator
(tactical, strategic). Wraps the streamablehttp_client + ClientSession
lifecycle (connect once, reuse for the episode) and instruments every
call_tool/read_resource through the same InstrumentedRecorder pattern the MCP
server itself uses, so MCP overhead is measured identically on both ends.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from instrumentation.recorder import InstrumentedRecorder


class MCPBridge:
    def __init__(self, url: str, recorder: InstrumentedRecorder):
        self.url = url
        self.recorder = recorder
        self._stack: AsyncExitStack = None
        self.session: ClientSession = None

    async def __aenter__(self) -> "MCPBridge":
        # AsyncExitStack guarantees a clean unwind of whatever was already
        # entered if a later step (e.g. session.initialize()) fails -- a
        # bare manual __aenter__/__aexit__ pairing leaves anyio's
        # streamablehttp_client task group half-open on partial failure
        # (e.g. connecting before the MCP server's port is listening yet),
        # which corrupts later attempts when the orphaned generator is
        # eventually garbage-collected from an unrelated task.
        stack = AsyncExitStack()
        try:
            read, write, _ = await stack.enter_async_context(streamablehttp_client(self.url))
            self.session = await stack.enter_async_context(ClientSession(read, write))
            await self.session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._stack is not None:
            await self._stack.aclose()

    async def call_tool(self, name: str, **kwargs) -> dict:
        async with self.recorder.record("mcp", f"call_tool:{name}", kwargs) as rec:
            result = await self.session.call_tool(name, arguments=kwargs)
            payload = json.loads(result.content[0].text) if result.content else {}
            rec.payload_out = payload
            return payload

    async def read_resource(self, uri: str) -> dict:
        async with self.recorder.record("mcp", f"read_resource:{uri}") as rec:
            result = await self.session.read_resource(uri)
            payload = json.loads(result.contents[0].text) if result.contents else {}
            rec.payload_out = payload
            return payload
