"""
Shared call-instrumentation used by both the MCP server/clients and the A2A
agent servers/clients, so message count / payload bytes / latency are
measured identically on every channel instead of ad hoc per call site.

One InstrumentedRecorder lives per process (the MCP server process; each of
the three agent processes; the orchestrator). Every call writes one row
straight to PostgreSQL (instrumentation/db.py) as it completes -- there is
no in-memory accumulation/dump-at-exit step anymore (that used to write
instrumentation/_runs/<run_id>/<owner>.json; PostgreSQL is now the only
sink). `pool` is attached after construction (via set_pool()) because the
recorder is built in sync code before the real serving event loop exists --
see the lifespan hooks in agents/base/a2a_app.py and mcp_server/server.py
for why that matters (asyncpg pools are bound to the loop they're created in).
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from instrumentation import db


def _payload_bytes(payload: Any) -> int:
    if payload is None:
        return 0
    try:
        return len(json.dumps(payload, default=str).encode("utf-8"))
    except (TypeError, ValueError):
        return len(str(payload).encode("utf-8"))


@dataclass
class _PendingRecord:
    payload_out: Any
    ok: bool
    error: Optional[str]


class InstrumentedRecorder:
    def __init__(self, owner: str, run_id: str = "", pool=None):
        self.owner = owner
        self.run_id = run_id
        self.pool = pool

    def set_pool(self, pool) -> None:
        self.pool = pool

    @asynccontextmanager
    async def record(self, channel: str, name: str, payload_in: Any = None):
        """Usage: `async with recorder.record("mcp", "set_servers", {"c": c}) as rec:
        ...; rec.payload_out = result`. `rec.payload_out` defaults to None
        (e.g. for calls where the caller doesn't bother setting it)."""
        t_start = time.monotonic()
        rec = _PendingRecord(payload_out=None, ok=True, error=None)
        try:
            yield rec
        except Exception as exc:  # noqa: BLE001 - record the failure, then re-raise
            rec.ok = False
            rec.error = str(exc)
            raise
        finally:
            await db.insert_call(
                self.pool, self.run_id, self.owner, channel, name, t_start,
                time.monotonic() - t_start, _payload_bytes(payload_in),
                _payload_bytes(rec.payload_out), rec.ok, rec.error,
            )

    async def record_llm(self, **kwargs) -> None:
        """Plain logging call, not a context manager -- LLMReasoner.reason()
        already owns its own try/except control flow around the provider
        call, so it just hands over everything collected once the call (or
        fallback) is settled."""
        await db.insert_llm_call(self.pool, self.run_id, self.owner, **kwargs)
