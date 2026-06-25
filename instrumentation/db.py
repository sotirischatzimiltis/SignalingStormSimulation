"""
PostgreSQL persistence -- the source of truth for run config/status, every
MCP/A2A/LLM call, and every storm's evolution record. Replaces the old
per-process JSON dumps (InstrumentedRecorder.dump()), status.json,
report.json, and memory/episode_<run_id>.json entirely; logs/*.log (plain
text, tailed live by the dashboard) are unaffected.

Connection string: STORMSIM_DATABASE_URL env var, defaulting to a dedicated
local container (see scripts/README.md "Database setup") -- deliberately
NOT the default Postgres port 5432, which on this machine already belongs
to an unrelated existing stack.

CRUD functions take `conn` as their first argument and work with either an
asyncpg.Pool or a single asyncpg.Connection interchangeably (both expose
the same execute/fetch/fetchrow interface) -- long-lived server processes
pass a pool (see get_pool()); short-lived scripts pass a single connection
(see connection()).
"""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import asyncpg

DEFAULT_DATABASE_URL = "postgresql://stormsim:stormsim@localhost:5433/stormsim"


def database_url() -> str:
    return os.environ.get("STORMSIM_DATABASE_URL", DEFAULT_DATABASE_URL)


async def _init_codecs(conn: asyncpg.Connection) -> None:
    """asyncpg doesn't auto-serialize Python dicts <-> jsonb -- without this,
    every INSERT/SELECT touching a JSONB column would need explicit
    json.dumps()/json.loads() at every call site instead of plain dicts."""
    await conn.set_type_codec(
        "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog", format="text",
    )


async def get_pool() -> asyncpg.Pool:
    """For long-lived server processes (mcp_server, the 3 agents) -- create
    once inside the actual serving event loop (see the lifespan hooks in
    agents/base/a2a_app.py and mcp_server/server.py), not in sync main()."""
    return await asyncpg.create_pool(database_url(), min_size=1, max_size=5, init=_init_codecs)


@asynccontextmanager
async def connection():
    """For short-lived scripts (run_agentic.py, dashboard.py) -- one
    connection per call site, same shape as the existing fetch_live()
    MCP-bridge pattern."""
    conn = await asyncpg.connect(database_url())
    await _init_codecs(conn)
    try:
        yield conn
    finally:
        await conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at DOUBLE PRECISION NOT NULL,
    updated_at DOUBLE PRECISION NOT NULL,
    phase TEXT NOT NULL,
    scenario TEXT,
    rt_factor DOUBLE PRECISION,
    llm_provider TEXT,
    config JSONB,
    mcp_url TEXT,
    error TEXT
);

CREATE TABLE IF NOT EXISTS calls (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    owner TEXT NOT NULL,
    channel TEXT NOT NULL,
    name TEXT NOT NULL,
    t_start DOUBLE PRECISION NOT NULL,
    latency_s DOUBLE PRECISION NOT NULL,
    bytes_in INTEGER NOT NULL,
    bytes_out INTEGER NOT NULL,
    ok BOOLEAN NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_calls_run_id ON calls(run_id);

CREATE TABLE IF NOT EXISTS llm_calls (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    owner TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    system_prompt TEXT,
    user_prompt TEXT,
    response_text TEXT,
    decision JSONB,
    tokens_in INTEGER,
    tokens_out INTEGER,
    latency_s DOUBLE PRECISION NOT NULL,
    source TEXT NOT NULL,
    t_start DOUBLE PRECISION NOT NULL,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_llm_calls_run_id ON llm_calls(run_id);

CREATE TABLE IF NOT EXISTS storms (
    id SERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    storm_index INTEGER NOT NULL,
    t0 DOUBLE PRECISION NOT NULL,
    td DOUBLE PRECISION NOT NULL,
    resilience JSONB NOT NULL,
    classification TEXT,
    lessons TEXT,
    decision_source TEXT,
    policy_before JSONB,
    policy_after JSONB,
    written_at DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_storms_run_id ON storms(run_id);
"""


async def ensure_schema(conn) -> None:
    await conn.execute(SCHEMA)


# ---------------------------------------------------------------------------
# runs
# ---------------------------------------------------------------------------
async def insert_run(conn, run_id: str, phase: str, config: dict) -> None:
    now = time.time()
    await conn.execute(
        """INSERT INTO runs (run_id, created_at, updated_at, phase, scenario, rt_factor,
               llm_provider, config, mcp_url, error)
           VALUES ($1, $2, $2, $3, $4, $5, $6, $7, NULL, NULL)
           ON CONFLICT (run_id) DO NOTHING""",
        run_id, now, phase, config.get("scenario"), config.get("rt_factor"),
        config.get("llm_provider"), config,
    )


async def update_run(conn, run_id: str, **fields) -> None:
    """Mirrors the old write_status(run_id, **fields) merge-update shape."""
    if not fields:
        return
    cols = list(fields.keys())
    set_clause = ", ".join(f"{c} = ${i + 2}" for i, c in enumerate(cols))
    await conn.execute(
        f"UPDATE runs SET {set_clause}, updated_at = $1 WHERE run_id = ${len(cols) + 2}",
        time.time(), *[fields[c] for c in cols], run_id,
    )


async def get_run(conn, run_id: str) -> Optional[dict]:
    row = await conn.fetchrow("SELECT * FROM runs WHERE run_id = $1", run_id)
    return dict(row) if row else None


async def list_runs(conn) -> list:
    rows = await conn.fetch("SELECT * FROM runs ORDER BY created_at DESC")
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# calls / llm_calls
# ---------------------------------------------------------------------------
async def insert_call(conn, run_id: str, owner: str, channel: str, name: str, t_start: float,
                       latency_s: float, bytes_in: int, bytes_out: int, ok: bool,
                       error: Optional[str]) -> None:
    await conn.execute(
        """INSERT INTO calls (run_id, owner, channel, name, t_start, latency_s,
               bytes_in, bytes_out, ok, error)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)""",
        run_id, owner, channel, name, t_start, latency_s, bytes_in, bytes_out, ok, error,
    )


async def insert_llm_call(conn, run_id: str, owner: str, provider: str, model: str,
                           system_prompt: str, user_prompt: str, response_text: Optional[str],
                           decision: dict, tokens_in: Optional[int], tokens_out: Optional[int],
                           latency_s: float, source: str, t_start: float,
                           error: Optional[str]) -> None:
    await conn.execute(
        """INSERT INTO llm_calls (run_id, owner, provider, model, system_prompt, user_prompt,
               response_text, decision, tokens_in, tokens_out, latency_s, source, t_start, error)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)""",
        run_id, owner, provider, model, system_prompt, user_prompt, response_text,
        decision, tokens_in, tokens_out, latency_s, source, t_start, error,
    )


async def get_calls(conn, run_id: str) -> list:
    rows = await conn.fetch("SELECT * FROM calls WHERE run_id = $1", run_id)
    return [dict(r) for r in rows]


async def get_llm_calls(conn, run_id: str) -> list:
    rows = await conn.fetch("SELECT * FROM llm_calls WHERE run_id = $1 ORDER BY t_start", run_id)
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# storms
# ---------------------------------------------------------------------------
async def insert_storm(conn, run_id: str, storm_index: int, t0: float, td: float,
                        resilience: dict, classification: Optional[str], lessons: Optional[str],
                        decision_source: Optional[str], policy_before: dict,
                        policy_after: dict) -> None:
    await conn.execute(
        """INSERT INTO storms (run_id, storm_index, t0, td, resilience, classification,
               lessons, decision_source, policy_before, policy_after, written_at)
           VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)""",
        run_id, storm_index, t0, td, resilience, classification, lessons,
        decision_source, policy_before, policy_after, time.time(),
    )


async def get_storms(conn, run_id: str) -> list:
    rows = await conn.fetch("SELECT * FROM storms WHERE run_id = $1 ORDER BY storm_index", run_id)
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Overhead report -- same aggregation logic as the old instrumentation/report.py
# build_overhead_report(), just fed DB rows instead of JSON-loaded dicts (these
# helpers don't care which: they operate on plain lists of dicts).
# ---------------------------------------------------------------------------
def _percentile(values: list, p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(p * (len(s) - 1))))
    return s[idx]


def _channel_stats(records: list) -> dict:
    latencies = [r["latency_s"] for r in records]
    bytes_total = sum(r["bytes_in"] + r["bytes_out"] for r in records)
    failures = sum(1 for r in records if not r["ok"])
    return {
        "count": len(records),
        "failures": failures,
        "total_bytes": bytes_total,
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
        "p95_latency_s": _percentile(latencies, 0.95),
        "max_latency_s": max(latencies) if latencies else 0.0,
    }


def _llm_summary(calls: list) -> dict:
    tokens_in = [c["tokens_in"] for c in calls if c.get("tokens_in") is not None]
    tokens_out = [c["tokens_out"] for c in calls if c.get("tokens_out") is not None]
    latencies = [c["latency_s"] for c in calls]
    return {
        "count": len(calls),
        "fallback_count": sum(1 for c in calls if c.get("source") == "fallback"),
        "total_tokens_in": sum(tokens_in),
        "total_tokens_out": sum(tokens_out),
        "mean_latency_s": (sum(latencies) / len(latencies)) if latencies else 0.0,
    }


async def build_overhead_report(conn, run_id: str) -> dict:
    all_records = await get_calls(conn, run_id)
    all_llm_calls = await get_llm_calls(conn, run_id)

    owners = sorted({r["owner"] for r in all_records} | {c["owner"] for c in all_llm_calls})
    per_owner = {}
    for owner in owners:
        records = [r for r in all_records if r["owner"] == owner]
        llm_calls = [c for c in all_llm_calls if c["owner"] == owner]
        per_owner[owner] = {
            "mcp": _channel_stats([r for r in records if r["channel"] == "mcp"]),
            "a2a": _channel_stats([r for r in records if r["channel"] == "a2a"]),
            "invocations": len(records),
            "llm": _llm_summary(llm_calls),
        }

    by_channel = {
        "mcp": _channel_stats([r for r in all_records if r["channel"] == "mcp"]),
        "a2a": _channel_stats([r for r in all_records if r["channel"] == "a2a"]),
    }

    return {
        "per_owner": per_owner,
        "by_channel": by_channel,
        "total_calls": len(all_records),
        "agent_invocation_count": sum(1 for r in all_records if r["channel"] == "a2a"),
        "llm_calls": all_llm_calls,
        "llm_summary": _llm_summary(all_llm_calls),
    }
