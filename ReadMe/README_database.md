# Database: PostgreSQL as the source of truth

Every run's structured data -- config/status, every MCP/A2A call, every LLM
call's full prompt/response/tokens, every storm's evolution record -- lives
in PostgreSQL. There is no more per-process JSON dump, `status.json`,
`report.json`, or `memory/episode_<run_id>.json`; only `logs/<agent>.log`
(plain stdout, tailed live by the dashboard) stays a file.

All of the logic lives in `instrumentation/db.py`. See `README_agents.md`
("Where the overhead numbers come from", "Seeing what each LLM call
actually did", "Evolution") for how each table gets written during an
episode; this file is the schema/query reference.

## Connecting

```bash
STORMSIM_DATABASE_URL=postgresql://stormsim:stormsim@localhost:5433/stormsim   # default, no need to set explicitly
psql "$STORMSIM_DATABASE_URL"
```

Deliberately **not** port 5432 -- that's reserved for whatever else might
already be running on a dev machine (this project's default assumes a
dedicated container; see `README_operations.md` "Database setup").

## Schema

```
runs
  run_id TEXT PRIMARY KEY
  created_at, updated_at DOUBLE PRECISION   -- unix timestamps
  phase TEXT                                 -- starting|ready|running|stopping|done|stopped|failed
  scenario TEXT                              -- single_storm|multi_storm
  rt_factor DOUBLE PRECISION
  llm_provider TEXT
  config JSONB                               -- full CLI args (vars(args))
  mcp_url TEXT
  error TEXT

calls                                        -- one row per instrumented MCP/A2A call
  id SERIAL PRIMARY KEY
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE
  owner TEXT          -- mcp_server|coordinator|tactical|strategic|orchestrator
  channel TEXT         -- mcp|a2a
  name TEXT            -- tool/resource name or A2A message kind
  t_start, latency_s DOUBLE PRECISION
  bytes_in, bytes_out INTEGER
  ok BOOLEAN
  error TEXT

llm_calls                                    -- one row per Reasoner.reason() call (real providers only)
  id SERIAL PRIMARY KEY
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE
  owner TEXT
  provider, model TEXT
  system_prompt, user_prompt, response_text TEXT
  decision JSONB
  tokens_in, tokens_out INTEGER              -- NULL if the provider didn't report them, or on fallback
  latency_s DOUBLE PRECISION
  source TEXT          -- "llm:<provider>:<model>" or "fallback"
  t_start DOUBLE PRECISION
  error TEXT

storms                                       -- one row per storm (the evolution record)
  id SERIAL PRIMARY KEY
  run_id TEXT REFERENCES runs(run_id) ON DELETE CASCADE
  storm_index INTEGER
  t0, td DOUBLE PRECISION                    -- self-detected onset/subside sim-time
  resilience JSONB                           -- {P, absorption, adaptation, trec, recovery_time}
  classification, lessons, decision_source TEXT
  policy_before, policy_after JSONB          -- {escalation_threshold, drop_prob_floor}
  written_at DOUBLE PRECISION
```

`ensure_schema()` runs `CREATE TABLE IF NOT EXISTS` for all four on every
`run_agentic.py` invocation -- no migration framework; this project's scale
doesn't warrant one yet. JSONB columns round-trip as plain Python dicts via
an `asyncpg` type codec (`instrumentation/db.py: _init_codecs`) -- never
`json.dumps()`/`json.loads()` manually at a call site.

## Example queries

```sql
-- Every LLM call that fell back to the deterministic rule (provider down/timeout/bad JSON)
SELECT run_id, owner, model, error FROM llm_calls WHERE source = 'fallback';

-- P trend across storms for one run (the evolution story)
SELECT storm_index, resilience->>'P' AS p, policy_before, policy_after
FROM storms WHERE run_id = 'run_1234567890' ORDER BY storm_index;

-- Mean MCP latency per run, most recent first
SELECT r.run_id, r.scenario, AVG(c.latency_s) AS mean_mcp_latency_s
FROM calls c JOIN runs r USING (run_id)
WHERE c.channel = 'mcp'
GROUP BY r.run_id, r.scenario, r.created_at
ORDER BY r.created_at DESC;

-- All runs that used real Ollama and their final resilience
SELECT r.run_id, r.rt_factor, s.resilience->>'P' AS final_p
FROM runs r JOIN storms s USING (run_id)
WHERE r.llm_provider = 'ollama' AND r.phase = 'done'
ORDER BY r.created_at DESC;
```

## Python access

```python
from instrumentation import db

async def main():
    async with db.connection() as conn:           # one-off, for scripts/notebooks
        runs = await db.list_runs(conn)
        storms = await db.get_storms(conn, "run_1234567890")
```

`db.connection()` opens and closes a single `asyncpg` connection per call --
the same pattern `scripts/dashboard.py` and `scripts/run_agentic.py` use.
Long-lived server processes (`mcp_server`, the 3 agents) use `db.get_pool()`
instead, created inside their own ASGI/MCP lifespan hook (see
`agents/base/a2a_app.py` and `mcp_server/server.py`) -- never in `main()`,
since `asyncpg` pools are bound to the event loop they're created in and
`uvicorn.run()`/`mcp.run()` each own their own loop.

**Note on the MCP server's lifespan hook specifically**: the MCP SDK's
`lifespan` parameter is scoped *per session* (one MCP streamable-http
connection), not once per process like Starlette's ASGI lifespan -- closing
the pool when any one session ends would break every other agent's
still-open session. `mcp_server/server.py`'s lifespan creates the pool at
most once (lock-guarded against the startup race between concurrent
sessions) and deliberately never closes it; it's reclaimed when the
short-lived, per-episode process exits.

## Not migrated

Historical JSON artifacts from before this migration (`report.json`,
`memory/episode_*.json`, per-process `<owner>.json` dumps) are not imported
into PostgreSQL -- they stay as standalone files for whatever old runs
already produced them. Only new runs use the database.
