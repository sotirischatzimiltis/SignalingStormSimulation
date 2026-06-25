# `agents/` — Coordinator / Tactical / Strategic over Real A2A

Three agents, mapped to control timescales, running as **parallel OS
processes**, communicating over the **Agent2Agent protocol** (package
`a2a-sdk`, the official Python SDK — `AgentExecutor` / `DefaultRequestHandler`
/ `EventQueue` / `TaskUpdater`, JSON-RPC over Starlette+uvicorn). Only
Tactical and Strategic also speak MCP to the simulator (`mcp_server/`) — the
Coordinator is pure A2A, which keeps the two overhead channels (inter-agent
coordination vs. tool/context retrieval) cleanly attributable in the
overhead report.

Every agent owns exactly one `Reasoner` (see `base/reason.py`) and calls
`await self.reasoner.reason(ctx)` at the single decision point. Three
backends are pluggable per agent (`base/llm.py`): `stub` (deterministic
rules + a randomized 100–500ms async latency standing in for inference
latency — reproducible, no model needed), `ollama` (local, default), or
`anthropic`/`openai` (cloud, via API key). On any provider failure or
unparseable output, `LLMReasoner` falls back to the same deterministic rule
the stub uses — a resilience demo should not itself crash because a local
model wasn't running. See **LLM backends** below for configuration.

---

## Design choice: every A2A exchange is short

Every message between agents is a short request/response — the callee's
`AgentExecutor.execute()` reaches a terminal task state (`COMPLETED` /
`FAILED`) within the same call (an instant ack, or after one stub-`reason()`
delay). There are **no long-held streaming A2A connections**. The one
genuinely long-running piece of work — Tactical's trigger-poll loop, which
spans the whole episode — runs as an independent background `asyncio` task
inside Tactical's own process, kicked off by a fast `run_episode` ack; it
only *emits* discrete A2A messages at meaningful moments (escalation,
episode-complete). This sidesteps A2A streaming-response semantics entirely
and keeps every A2A message separately countable for the overhead report.
All payloads are structured (`a2a.helpers.new_data_message` /
`get_data_parts` — protobuf `Struct`, not JSON-stuffed-into-text).

## `base/` — shared boilerplate (not copy-pasted 3x)

- **`a2a_app.py`** — `make_agent_card`, `build_agent_app`,
  `run_agent_process` (the `AgentCard` / `DefaultRequestHandler` /
  `InMemoryTaskStore` / JSON-RPC route wiring, confirmed against the actually
  installed `a2a-sdk` 1.1.0). `send_data_message(url, data, recorder, name)`
  is the client-side helper every outbound A2A call in this system goes
  through — it resolves the target's `AgentCard`, sends one data message,
  awaits the full response, and is instrumented identically everywhere it's
  used. `wait_until_ready` / `probe_agent_ready` are the **protocol-level**
  readiness probes `scripts/run_agentic.py` uses at startup (fetching the
  real agent card / doing a real MCP handshake — not a fabricated
  `/healthz`).
- **`reason.py`** — `Reasoner` protocol + `StubReasoner`.
- **`llm.py`** — `build_reasoner(provider, model, base_url, prompt_fn,
  fallback_fn, system_prompt, ...)`, the factory every agent's `__main__.py`
  calls. `LLMReasoner` wraps a provider call (`ollama` via plain `httpx` to
  the local daemon's `/api/chat` with `format: "json"`; `anthropic`/`openai`
  via their HTTP APIs directly, no extra SDK dependency) with the prompt
  builder and a best-effort JSON extractor, falling back to `fallback_fn` on
  any failure. `warmup()` does one throwaway call at agent startup so a cold
  model load (~15–20s observed for a 3B Ollama model) is paid before the
  episode starts, not charged against the first real decision.
- **`mcp_client.py`** — `MCPBridge`, the async context manager Tactical and
  Strategic use to talk to `mcp_server/`. Uses `contextlib.AsyncExitStack`
  internally — a bare manual `__aenter__`/`__aexit__` pairing was found (the
  hard way) to leave `anyio`'s task group half-open if the connection attempt
  fails partway through (e.g. probing before the MCP server's port is up
  yet), which corrupts later retries when the orphaned generator is
  eventually garbage-collected from an unrelated task.

## Coordinator (SMO level) — `coordinator/`

Pure A2A, no MCP. Routes the episode lifecycle:
- `start_episode` (from `scripts/run_agentic.py`) → sends `run_episode` to
  Tactical (fast ack), then **awaits** Tactical's separate `episode_complete`
  message (via an `asyncio.Event`, so concurrent A2A tasks on the same event
  loop interleave normally) plus any still-in-flight `storm_complete`
  handling (below), then relays the accumulated per-storm results back as
  the `start_episode` response. This call legitimately blocks for the whole
  episode (`scripts/run_agentic.py` uses `httpx` with no timeout on this
  specific call).
- `storm_complete` (from Tactical, once per storm — see Tactical below) →
  acks immediately; the actual work (send `validate_recovery` to Strategic
  for that storm's window, then forward Strategic's proposed `policy_update`
  back to Tactical) runs as a background task, tracked in
  `self._pending_storm_tasks` and gathered before `start_episode` returns.
  Mirrors Tactical's own ack-fast pattern, for the same reason: Strategic's
  reasoning latency must never block Tactical's poll loop.
- `escalation` (from Tactical) → consults its reasoner for a logged
  approve/comment judgement and surfaces it in the ack. With one agent path
  there is nothing yet for that judgement to actually *override* (no
  competing proposals, no guardrail breached); real conflict resolution
  becomes meaningful once there's something to resolve — the LLM is
  genuinely consulted, control flow just doesn't branch on it yet.

## Tactical (Near-RT) — `tactical/`

Owns absorption + adaptation + recovery scale-down. `run_episode` is acked
immediately; the actual work is a background poll loop (`poll_interval_s`,
default 2s real time — deliberately much slower than the sim's
`sample_dt_s=0.5s` tick, per the project's own latency-budget constraint).
Each poll reads `telemetry://latest` + `sim://status` via MCP and evaluates
client-side triggers (MCP has no push primitive):
- **onset** — `lam_target` rises past `1.5×` the captured pre-storm baseline.
- **escalation** — `queue_len` exceeds `self._policy["escalation_threshold"]`
  (tunable, see Evolution below; default 40); debounced (10s) so a sustained
  overload doesn't spam the Coordinator every poll.
- **subside** — `lam_target` has held near baseline for several consecutive
  polls. This doubles as "recovery scale-down": re-querying `lyapunov_solve`
  with the now-lower `lam_target` naturally proposes a smaller `c`, so it is
  not a separate decision branch.

Onset/subside bookkeeping is evaluated **independently** of the escalation
check, not as one mutually-exclusive if/elif chain — a severe storm can blow
the queue past the escalation threshold in the very first poll after onset,
and checking escalation first would silently skip marking `in_storm` (and
therefore never detect that storm's subside either) for exactly the storms
violent enough to matter most. `trigger` (used below for which lever to
fire) still gives escalation priority when both coincide on the same poll;
`storm_event` (onset/subside) is tracked separately and always wins for the
purpose of pairing a storm's start with its end.

On every trigger: call the `lyapunov_solve` tool (adaptation lever — the
existing solver, not re-derived) then `set_servers` **immediately**; on
escalation specifically, also reason about the rate-limit decision (the
absorption lever Lyapunov alone doesn't have), apply
`set_malicious_drop_prob` (floored at `self._policy["drop_prob_floor"]`),
and send a debounced `escalation` message to the Coordinator. On confirmed
subside, send `storm_complete` (see Evolution below). On loop exit (sim
signals done), sends `episode_complete` — if a storm's subside never
resolved in time, it's reported as `storm_complete` first anyway, using the
final polled time as `subside_t`, so every onset gets a matching entry.

**Found the hard way:** the fast, deterministic adaptation lever
(`lyapunov_solve`/`set_servers`) must never be sequenced *after* `await
self.reasoner.reason(...)`. With the stub's fake 100–500ms latency this
ordering was invisible; with a real local LLM (~6–12s per call observed with
`llama3.2:latest`), gating the fast lever behind the slow one let the queue
explode for the entire decision latency before servers ever scaled —
regressing P all the way back to the no-intervention baseline in testing.
The fix: apply `set_servers` first, then run the rate-limit reasoning as a
**background `asyncio.create_task`** (capped at 3 concurrent, pruned of
finished ones) so it never blocks the fast lever or the next poll. The same
principle now also applies to `storm_complete`: it's sent as a background
task too, since Strategic's per-storm reasoning must never delay the poll
loop (which is calm by definition right after a subside). All pending
background tasks (rate-limit decisions and storm_complete sends) are
awaited before the episode-end MCP session closes.

## Strategic (Non-RT) — `strategic/`

Once per storm (not once per episode — `single_storm_traffic()` is just the
N=1 case of the same path). On `validate_recovery`: pulls the full telemetry
history via one MCP `get_telemetry` call, computes `sim/metrics.py`'s
`resilience_score()` directly (pure post-processing, no MCP needed for the
math) over that storm's *self-detected* `(onset_t, subside_t)` window
(no hardcoded scenario boundaries — see Tactical above), runs the
reasoner, and inserts one row into PostgreSQL's `storms` table (see
`README_database.md`) — one `INSERT` per storm, no read-modify-write file.

## Evolution: feeding memory back into the next storm

`multi_storm_traffic()` (three storms of growing intensity in **one**
episode) is what this is for — its own purpose is to let the system handle
storm 3 better than storm 1 once Strategic's per-storm assessment feeds
back into Tactical, within that one episode (not across separate runs).

Two A2A message kinds carry this loop, reusing the existing hub-and-spoke
pattern (Tactical and Strategic never talk directly, only through
Coordinator):
- **`storm_complete`** (Tactical → Coordinator): `{storm_index, onset_t,
  subside_t, policy_used}`. Tactical is the sole source of truth for its own
  policy, so it reports what was actually active during that storm rather
  than Coordinator/Strategic keeping a shadow copy.
- **`policy_update`** (Coordinator → Tactical): `{escalation_threshold,
  drop_prob_floor}`, the absolute new values for the *next* storm, replacing
  `self._policy` wholesale.

`validate_recovery` (Coordinator → Strategic) is reused for the per-storm
call, parameterized by the window above instead of always covering the
whole episode. Strategic's reasoner now also proposes the next storm's
policy (`default_decision_fn`/`prompt_fn` in `strategic/executor.py`):
tighten both knobs (`escalation_threshold` ×0.8, floored at 10;
`drop_prob_floor` +0.2, capped at 1.0) when that storm's `absorption < 0.7`
(the queue grew too much before Tactical reacted), otherwise hold. Lyapunov's
`V`/`W` weights are deliberately **not** touched by this loop — that lever
stays the existing deterministic baseline controller, untuned.

PostgreSQL's `storms` table now holds one row per storm with `resilience`,
`classification`, `lessons`, `decision_source`, and
`policy_before`/`policy_after` (JSONB columns) — query it directly (see
`README_database.md` for example queries) to see exactly what changed and
why between storms. `scripts/dashboard.py`'s Live Run tab charts P and the
two policy knobs per storm_index when a report has more than one; its
Compare Runs tab adds `P_first`/`P_last`/`improved` columns for multi-storm
runs.

## LLM backends

Each agent's `__main__.py` takes `--llm-provider {stub,ollama,anthropic,openai}`
(default `ollama`), `--llm-model`, `--llm-base-url` (Ollama only, default
`http://localhost:11434`). `scripts/run_agentic.py` exposes the same flags
plus per-agent model overrides (`--coordinator-model`, `--tactical-model`,
`--strategic-model`), applying one provider to all three agents (mixing
providers per agent is supported by each agent's own flags if launched
manually — `run_agentic.py` just doesn't expose that combination via CLI
yet). Cloud providers read their API key from the standard env var
(`ANTHROPIC_API_KEY` / `OPENAI_API_KEY`) — subprocesses inherit the
orchestrator's environment, so `export`ing it before running is enough.

```bash
# default: all three agents on a local Ollama model
python -m scripts.run_agentic

# match the original size hierarchy once smaller models are pulled
ollama pull llama3.2:1b
python -m scripts.run_agentic --tactical-model llama3.2:1b

# use a cloud model for Strategic (the "largest model" tier) instead
OPENAI_API_KEY=... python -m agents.strategic --llm-provider openai --llm-model gpt-4o-mini ...

# fall back to fast, deterministic rules (no model call at all)
python -m scripts.run_agentic --llm-provider stub
```

**`rt_factor` matters a lot more with a real model than with the stub.** A
real local 3B Ollama model takes ~1–12s per decision (vs. the stub's fake
100–500ms). At the demo's original `rt_factor=0.2` (5x time compression),
that latency costs 5x as many *simulated* seconds, which can consume most or
all of `single_storm_traffic()`'s 60-second storm window before the action
ever lands — verified to regress P all the way back to the no-intervention
baseline in testing. **Default is now `rt_factor=1.0`** (true real time,
~17 minutes for the full episode) specifically so a real model's latency
gets a fair fraction of the storm's actual response window, matching how a
real deployment would be paced. A verified `rt_factor=1.0` run with real
Ollama (`llama3.2:latest`) decisions scored **P=1.06** — yes, just over 1.0,
not a bug: `resilience_score`'s absorption term is utility-during-storm
divided by the *pre-storm baseline* utility, and aggressive enough server
scaling can push momentary utility above that baseline, so the ratio can
exceed 1. Lower `rt_factor` values remain useful for fast plumbing checks
with `--llm-provider stub`, just don't expect a slow real model to look good
under heavy time compression.

## Seeing what each LLM call actually did

Every real (non-stub) `Reasoner.reason()` call is logged in full -- not just
the one-line decision prints already in `tactical/executor.py`. `agents/base/llm.py`'s
provider functions (`_ollama_call`/`_anthropic_call`/`_openai_call`) now return
a `ProviderResponse(text, tokens_in, tokens_out)` instead of a bare string,
reading token counts from each provider's own response (`prompt_eval_count`/
`eval_count` for Ollama, `usage.input_tokens`/`output_tokens` for Anthropic,
`usage.prompt_tokens`/`completion_tokens` for OpenAI). `LLMReasoner.reason()`
prints a one-line `[llm] <provider>:<model> tokens_in=.. tokens_out=..
latency=..s source=..` summary (visible live in that agent's log tail) and,
if constructed with a `recorder=` (every agent's `__main__.py` passes its own
`InstrumentedRecorder`), logs the full system/user prompt, raw response,
parsed decision, token counts, latency, and source (`llm:<provider>:<model>`
or `fallback`) via a new `record_llm()` method -- a separate `LLMCallRecord`
type from `CallRecord`, since the MCP/A2A overhead story deliberately never
keeps payload content, only byte counts.

Each call writes one row straight to PostgreSQL's `llm_calls` table as it
completes (via the same per-process `InstrumentedRecorder`, now backed by
`instrumentation/db.py` instead of an in-memory list dumped to a JSON file
on exit -- see `README_database.md`), so `db.build_overhead_report()` picks
them up automatically (`report["llm_calls"]`, flattened across owners and
tagged; `report["llm_summary"]` / `per_owner[...]["llm"]` for count/
fallback-count/total tokens/mean latency) with no changes needed to
`scripts/run_agentic.py` -- it already queries whatever
`build_overhead_report()` returns. `scripts/dashboard.py`'s Live Run /
final-report view reads `report["overhead"]["llm_calls"]` into a table (one
row per call, prompt/response previewed) with a selector to see any one
call's full system prompt, user prompt, raw response, parsed decision, and
error (if it fell back) in full. `--llm-provider stub` never makes a real
provider call, so `llm_calls` is empty for stub runs -- the dashboard
section and the report's "LLM usage:" line simply don't appear, rather than
showing empty/zero noise.

## Where the overhead numbers come from

Every process keeps its own `InstrumentedRecorder` (MCP calls for
`mcp_server`/Tactical/Strategic; A2A calls for all four roles), writing one
row to PostgreSQL's `calls` table per call as it completes.
`instrumentation/db.py`'s `build_overhead_report()` queries and aggregates
them after the episode (no per-process dump-and-merge step -- see
`README_database.md`). Both the MCP **and**
A2A side of the same logical call are recorded independently (client-side in
the agent process, server-side in the callee's process) — by design, so the
report's `by_channel` totals double-count each call from both ends; this is
intentional, not a bug, since the paper's overhead story needs both
perspectives. One caveat: the Coordinator's `start_episode` A2A call's
recorded "latency" spans the *entire episode* (it's a single long-blocked
await by design — see above), so it dominates the A2A channel's mean/p95
latency figures; read the `escalation` / `run_episode` / `episode_complete` /
`storm_complete` / `validate_recovery` / `policy_update` message latencies
individually for the actually interesting per-message overhead numbers.
