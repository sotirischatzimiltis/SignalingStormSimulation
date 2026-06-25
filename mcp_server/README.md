# `mcp_server/` — MCP Server Wrapping the Simulator

Exposes the live `StormSim` to external callers (the agents) over the **Model
Context Protocol** (package `mcp`, the official Python SDK, FastMCP): telemetry
as readable resources, actuators and the existing solver/forecaster as tools.
Every call is instrumented (count, payload bytes, latency) via
`instrumentation/recorder.py`, separately from the A2A inter-agent traffic
(see `../agents/README.md`) — that separation is what lets the paper attribute
overhead to "talking to the simulator" vs. "agents talking to each other."

---

## Why a thread boundary, and how it's crossed safely

FastMCP serves requests on an asyncio event loop. The simulator
(`sim/simulator.py`'s `StormSim`) is a SimPy discrete-event loop that, under
`cfg.realtime=True`, paces itself to wall-clock time via
`simpy.rt.RealtimeEnvironment` — a **blocking** call (`sim.run(...)`) that must
run on its own thread so the MCP server's event loop can serve requests
concurrently.

This reuses the exact pattern `scripts/gui.py`'s `SimRunner` already
established for the same problem (an external control surface — there,
matplotlib sliders; here, MCP tool handlers — living on a different thread
than the sim): the sim runs on a background thread; the foreign thread never
calls `sim.set_servers()` / `sim.set_malicious_drop_prob()` directly — it only
writes a **commanded target** into shared state; a small `step(sim, s)`
controller (same contract as every other baseline controller in
`sim/controllers.py`), installed via `sim.run(controller=...)` and therefore
running **on the sim's own thread**, polls that state once per `sample_dt_s`
tick and is the only code path that calls the real actuators. See
`sim_host.py`'s `_BridgeController`.

This poll delay is not incidental plumbing — it is exactly what makes a
late-arriving command show up as queue growth: the lateness *is* the
measured agentic overhead acting on the queue (the project's core
real-time-overhead requirement).

## Files

- **`sim_host.py`** — `SimHost` owns one live `StormSim` plus `CommandState`
  (the plain-Python shared target state) and the `_BridgeController`.
  `SimHost.start()` is an **explicit** trigger, not automatic on
  construction — the sim clock must not start ticking until the orchestrator
  has confirmed every agent process is actually up and listening. With
  `single_storm_traffic()`'s storm window lasting only 12 real seconds at the
  demo's default `rt_factor=0.2`, an unconditioned auto-start can let the
  entire storm elapse in real time before the agents are wired up to watch
  it. `start_episode` (a tool, below) is what calls `SimHost.start()`.
- **`tools.py`** — plain functions over a `SimHost`, callable and
  unit-testable without FastMCP/HTTP running:
  - `tool_set_servers` / `tool_set_malicious_drop_prob` — queue a commanded
    target only; never touch the actuator directly (see above).
  - `tool_lyapunov_solve` — calls `sim.controllers.LyapunovController`'s
    existing drift-plus-penalty `_objective`/`_lambda_estimate` math. The
    prior paper's solver is reused, never re-derived by a (future) LLM; the
    tool returns the proposed `c` but does **not** apply it — the caller
    decides whether to follow up with `set_servers`.
  - `forecast_arrival_rate` / `tool_forecast` — a real, causal estimator
    (linear trend over recent telemetry only). Deliberately **not** the
    oracle `ForecastLyapunov._lambda_estimate` uses (which peeks
    `cfg.traffic.rates_at()` into the future) — the whole point of this tool
    is to be something an agent could legitimately call mid-storm.
    `forecast_arrival_rate` has no MCP dependency, so a future
    `ForecastLyapunov`-via-forecaster variant (needed for the fairness
    ablation — see `../sim/README_controllers.md`) can import it directly.
  - `tool_get_telemetry` — bulk read (used by the strategic agent's
    episode-end recovery validation).
- **`server.py`** — `build_mcp_app(host, recorder, ...)`: wires each
  `tools.py` function behind `@mcp.tool()` / `@mcp.resource()`, each wrapped
  identically by `recorder.record(...)`. Resources: `telemetry://latest`,
  `sim://status` (`is_started`, `is_done`, `mu_single`). Tools:
  `start_episode`, `set_servers`, `set_malicious_drop_prob`,
  `lyapunov_solve`, `forecast`, `get_telemetry`.
- **`__main__.py`** — `python -m mcp_server --rt-factor 0.2 --port 8800
  --run-id <id>`. Builds the `SimConfig` (matching
  `scripts/compare_baselines.py`'s `lq_max=1500`), constructs `SimHost`
  (without starting it), serves the FastMCP app until terminated. Its
  `InstrumentedRecorder` writes straight to PostgreSQL as calls happen (see
  `../instrumentation/README.md`) -- nothing dumped to disk on exit.

## MCP has no push/subscribe primitive

Resources are pull-only. The tactical agent detects triggers by **polling**
`telemetry://latest` on its own schedule (deliberately much slower than the
sim's `sample_dt_s` control tick — see `../agents/README.md`), not by being
notified.

## Typical use (manual, e.g. via the MCP Inspector or a throwaway client)

```python
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

async with streamablehttp_client("http://127.0.0.1:8800/mcp") as (read, write, _):
    async with ClientSession(read, write) as session:
        await session.initialize()
        await session.call_tool("start_episode", arguments={})
        await session.call_tool("set_servers", arguments={"c": 8})
        sample = await session.read_resource("telemetry://latest")
```
