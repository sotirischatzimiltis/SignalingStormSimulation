"""
FastMCP app wiring: each tool/resource is a thin wrapper around the plain
functions in tools.py, instrumented identically via
instrumentation.recorder.InstrumentedRecorder so every MCP call's count,
payload bytes, and latency are measured the same way regardless of which
tool/resource it is.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict

from mcp.server.fastmcp import FastMCP

from instrumentation import db
from instrumentation.recorder import InstrumentedRecorder

from . import tools
from .sim_host import SimHost


def build_mcp_app(
    host: SimHost,
    recorder: InstrumentedRecorder,
    bind_host: str = "127.0.0.1",
    port: int = 8800,
) -> FastMCP:
    # IMPORTANT: the MCP SDK's `lifespan` is scoped per-session (entered/
    # exited once per streamable-http connection -- see Server.run() in
    # mcp/server/lowlevel/server.py), NOT once per process like Starlette's
    # ASGI lifespan (which agents/base/a2a_app.py relies on correctly). This
    # server has several concurrent long-lived sessions (Tactical, Strategic,
    # the orchestrator) plus short-lived ones (readiness probes) sharing one
    # `recorder`/pool -- closing the pool when any ONE session ends breaks
    # every other still-open session. So: create the pool at most once
    # (whichever session's lifespan enters first, guarded by a lock against
    # the startup race between concurrent sessions), and deliberately never
    # close it here -- it's reclaimed when this short-lived, per-episode
    # process exits.
    pool_lock = asyncio.Lock()

    @asynccontextmanager
    async def lifespan(server: FastMCP):
        async with pool_lock:
            if recorder.pool is None:
                recorder.set_pool(await db.get_pool())
        yield

    mcp = FastMCP(
        "stormsim-mcp",
        host=bind_host,
        port=port,
        streamable_http_path="/mcp",
        json_response=True,
        lifespan=lifespan,
    )

    @mcp.tool()
    async def start_episode() -> dict:
        """Starts the sim clock. Call once, after all agent processes are
        confirmed ready -- not automatic on server boot (see sim_host.py)."""
        async with recorder.record("mcp", "start_episode") as rec:
            result = tools.tool_start_episode(host)
            rec.payload_out = result
            return result

    @mcp.tool()
    async def set_servers(c: int) -> dict:
        """Queue a commanded server count c (the adaptation lever). Takes
        effect on the sim's next control tick, not instantaneously."""
        async with recorder.record("mcp", "set_servers", {"c": c}) as rec:
            result = tools.tool_set_servers(host, c)
            rec.payload_out = result
            return result

    @mcp.tool()
    async def set_malicious_drop_prob(p: float) -> dict:
        """Queue a commanded malicious-attempt drop probability p (the
        absorption / rate-limiting lever). Takes effect on the sim's next
        control tick."""
        async with recorder.record("mcp", "set_malicious_drop_prob", {"p": p}) as rec:
            result = tools.tool_set_malicious_drop_prob(host, p)
            rec.payload_out = result
            return result

    @mcp.tool()
    async def lyapunov_solve(V: float = 1000.0, W: float = 1.0, lam_override: float = None) -> dict:
        """Drift-plus-penalty optimum c for the current telemetry (the prior
        paper's solver, called as a tool -- not re-derived). Does NOT apply
        the result; the caller decides whether to follow up with
        set_servers."""
        payload_in = {"V": V, "W": W, "lam_override": lam_override}
        async with recorder.record("mcp", "lyapunov_solve", payload_in) as rec:
            result = tools.tool_lyapunov_solve(host, V=V, W=W, lam_override=lam_override)
            rec.payload_out = result
            return result

    @mcp.tool()
    async def forecast(horizon_s: float = 5.0, window_s: float = 30.0) -> dict:
        """Causal arrival-rate forecast (linear trend over recent telemetry
        only -- not an oracle peek at the traffic schedule)."""
        payload_in = {"horizon_s": horizon_s, "window_s": window_s}
        async with recorder.record("mcp", "forecast", payload_in) as rec:
            result = tools.tool_forecast(host, horizon_s=horizon_s, window_s=window_s)
            rec.payload_out = result
            return result

    @mcp.tool()
    async def get_telemetry(last_n: int = 1) -> dict:
        """Bulk telemetry read (e.g. for the strategic agent's episode-end
        recovery validation)."""
        async with recorder.record("mcp", "get_telemetry", {"last_n": last_n}) as rec:
            result = tools.tool_get_telemetry(host, last_n=last_n)
            rec.payload_out = {"n_samples": len(result["samples"])}
            return result

    @mcp.resource("telemetry://latest")
    async def telemetry_latest() -> dict:
        async with recorder.record("mcp", "resource:telemetry_latest") as rec:
            sample = host.latest_telemetry()
            result = asdict(sample) if sample else {}
            rec.payload_out = result
            return result

    @mcp.resource("sim://status")
    async def sim_status() -> dict:
        async with recorder.record("mcp", "resource:sim_status") as rec:
            result = {
                "is_started": host.is_started(),
                "is_done": host.is_done(),
                "mu_single": host.mu_single(),
            }
            rec.payload_out = result
            return result

    return mcp
