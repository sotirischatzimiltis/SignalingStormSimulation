"""
Tool bodies, kept as plain functions over a SimHost so they're callable (and
unit-testable) without FastMCP/HTTP running. mcp_server/server.py wraps each
of these in an @mcp.tool()/@mcp.resource() decorator plus instrumentation.

set_servers / set_malicious_drop_prob only ever queue a commanded target on
the SimHost (see sim_host.py) -- they never call the sim's actuators
directly, since this code runs on the MCP/asyncio thread, not the sim's own
thread.

lyapunov_solve reuses sim.controllers.LyapunovController's existing
_objective/_lambda_estimate math (the drift-plus-penalty solver from the
prior paper) -- it is never re-derived here, only called and *not* applied
(the caller, e.g. an agent, decides whether/when to actually call
set_servers with the result).

forecast_arrival_rate is a real, causal estimator (linear trend over recent
telemetry only) -- deliberately NOT the oracle ForecastLyapunov uses
(sim.cfg.traffic.rates_at), since the whole point of this tool is to replace
that oracle with something an agent could legitimately call mid-storm.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import List, Optional

from sim.controllers import LyapunovController
from sim.simulator import TelemetrySample

from .sim_host import SimHost


def tool_start_episode(host: SimHost) -> dict:
    """Explicitly starts the sim clock. Called once by the orchestrator
    after all agent processes are confirmed ready -- see sim_host.py for why
    this must not happen automatically on process boot."""
    host.start()
    return {"started": True}


def tool_set_servers(host: SimHost, c: int) -> dict:
    host.command_set_servers(c)
    latest = host.latest_telemetry()
    return {"accepted_c": c, "queued_at_sim_t": latest.t if latest else None}


def tool_set_malicious_drop_prob(host: SimHost, p: float) -> dict:
    host.command_set_drop_prob(p)
    latest = host.latest_telemetry()
    return {"accepted_drop_prob": p, "queued_at_sim_t": latest.t if latest else None}


def tool_lyapunov_solve(
    host: SimHost, V: float = 1000.0, W: float = 1.0, lam_override: Optional[float] = None
) -> dict:
    s = host.latest_telemetry()
    if s is None:
        raise RuntimeError("no telemetry yet")
    ctrl = LyapunovController(V=V, W=W)
    lam = lam_override if lam_override is not None else ctrl._lambda_estimate(host.sim, s)
    best_c, best_obj = host.sim.c, float("inf")
    for c in range(1, host.sim.cfg.c_max + 1):
        obj = ctrl._objective(host.sim, s, c, lam)
        if obj < best_obj:
            best_obj, best_c = obj, c
    return {"c": best_c, "objective": best_obj, "lam_used": lam, "t": s.t}


def forecast_arrival_rate(
    history: List[TelemetrySample], horizon_s: float, window_s: float = 30.0
) -> dict:
    """Plain function, no MCP dependency, so a future controller (e.g. a
    ForecastLyapunov variant for the fairness ablation) can import and call
    it directly without depending on the MCP layer."""
    if not history:
        return {"lam_hat": 0.0, "method": "empty", "window_s": window_s}
    t_now = history[-1].t
    window = [s for s in history if s.t >= t_now - window_s]
    if len(window) < 2:
        return {"lam_hat": history[-1].lam_target, "method": "flat", "window_s": window_s}

    xs = [s.t for s in window]
    ys = [s.lam_target for s in window]
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    den = sum((x - mean_x) ** 2 for x in xs)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / den if den > 0 else 0.0
    intercept = mean_y - slope * mean_x
    lam_hat = max(0.0, slope * (t_now + horizon_s) + intercept)
    return {"lam_hat": lam_hat, "method": "linear_trend", "window_s": window_s, "slope": slope}


def tool_forecast(host: SimHost, horizon_s: float = 5.0, window_s: float = 30.0) -> dict:
    history = host.telemetry_window(last_n=300)
    return forecast_arrival_rate(history, horizon_s, window_s)


def tool_get_telemetry(host: SimHost, last_n: int = 1) -> dict:
    samples = host.telemetry_window(last_n)
    return {"samples": [asdict(s) for s in samples], "mu_single": host.mu_single()}
