"""
Entry point: `python -m mcp_server --rt-factor 0.2 --port 8800 --run-id demo`

Builds a SimConfig for the requested scenario with realtime pacing on, starts
the SimHost (sim runs on its own background thread), then serves the FastMCP
app on the main thread until terminated. Every MCP call is written straight
to PostgreSQL as it happens (instrumentation/db.py) via the FastMCP lifespan
hook in mcp_server/server.py -- there is no per-process dump-on-exit step.
"""

from __future__ import annotations

import argparse
import signal
import sys

from sim.config import RRCConfig, SimConfig, multi_storm_traffic, open_ran_arch, single_storm_traffic

from instrumentation.recorder import InstrumentedRecorder

from .server import build_mcp_app
from .sim_host import SimHost

LQ_MAX = 1500.0  # matches scripts/compare_baselines.py, for comparable resilience scores


def build_cfg(args: argparse.Namespace) -> SimConfig:
    traffic = multi_storm_traffic() if args.scenario == "multi_storm" else single_storm_traffic()
    return SimConfig(
        arch=open_ran_arch(),
        rrc=RRCConfig(t300_ms=1000, max_attempts=5),
        traffic=traffic,
        c0=1,
        c_max=args.c_max,
        lq_max=LQ_MAX,
        seed=args.seed,
        realtime=True,
        rt_factor=args.rt_factor,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.add_argument("--rt-factor", type=float, default=1.0)
    p.add_argument("--scenario", choices=["single_storm", "multi_storm"], default="single_storm")
    p.add_argument("--c-max", type=int, default=16)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = build_cfg(args)

    # NOTE: sim_host.start() is NOT called here -- the clock only starts once
    # the orchestrator calls the start_episode MCP tool, after confirming all
    # agent processes are ready (see sim_host.py for why).
    sim_host = SimHost(cfg)

    recorder = InstrumentedRecorder(owner="mcp_server", run_id=args.run_id)

    def _on_signal(signum, frame):
        sys.exit(0)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    mcp = build_mcp_app(sim_host, recorder, bind_host=args.host, port=args.port)
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
