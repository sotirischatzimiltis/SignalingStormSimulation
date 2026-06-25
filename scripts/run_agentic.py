"""
Stands up the MCP server + the three A2A agents, drives one episode of
single_storm_traffic() (or multi_storm_traffic()) end to end, then tears
everything down and prints/saves the combined resilience + overhead report.

Two process-supervision backends, selected by --backend (default
"subprocess", today's exact behavior -- this is additive, not a breaking
change):
  --backend subprocess (default): spawns mcp_server + the 3 agents as real
    OS subprocesses against the active venv, exactly as before.
  --backend docker: spawns the same 4 processes as Docker Compose services
    (see docker-compose.yml) instead. Requires `docker compose up -d
    postgres` to already be running (Postgres is mandatory either way --
    see instrumentation/db.py -- only the *agent* processes are
    containerized, not the database).

Run from the project root:
    python -m scripts.run_agentic
    python -m scripts.run_agentic --rt-factor 1.0     # paper-quality real-time overhead numbers
    python -m scripts.run_agentic --rt-factor 0.05     # fast smoke test
    python -m scripts.run_agentic --backend docker     # containerized agents

Process topology (see ReadMe/README_agents.md, ReadMe/README_mcp_server.md):
  mcp_server (8800) -- owns the live StormSim; coordinator (9001); tactical
  (9002); strategic (9003). This script is the 5th, short-lived orchestrator
  -- it always runs on the host (never containerized itself), since it's
  the thing driving `docker compose` when --backend docker is selected.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.base.a2a_app import probe_agent_ready, send_data_message, wait_until_ready  # noqa: E402
from agents.base.mcp_client import MCPBridge  # noqa: E402
from instrumentation import db  # noqa: E402
from instrumentation.recorder import InstrumentedRecorder  # noqa: E402
from instrumentation.report import format_report  # noqa: E402

DOCKER_SERVICES = ["mcp_server", "coordinator", "tactical", "strategic"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--backend", choices=["subprocess", "docker"], default="subprocess",
        help="subprocess (default): bare OS processes against the venv. docker: the same 4 "
             "processes as Docker Compose services (docker-compose.yml) -- requires "
             "`docker compose up -d postgres` already running.",
    )
    p.add_argument(
        "--rt-factor", type=float, default=1.0,
        help="Wall-clock seconds per simulated second. 1.0 = true real time (~17 min for "
             "single_storm_traffic()'s 1010s horizon) -- gives a real LLM's multi-second decision "
             "latency a fair fraction of the storm's 60s window to land in. Lower values compress "
             "time but can make even a fast LLM's latency dominate a short storm window.",
    )
    p.add_argument("--scenario", choices=["single_storm", "multi_storm"], default="single_storm")
    p.add_argument("--c-max", type=int, default=16)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--poll-interval-s", type=float, default=2.0)
    p.add_argument(
        "--llm-provider", choices=["stub", "ollama", "anthropic", "openai"], default="ollama",
        help="Decision backend for all three agents. 'stub' = deterministic rules, no model call.",
    )
    p.add_argument(
        "--llm-base-url", default="http://localhost:11434",
        help="Ollama daemon URL. With --backend docker and the default value, this is "
             "automatically rewritten to http://host.docker.internal:11434 so containers can "
             "still reach a host-run Ollama daemon -- pass an explicit value to override.",
    )
    p.add_argument("--coordinator-model", default="llama3.2:latest")
    p.add_argument("--tactical-model", default="llama3.2:latest")
    p.add_argument("--strategic-model", default="llama3.2:latest")
    p.add_argument("--stub-latency-min", type=float, default=0.1)
    p.add_argument("--stub-latency-max", type=float, default=0.5)
    p.add_argument("--run-id", default=None)
    p.add_argument(
        "--ready-timeout-s", type=float, default=60.0,
        help="Generous default: each agent warms up its LLM (cold Ollama model load can take "
             "~15-20s) before its A2A server starts listening, and this probe waits for that too.",
    )
    p.add_argument("--episode-timeout-s", type=float, default=1800.0)
    return p.parse_args()


def spawn(run_id: str, log_dir: Path, name: str, module: str, extra_args: List[str], log_files: list):
    """--backend subprocess: real OS process against the active venv."""
    log_path = log_dir / f"{name}.log"
    log_file = open(log_path, "w")
    log_files.append(log_file)
    args = [sys.executable, "-m", module, "--run-id", run_id, *extra_args]
    return subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdout=log_file, stderr=subprocess.STDOUT)


def docker_up(env: dict) -> None:
    """--backend docker: starts the 4 ephemeral agent/mcp_server services.
    Deliberately uses the default (directory-derived) Compose project --
    NOT a per-run -p override -- so `depends_on: postgres` resolves to the
    one persistent postgres service/network already running under that
    same default project, instead of spinning up (and port-conflicting
    with) a second one scoped to a throwaway per-run project. This mirrors
    the existing single-run-at-a-time constraint the subprocess backend
    already has (fixed ports 8800/9001/9002/9003 -- see README_operations.md)."""
    subprocess.run(
        ["docker", "compose", "up", "-d", "--build", *DOCKER_SERVICES],
        cwd=str(PROJECT_ROOT), env=env, check=True,
    )


def docker_log_tails(log_dir: Path, env: dict, log_files: list) -> List[subprocess.Popen]:
    """Mirrors the subprocess backend's "redirect stdout to a file" so
    scripts/dashboard.py's tail_log() works identically either way."""
    tails = []
    for service in DOCKER_SERVICES:
        log_path = log_dir / f"{service}.log"
        log_file = open(log_path, "w")
        log_files.append(log_file)
        proc = subprocess.Popen(
            ["docker", "compose", "logs", "-f", "--no-color", service],
            cwd=str(PROJECT_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT,
        )
        tails.append(proc)
    return tails


def docker_down(env: dict) -> None:
    # Scoped to just the 4 ephemeral services -- never tears down postgres
    # (the persistent service) or the shared network/volume.
    subprocess.run(["docker", "compose", "down", *DOCKER_SERVICES], cwd=str(PROJECT_ROOT), env=env)


async def main_async() -> None:
    args = parse_args()
    run_id = args.run_id or f"run_{int(time.time())}"
    log_dir = PROJECT_ROOT / "instrumentation" / "_runs" / run_id / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Always the host-published ports -- the orchestrator runs on the host
    # in both backends (docker-compose.yml publishes these same ports),
    # only the *agents'* CLI args for reaching each other differ by backend
    # (service names, baked into docker-compose.yml's command: overrides).
    mcp_url = "http://127.0.0.1:8800/mcp"
    coordinator_url = "http://127.0.0.1:9001"
    tactical_url = "http://127.0.0.1:9002"
    strategic_url = "http://127.0.0.1:9003"

    llm_base_url = args.llm_base_url
    if args.backend == "docker" and llm_base_url == "http://localhost:11434":
        llm_base_url = "http://host.docker.internal:11434"

    print(f"[run_agentic] run_id={run_id} backend={args.backend} rt_factor={args.rt_factor} scenario={args.scenario}")

    procs = []
    log_files: list = []
    log_tail_procs: list = []
    result = None
    orchestrator_recorder = InstrumentedRecorder(owner="orchestrator", run_id=run_id)
    docker_env = {**os.environ}

    async with db.connection() as conn:
        await db.ensure_schema(conn)
        orchestrator_recorder.set_pool(conn)
        await db.insert_run(conn, run_id, "starting", vars(args))

        try:
            if args.backend == "subprocess":
                procs.append(spawn(run_id, log_dir, "mcp_server", "mcp_server", [
                    "--port", "8800", "--rt-factor", str(args.rt_factor),
                    "--scenario", args.scenario, "--c-max", str(args.c_max), "--seed", str(args.seed),
                ], log_files))
                procs.append(spawn(run_id, log_dir, "coordinator", "agents.coordinator", [
                    "--port", "9001", "--tactical-url", tactical_url, "--strategic-url", strategic_url,
                    "--llm-provider", args.llm_provider, "--llm-model", args.coordinator_model,
                    "--llm-base-url", llm_base_url,
                    "--stub-latency-min", str(args.stub_latency_min), "--stub-latency-max", str(args.stub_latency_max),
                    "--seed", str(args.seed),
                ], log_files))
                procs.append(spawn(run_id, log_dir, "tactical", "agents.tactical", [
                    "--port", "9002", "--mcp-url", mcp_url, "--coordinator-url", coordinator_url,
                    "--poll-interval-s", str(args.poll_interval_s),
                    "--llm-provider", args.llm_provider, "--llm-model", args.tactical_model,
                    "--llm-base-url", llm_base_url,
                    "--stub-latency-min", str(args.stub_latency_min), "--stub-latency-max", str(args.stub_latency_max),
                    "--seed", str(args.seed),
                ], log_files))
                procs.append(spawn(run_id, log_dir, "strategic", "agents.strategic", [
                    "--port", "9003", "--mcp-url", mcp_url,
                    "--llm-provider", args.llm_provider, "--llm-model", args.strategic_model,
                    "--llm-base-url", llm_base_url,
                    "--stub-latency-min", str(args.stub_latency_min), "--stub-latency-max", str(args.stub_latency_max),
                    "--seed", str(args.seed),
                ], log_files))
            else:
                docker_env.update({
                    "RUN_ID": run_id,
                    "RT_FACTOR": str(args.rt_factor),
                    "SCENARIO": args.scenario,
                    "C_MAX": str(args.c_max),
                    "SEED": str(args.seed),
                    "LLM_PROVIDER": args.llm_provider,
                    "LLM_BASE_URL": llm_base_url,
                    "COORDINATOR_MODEL": args.coordinator_model,
                    "TACTICAL_MODEL": args.tactical_model,
                    "STRATEGIC_MODEL": args.strategic_model,
                    "STUB_LATENCY_MIN": str(args.stub_latency_min),
                    "STUB_LATENCY_MAX": str(args.stub_latency_max),
                    "POLL_INTERVAL_S": str(args.poll_interval_s),
                })
                print("[run_agentic] docker compose up -d --build ...")
                docker_up(docker_env)
                log_tail_procs = docker_log_tails(log_dir, docker_env, log_files)

            print("[run_agentic] waiting for readiness (protocol-level probes)...")

            async def probe_mcp():
                async with MCPBridge(mcp_url, InstrumentedRecorder(owner="_readiness_probe")):
                    pass

            await wait_until_ready(probe_mcp, timeout_s=args.ready_timeout_s)
            await wait_until_ready(lambda: probe_agent_ready(coordinator_url), timeout_s=args.ready_timeout_s)
            await wait_until_ready(lambda: probe_agent_ready(tactical_url), timeout_s=args.ready_timeout_s)
            await wait_until_ready(lambda: probe_agent_ready(strategic_url), timeout_s=args.ready_timeout_s)
            print("[run_agentic] all processes ready.")
            await db.update_run(conn, run_id, phase="ready")

            # Start the sim clock ONLY now -- see mcp_server/sim_host.py for why
            # this must be an explicit, orchestrator-triggered action rather than
            # automatic on process boot (storm windows are short in real time).
            async with MCPBridge(mcp_url, orchestrator_recorder) as mcp:
                await mcp.call_tool("start_episode")
            print("[run_agentic] sim clock started; kicking off the episode...")
            await db.update_run(conn, run_id, phase="running", mcp_url=mcp_url)

            t_start = time.monotonic()
            result = await asyncio.wait_for(
                send_data_message(coordinator_url, {"kind": "start_episode", "data": {}}, orchestrator_recorder, "start_episode"),
                timeout=args.episode_timeout_s,
            )
            elapsed = time.monotonic() - t_start
            print(f"[run_agentic] episode finished in {elapsed:.1f}s wall-clock.")

        except (KeyboardInterrupt, asyncio.CancelledError):
            print("[run_agentic] stop requested.")
            await db.update_run(conn, run_id, phase="stopping")
            # don't re-raise: fall through to teardown + the "stopped" status
            # write below, same as the "episode did not complete" path.
        except Exception as exc:  # noqa: BLE001
            await db.update_run(conn, run_id, phase="failed", error=str(exc))
            raise
        finally:
            print("[run_agentic] tearing down...")
            if args.backend == "subprocess":
                for proc in procs:
                    proc.terminate()
                for proc in procs:
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
            else:
                for proc in log_tail_procs:
                    proc.terminate()
                docker_down(docker_env)
            for f in log_files:
                f.close()

        if result is None:
            print("[run_agentic] episode did not complete; skipping report. Check logs under", log_dir)
            await db.update_run(conn, run_id, phase="stopped")
            return

        report = await db.build_overhead_report(conn, run_id)
        storms = await db.get_storms(conn, run_id)
        resilience = storms[-1]["resilience"] if storms else None

        print()
        print(format_report(report, resilience=resilience, storms=storms))
        print(f"\n[run_agentic] full report stored in PostgreSQL under run_id={run_id}")
        await db.update_run(conn, run_id, phase="done")


def main() -> None:
    def _on_sigterm(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _on_sigterm)
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("[run_agentic] stopped.")


if __name__ == "__main__":
    main()
