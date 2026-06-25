"""`python -m agents.tactical --port 9002 --mcp-url http://127.0.0.1:8800/mcp --coordinator-url http://127.0.0.1:9001 --run-id demo`"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from agents.base.a2a_app import build_agent_app, make_agent_card, run_agent_process
from agents.base.llm import build_reasoner
from instrumentation.recorder import InstrumentedRecorder

from .executor import SYSTEM_PROMPT, TacticalExecutor, default_decision_fn, prompt_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--advertise-host", default=None,
        help="Address other agents should dial to reach this one, if different from --host "
             "(e.g. --backend docker binds 0.0.0.0 but must advertise the service name "
             "'tactical' in its agent card). Defaults to --host.",
    )
    p.add_argument("--port", type=int, default=9002)
    p.add_argument("--mcp-url", default="http://127.0.0.1:8800/mcp")
    p.add_argument("--coordinator-url", default="http://127.0.0.1:9001")
    p.add_argument("--poll-interval-s", type=float, default=3.0)
    p.add_argument("--llm-provider", choices=["stub", "ollama", "anthropic", "openai"], default="ollama")
    p.add_argument("--llm-model", default="llama3.2:latest")
    p.add_argument("--llm-base-url", default="http://localhost:11434")
    p.add_argument("--stub-latency-min", type=float, default=0.1)
    p.add_argument("--stub-latency-max", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=3)
    p.add_argument("--run-id", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    recorder = InstrumentedRecorder(owner="tactical", run_id=args.run_id)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    reasoner = build_reasoner(
        args.llm_provider, args.llm_model, args.llm_base_url,
        prompt_fn=prompt_fn, fallback_fn=default_decision_fn, system_prompt=SYSTEM_PROMPT,
        stub_latency_range_s=(args.stub_latency_min, args.stub_latency_max), seed=args.seed,
        recorder=recorder,
    )
    if hasattr(reasoner, "warmup"):
        print(f"[tactical] warming up {args.llm_provider}:{args.llm_model}...", flush=True)
        asyncio.run(reasoner.warmup())
        print("[tactical] warmup done.", flush=True)
    executor = TacticalExecutor(
        mcp_url=args.mcp_url,
        coordinator_url=args.coordinator_url,
        recorder=recorder,
        reasoner=reasoner,
        poll_interval_s=args.poll_interval_s,
    )
    card = make_agent_card(
        name="Tactical",
        description="Near-RT agent: trigger-driven absorption + adaptation + recovery scale-down via MCP tools.",
        host=args.host,
        advertise_host=args.advertise_host,
        port=args.port,
        skill_id="run_episode",
        skill_description="Runs the trigger-poll loop for one episode against the simulator's MCP server.",
    )
    app = build_agent_app(executor, card, recorder)
    run_agent_process(app, args.host, args.port)


if __name__ == "__main__":
    main()
