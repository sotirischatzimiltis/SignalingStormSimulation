"""`python -m agents.strategic --port 9003 --mcp-url http://127.0.0.1:8800/mcp --run-id demo`"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys

from agents.base.a2a_app import build_agent_app, make_agent_card, run_agent_process
from agents.base.llm import build_reasoner
from instrumentation.recorder import InstrumentedRecorder

from .executor import SYSTEM_PROMPT, StrategicExecutor, default_decision_fn, prompt_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument(
        "--advertise-host", default=None,
        help="Address other agents should dial to reach this one, if different from --host "
             "(e.g. --backend docker binds 0.0.0.0 but must advertise the service name "
             "'strategic' in its agent card). Defaults to --host.",
    )
    p.add_argument("--port", type=int, default=9003)
    p.add_argument("--mcp-url", default="http://127.0.0.1:8800/mcp")
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
    recorder = InstrumentedRecorder(owner="strategic", run_id=args.run_id)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    signal.signal(signal.SIGINT, lambda *_: sys.exit(0))

    reasoner = build_reasoner(
        args.llm_provider, args.llm_model, args.llm_base_url,
        prompt_fn=prompt_fn, fallback_fn=default_decision_fn, system_prompt=SYSTEM_PROMPT,
        stub_latency_range_s=(args.stub_latency_min, args.stub_latency_max), seed=args.seed,
        recorder=recorder,
    )
    if hasattr(reasoner, "warmup"):
        print(f"[strategic] warming up {args.llm_provider}:{args.llm_model}...", flush=True)
        asyncio.run(reasoner.warmup())
        print("[strategic] warmup done.", flush=True)
    executor = StrategicExecutor(mcp_url=args.mcp_url, recorder=recorder, reasoner=reasoner, run_id=args.run_id)
    card = make_agent_card(
        name="Strategic",
        description="Non-RT agent: once-per-episode recovery validation + memory write.",
        host=args.host,
        advertise_host=args.advertise_host,
        port=args.port,
        skill_id="validate_recovery",
        skill_description="Computes the episode's resilience score from final telemetry and writes a memory file.",
    )
    app = build_agent_app(executor, card, recorder)
    run_agent_process(app, args.host, args.port)


if __name__ == "__main__":
    main()
