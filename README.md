# Open RAN Signaling-Storm Resilience Simulator

A discrete-event simulator for studying **resilience to signaling storms** in
Open RAN, and the testbed for an agentic, lifecycle-spanning resilience
orchestrator: an MCP-driven simulator, a three-agent (coordinator/tactical/
strategic) system coordinating over A2A, real pluggable LLM decision
policies, PostgreSQL-backed run history, and an optional Docker Compose
deployment.

It rebuilds and extends the model from *"Surviving the Storm: The Impacts of
Open RAN Disaggregation on Latency and Resilience"* (arXiv:2505.00605): instead
of an analytical M/M/c queue, it simulates **individual UEs** attaching through
the disaggregated control plane, with **explicit RRC timers and retries** so
that storms amplify the way real ones do.

## Quick start

```bash
pip install -r requirements.txt
python -m scripts.compare_baselines        # no agents, no database -- the resilience ladder in seconds

docker compose up -d postgres              # once, for everything below
python -m scripts.run_agentic --llm-provider stub --rt-factor 0.05   # fast end-to-end smoke test
streamlit run scripts/dashboard.py         # browser UI: configure/start/watch/compare runs
```

## Documentation

- [`ReadMe/README.md`](ReadMe/README.md) -- full architecture and design rationale
- [`ReadMe/README_operations.md`](ReadMe/README_operations.md) -- practical runbook: every command, every flag, troubleshooting
- [`ReadMe/README_agents.md`](ReadMe/README_agents.md) -- the coordinator/tactical/strategic agents, LLM backends, evolution loop
- [`ReadMe/README_mcp_server.md`](ReadMe/README_mcp_server.md) -- the MCP server's tools/resources
- [`ReadMe/README_database.md`](ReadMe/README_database.md) -- PostgreSQL schema and example queries

## License

MIT -- see [`LICENSE`](LICENSE).
