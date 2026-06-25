# Operations: how to run anything in this repo

A practical runbook -- one place to look up "how do I actually run X" without
digging through the architecture docs. For *why* things are built the way
they are, see `README.md` (overview) and the per-module docs it links to.

All commands below assume you're in the project root with the venv active:

```bash
cd /path/to/SignalingStormSimulation
source .venv/bin/activate
```

First time only: `pip install -r requirements.txt` (Python 3.10+).

**Also required for anything agentic** (not the plain-simulator commands):
a PostgreSQL container, since every run's structured data (config/status,
every MCP/A2A/LLM call, every storm's evolution record) lives there now --
see "0. Database setup" below. This is new and mandatory regardless of
which process-supervision backend you use (subprocess or docker).

---

## Quick reference

| What you want | Command |
|---|---|
| Reproduce the baseline resilience ladder (no agents, ~seconds) | `python -m scripts.compare_baselines` |
| Generate comparison plots (no agents) | `python -m scripts.plot_compare` |
| Live slider GUI over the plain simulator (no agents) | `python -m scripts.gui` |
| Start Postgres (once, leave running) | `docker compose up -d postgres` |
| Run one agentic episode from the CLI | `python -m scripts.run_agentic [flags]` |
| ...with the agents containerized instead of bare processes | `python -m scripts.run_agentic --backend docker [flags]` |
| Run/watch/compare agentic episodes in a browser | `streamlit run scripts/dashboard.py` |
| Query the database directly | `psql postgresql://stormsim:stormsim@localhost:5433/stormsim` |

---

## 0. Database setup (required before any agentic run)

```bash
docker compose up -d postgres
```

Starts a dedicated Postgres container (port **5433**, not the default 5432
-- deliberately separate from anything else already running on your
machine) with a named volume, so data survives container restarts. Leave it
running across multiple `run_agentic.py` invocations -- it's a persistent
service, not part of the per-episode spawn/teardown cycle either backend
uses. Schema (`runs`/`calls`/`llm_calls`/`storms` tables) is created
automatically on first connection (`instrumentation/db.py: ensure_schema()`)
-- no manual migration step.

Check it's up: `docker compose ps` should show `postgres` as `healthy`.
See `README_database.md` for the schema and example queries.

---

## 1. Baseline comparison and plots (no agents, no LLMs, instant)

```bash
python -m scripts.compare_baselines   # prints the resilience ladder (text)
python -m scripts.plot_compare        # writes plots/summary.png + plots/timeseries.png
```

These only exercise `sim/` (the discrete-event simulator + baseline
controllers) -- no MCP server, no agents, nothing to start or stop. Good
sanity check that the environment is set up correctly; both finish in a few
seconds.

## 2. Live slider GUI (plain simulator, no agents)

```bash
python -m scripts.gui
```

Opens a matplotlib window with sliders to manually drive the simulator in
real time (server count, malicious drop probability). Useful for building
intuition about the storm dynamics. Has nothing to do with the agentic
layer -- no MCP, no A2A, no LLMs.

## 3. One agentic episode from the CLI

```bash
python -m scripts.run_agentic
```

Spawns the MCP server + the three A2A agents (coordinator/tactical/
strategic) as real subprocesses, runs one episode end to end, tears
everything down, and prints the combined resilience + overhead report
(queried back from PostgreSQL once the episode completes).

**Key flags** (`python -m scripts.run_agentic --help` for the full list):

| Flag | Default | What it does |
|---|---|---|
| `--backend` | `subprocess` | or `docker` -- runs mcp_server + the 3 agents as Docker Compose services instead of bare processes; see "3b. Docker backend" below |
| `--scenario` | `single_storm` | or `multi_storm` (3 storms, exercises the evolution loop -- see `README_agents.md`) |
| `--rt-factor` | `1.0` | wall-clock seconds per simulated second. `1.0` = true real time (~17 min for single_storm, ~18 min for multi_storm). Lower compresses time but starves a real LLM's decision latency -- use `0.05`-`0.1` only with `--llm-provider stub` for a fast (~1 min) smoke test. **Don't combine a low `--rt-factor` with a real LLM provider**: if process startup/warmup eats into the pre-storm window in real time, Tactical's first telemetry poll can land *after* the storm has already started, which corrupts its self-detected baseline arrival rate and silently skips onset/subside detection (escalation still fires off the queue threshold, but no storm gets recorded) -- this is a pre-existing timing sensitivity in the onset-detection design, not a database/Docker issue |
| `--llm-provider` | `ollama` | or `stub` (deterministic rules, no model call -- fast, reproducible), `anthropic`, `openai` (need `ANTHROPIC_API_KEY`/`OPENAI_API_KEY` in your environment) |
| `--llm-base-url` | `http://localhost:11434` | Ollama daemon URL (only matters for `--llm-provider ollama`; auto-rewritten to `http://host.docker.internal:11434` under `--backend docker` unless you override it) |
| `--coordinator-model` / `--tactical-model` / `--strategic-model` | `llama3.2:latest` | per-agent model override |
| `--seed` | `3` | simulator RNG seed |
| `--poll-interval-s` | `2.0` | how often Tactical polls telemetry (real seconds) |
| `--run-id` | auto (`run_<timestamp>`) | the database key for this run; also names `instrumentation/_runs/<run_id>/logs/` for log tailing |

```bash
# fast smoke test, no Ollama needed (~1 min)
python -m scripts.run_agentic --llm-provider stub --rt-factor 0.05

# real demo with local Ollama, single storm (~17 min)
python -m scripts.run_agentic

# real demo, multi-storm evolution loop (~18 min)
python -m scripts.run_agentic --scenario multi_storm

# same fast smoke test, but containerized
python -m scripts.run_agentic --llm-provider stub --rt-factor 0.05 --backend docker
```

**Prerequisite for `--llm-provider ollama` (the default):** a local Ollama
daemon must be running with the model pulled, e.g. `ollama pull
llama3.2:latest`. Check it's reachable: `curl http://localhost:11434/api/tags`.
If it's not running, either start it or pass `--llm-provider stub`.

**Stopping it:** `Ctrl+C` in the terminal it's running in. This is a real
signal handler (not just a process kill) -- it cleanly tears down all 4
subprocesses (or containers) before exiting, same as a normal completion.

## 3b. Docker backend

```bash
docker compose up -d postgres            # once, if not already running
python -m scripts.run_agentic --backend docker [flags]
```

Runs mcp_server + the 3 agents as Docker Compose services (`docker-compose.yml`,
one shared image built from the root `Dockerfile`) instead of bare OS
processes -- everything else about the command (flags, output, teardown) is
identical to the subprocess backend. `run_agentic.py` itself always stays on
the host (it's the thing driving `docker compose` for the docker backend);
only mcp_server/coordinator/tactical/strategic get containerized.

What's different under the hood, if you're curious or debugging:
- Each agent binds `0.0.0.0` inside its container (required to accept
  connections from other containers) but talks to its peers using their
  Compose service names (`mcp_server`, `coordinator`, `tactical`,
  `strategic`) baked into `docker-compose.yml`'s `command:` overrides --
  not `127.0.0.1`.
- The host-side orchestrator (`run_agentic.py`) still reaches every
  container via `127.0.0.1:<port>`, since Compose publishes the same ports
  to the host that the subprocess backend would have bound directly.
- `agents/base/a2a_app.py`'s `send_data_message()` overwrites whatever URL
  a peer's agent card self-reports with the URL *the caller itself* used to
  reach it -- without this, the a2a-sdk client dials the peer's
  self-reported address for the actual message send (even though the
  initial agent-card fetch uses the caller's own URL), which breaks the
  moment bind-address and reachable-address differ, as they must under
  Docker.
- A real local Ollama daemon (running on the host, not in a container) is
  reachable from inside the containers via `host.docker.internal:11434` --
  no need to containerize Ollama or duplicate the model download.
- `docker compose down` (scoped to just the 4 ephemeral services, never
  `postgres`) runs automatically on teardown, same guarantee as the
  subprocess backend's process termination.

`scripts/dashboard.py`'s sidebar has a **Backend** selector (subprocess/docker)
that passes straight through to this same flag.

## 4. Dashboard (browser, start/stop/compare experiments interactively)

```bash
streamlit run scripts/dashboard.py
```

Streamlit prints a `Local URL` (default `http://localhost:8501`) and opens
it in your browser automatically. Sidebar lets you configure the same flags
as the CLI above (**Backend** subprocess/docker, scenario, rt_factor,
provider, models, seed...) and click **Start experiment**, which spawns
`scripts/run_agentic.py` for you. Two tabs:
- **Live Run** -- live telemetry charts, Tactical's trigger/decision log
  (and the LLM call log, if using a real provider -- prompts, responses,
  token counts), the per-storm evolution chart for multi-storm runs, and the
  final report once it completes.
- **Compare Runs** -- every completed run in the database in one table,
  with charts across whichever runs you select.

**Stopping a run:** click **Stop current run** in the sidebar (sends a real
SIGTERM that the orchestrator catches and uses to cleanly tear down its 4
subprocesses -- same guarantee as Ctrl+C above). Only one experiment can run
at a time (the MCP server and agents bind fixed ports), so **Start** is
disabled while a run is active.

**Stopping the dashboard itself:** `Ctrl+C` in the terminal running
`streamlit run`.

**If you close the browser tab or kill the terminal while a run is
active:** the orchestrator and its 4 subprocesses are independent OS
processes -- they keep running until the episode finishes on its own. If you
want to force-stop everything immediately instead:
```bash
lsof -ti:8800,9001,9002,9003 | xargs kill -9   # mcp_server, coordinator, tactical, strategic
lsof -ti:8501 | xargs kill -9                  # the dashboard itself, if needed
```

## 5. Where results end up

**PostgreSQL** (`docker compose up -d postgres`, port 5433) is the source of
truth for everything structured -- see `README_database.md` for the full
schema and example queries. In brief, keyed by `run_id`:
- `runs` -- current phase (`starting`/`ready`/`running`/`done`/`stopped`/
  `failed`) and the full config it ran with; what the dashboard polls live.
- `calls` -- every instrumented MCP/A2A call (latency, byte counts, success).
- `llm_calls` -- every real-provider LLM call in full (system/user prompt,
  raw response, parsed decision, token counts, latency) -- empty for
  `--llm-provider stub` runs.
- `storms` -- one row per storm: resilience score, classification, the
  policy it ran with, and the policy proposed for the next storm (the
  evolution record).

`instrumentation/_runs/<run_id>/logs/<agent>.log` is the one thing still on
disk -- each agent's full stdout (trigger detections, LLM call summaries,
policy updates...), tailed live by the dashboard. Nothing else under
`instrumentation/_runs/` is written anymore (no `status.json`, `report.json`,
or per-process `<agent>.json` dumps); `memory/episode_<run_id>.json` is also
gone, replaced by the `storms` table.

## 6. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `run_agentic.py` fails immediately with a connection/`asyncpg` error | Postgres isn't running -- `docker compose up -d postgres`, then confirm `docker compose ps` shows it `healthy`. |
| `run_agentic.py` hangs at "waiting for readiness" | A previous run's processes (or, under `--backend docker`, containers -- `docker compose ps`) are still bound to the ports. Force-clean (see above, or `docker compose down mcp_server coordinator tactical strategic` for the docker backend) and retry. |
| Dashboard's **Start** button does nothing / errors immediately | Check `instrumentation/_runs/<run_id>/orchestrator_stdout.log` for the actual error. |
| `--backend docker` run fails with a container port/network error mentioning `postgres` | A *second* Postgres got created under a different Compose project name. Always start Postgres via the project's default `docker compose up -d postgres` (no `-p` override) -- `run_agentic.py` deliberately never passes `-p` for exactly this reason, so the agent containers' `depends_on: postgres` resolves to the same persistent instance. |
| All decisions show `"_source": "fallback"` with `--llm-provider ollama` | Ollama isn't reachable at `--llm-base-url` (under `--backend docker`, this should auto-resolve to `host.docker.internal:11434`), the model isn't pulled, or the call timed out -- check `curl http://localhost:11434/api/tags` and the agent's `.log` file for the `[llm] ... error=...` line. The system is designed to fall back to deterministic rules rather than crash, so this fails silently into a working (if less interesting) run unless you check the logs. |
| A run completes but the database shows zero rows in `storms` even though escalation/decision activity is visible in the logs | Tactical's onset-detection baseline got captured *after* the storm already started (see the `--rt-factor` warning above) -- usually means rt_factor was set too low for a real (non-stub) LLM provider's startup/warmup overhead. Re-run with `--rt-factor 1.0` or `--llm-provider stub`. |
| Streamlit shows a stale dashboard after I edited `dashboard.py` | Streamlit's default file watcher is polling-based and can lag; restart it (`Ctrl+C` then re-run) to be sure you're on the latest code. |
| `compare_baselines.py` output doesn't match the numbers in `README.md` | Something changed in `sim/`. This should never happen from agent/dashboard work -- if it does, it's a real regression, not expected drift. |
