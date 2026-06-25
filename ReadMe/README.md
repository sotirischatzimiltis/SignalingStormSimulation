# Open RAN Signaling-Storm Resilience Simulator

A discrete-event simulator for studying **resilience to signaling storms** in
Open RAN, and the testbed for an agentic, lifecycle-spanning resilience
orchestrator.

It rebuilds and extends the model from *"Surviving the Storm: The Impacts of
Open RAN Disaggregation on Latency and Resilience"* (arXiv:2505.00605): instead
of an analytical M/M/c queue, it simulates **individual UEs** attaching through
the disaggregated control plane, with **explicit RRC timers and retries** so
that storms amplify the way real ones do. The server pool is reconfigurable at
runtime, so closed-loop controllers — and, next, a multi-agent system — can
intervene.

**Just want to run something?** See `README_operations.md` for a practical
runbook (every command, every flag, where results land, troubleshooting) --
this file covers the *why*, that one covers the *how*.

---

## Why this exists

The companion survey frames network resilience as a five-stage lifecycle:
**anticipation → absorption → adaptation → recovery → evolution**. The prior
paper demonstrated only absorption and adaptation, via a single reactive lever
(Lyapunov server scaling). This testbed is built to:

1. exercise **all five stages** on a concrete signaling-storm scenario;
2. drive them with a **multi-agent agentic system** (coordinator + tactical +
   strategic) communicating over **A2A**, acting on the sim via **MCP**;
3. show that coordination + learning beats standalone Lyapunov; and
4. **quantify the overhead** the agentic layer introduces.

---

## Layout

```
stormsim/
├── sim/
│   ├── config.py        # parameters + scenario builders        (see README_config.md)
│   ├── simulator.py     # SimPy discrete-event engine           (see README_simulator.md)
│   ├── metrics.py       # utility u(t) + resilience P           (see README_metrics.md)
│   └── controllers.py   # baseline controllers                  (see README_controllers.md)
├── mcp_server/           # MCP server wrapping the simulator     (see README_mcp_server.md)
├── agents/               # coordinator/tactical/strategic over A2A (see README_agents.md)
├── instrumentation/      # MCP/A2A/LLM call recorder + PostgreSQL access  (see README_database.md)
├── scripts/
│   ├── compare_baselines.py   # reproduces the resilience ladder (text)
│   ├── plot_compare.py        # comparison plots (time series + summary)
│   ├── gui.py                 # live real-time dashboard with sliders (plain simulator, no agents)
│   ├── run_agentic.py         # stands up mcp_server + the 3 agents, runs one episode
│   └── dashboard.py           # `streamlit run` web dashboard: start/watch/compare agentic runs
├── plots/                     # generated figures (created by plot_compare)
├── Dockerfile, docker-compose.yml   # optional containerized backend     (see README_operations.md)
├── requirements.txt
└── README.md            # this file
```

### How the pieces fit

```
config.py  ──►  simulator.py  ──►  telemetry stream  ──►  metrics.py  ──►  P, u(t)
                     ▲                                          
                     │ set_servers() / set_malicious_drop_prob()
                     │
              controllers.py   (fixed · Lyapunov · forecast · ‹agents next›)
```

`config` describes a run; `simulator` executes it and emits telemetry;
`metrics` scores that telemetry; `controllers` close the loop by acting on the
simulator each control step. The dependency arrows only point one way, so each
module can be read and tested in isolation.

---

## Install

```bash
cd stormsim
python -m venv .venv && source .venv/bin/activate     # optional
pip install -r requirements.txt                        # simpy, numpy, matplotlib
```
Requires Python 3.10+.

## Run a comparison

```bash
python -m scripts.compare_baselines
```

Expected output (single-storm scenario, Open RAN, 20 → 200 → 20 UEs/s):

```
Fixed c=1          P=0.588  ...  peakQ=1065  avgC=1.0  fail=12088
Fixed c=2          P=0.581  ...  peakQ=1055  avgC=2.0  fail=11670
Fixed c=8          P=0.796  ...  peakQ=  31  avgC=8.0  fail=0
Lyapunov V=1000    P=0.842  ...  peakQ=  60  avgC=1.4  fail=0
Lyap+Forecast      P=0.832  ...  peakQ=  54  avgC=1.4  fail=0
```

Reading it: fixed under-provisioning is overwhelmed (12k failures); fixed `c=8`
copes but burns 8 servers constantly; **Lyapunov is the best baseline** — it
matches fixed-8's stability with an average of 1.4 servers by scaling on demand.
That ~0.84 (and the efficiency gap) is the bar the agentic system must beat.

## Run the agentic demo

Requires PostgreSQL running first (`docker compose up -d postgres`, once --
see `README_operations.md` "Database setup"): every run's structured data
(config/status, every MCP/A2A/LLM call, every storm's evolution record)
lives there now, not in JSON files (see `README_database.md`).

```bash
python -m scripts.run_agentic                       # default: single_storm, real LLMs via local
                                                      # Ollama, rt_factor=1.0 (true real time, ~17 min)
python -m scripts.run_agentic --llm-provider stub    # deterministic rules, no model call (~17 min;
                                                      # use --rt-factor 0.05 with this for a ~50s smoke test)
python -m scripts.run_agentic --scenario multi_storm # 3 storms of growing intensity, one episode --
                                                      # exercises the evolution loop (see README_agents.md)
python -m scripts.run_agentic --backend docker       # same thing, mcp_server + the 3 agents containerized
                                                      # instead of bare processes (see README_operations.md)
```

Spawns the MCP server (`mcp_server/`) and the three A2A agents (`agents/`) as
real OS processes (or, with `--backend docker`, Docker Compose services),
waits for protocol-level readiness, starts the sim clock, runs one episode
end to end, tears everything down, and prints the combined resilience +
overhead report (queried back from PostgreSQL). See `README_mcp_server.md`
and `README_agents.md` for the architecture and LLM backend configuration
(local Ollama by default; pluggable to Anthropic/OpenAI via API key).

`rt_factor=1.0` is the default deliberately: a real local LLM's multi-second
decision latency needs a fair fraction of `single_storm_traffic()`'s 60s
storm window to land in — compressing time (lower `rt_factor`) makes even a
fast model's latency dominate a short window and was verified to erase the
agentic advantage entirely. A real run with real Ollama (`llama3.2:latest`)
decisions scored **P≈1.06** on `single_storm_traffic()` — above both the
Lyapunov baseline (0.84) and a stub-policy run (0.86) — with ~2000
instrumented MCP calls (single-digit-ms latency) and 6 A2A messages. See
`README_agents.md` for why P can exceed 1.0 (not a bug) and a real
ordering bug this surfaced (the fast Lyapunov lever was briefly gated behind
the slow LLM call before the fix).

## Dashboard: run and watch experiments interactively

```bash
streamlit run scripts/dashboard.py
```

A browser-based control panel for the agentic system: configure backend
(subprocess/docker) / scenario / `rt_factor` / LLM provider+model / seed in
the sidebar and click **Start experiment** to launch it (spawns
`scripts/run_agentic.py` as a subprocess, exactly like the CLI above); watch
live telemetry (queue length, servers online, arrival rate, utility),
Tactical's trigger/decision log, and the final resilience + overhead report
as the episode completes. A **Compare Runs** tab lists every completed run
in PostgreSQL for side-by-side comparison (e.g. stub vs. LLM-backed, or
different `rt_factor`/models). The dashboard only *watches* — it reads the
same `telemetry://latest`/`sim://status` MCP resources any client may poll
and tails each agent's log file; it never calls an actuator tool directly.
Only one experiment runs at a time (fixed ports), so **Start** is disabled
while a run is active.

## Plots

```bash
python -m scripts.plot_compare      # writes plots/timeseries.png and plots/summary.png
```
`timeseries.png` overlays queue length, utility, server count, and arrival rate
over time for each controller; `summary.png` shows resilience `P` and average
servers used (the efficiency view).

## Live GUI

```bash
python -m scripts.gui               # run on a machine with a display
```
A real-time dashboard: the simulator runs in real time on a background thread,
four plots update live, and sliders let you change the benign/botnet arrival
rates, server count, malicious-drop fraction, and T300 timer **while it runs**.
Useful for building intuition (e.g. start a storm by pushing the arrival rate
past `c·µ`, then try to rescue it by adding servers vs. raising T300). Needs an
interactive matplotlib backend (TkAgg/QtAgg); not for headless machines.

## Real-time vs. virtual time

By default the simulator runs in **virtual time** (as fast as the CPU allows) —
correct for experiments. Set `realtime=True, rt_factor=1.0` in `SimConfig` to
pace the clock to wall-clock time (`rt_factor` = wall-seconds per sim-second;
`0.1` = 10x faster, `2.0` = 2x slower). Real-time mode is what the GUI uses and
what the agentic system will use so that agent decision latency is incurred
against the live clock.

## Minimal API example

```python
from sim.config import SimConfig, open_ran_arch, RRCConfig, single_storm_traffic
from sim.simulator import StormSim
from sim.controllers import LyapunovController
from sim.metrics import UtilityParams, resilience_score

cfg = SimConfig(arch=open_ran_arch(), rrc=RRCConfig(t300_ms=1000, max_attempts=5),
                c0=1, c_max=10, lq_max=1500, traffic=single_storm_traffic(), seed=3)
sim = StormSim(cfg)
sim.run(controller=LyapunovController(V=1000, W=1))

up = UtilityParams(lq_max=1500, kB=0.004)
print(resilience_score(sim.telemetry, sim.mu_single, up, t0=50, td=110)["P"])
```

---

## Validation

The rebuild is checked against the prior paper before any new results:

| Check | Target (paper) | This sim |
|---|---|---|
| Service rate, monolithic | 32.52 UEs/s | 32.52 |
| Service rate, Open RAN | 28.37 UEs/s | 28.37 |
| Table VII, proc=10 ms | 93.02 / 65.57 | 93.02 / 65.57 |
| Table VII, proc=50 ms | 19.70 / 18.09 | 19.70 / 18.10 |
| Mean delay W at ρ=0.5 (Open RAN) | 70.52 ms | ≈70.7 ms |

Beyond the analytical match, the discrete-event layer adds **retry
amplification** (timeouts → retries → more load), which the prior M/M/c model
could not represent and which is what makes a storm a *storm*.

---

## Scenarios

- `single_storm_traffic()` — the prior paper's 20 → 200 → 20 UEs/s spike.
- `multi_storm_traffic()` — three storms of growing intensity **with a malicious
  botnet** component, for the **evolution** experiment (storm 3 handled better
  than storm 1 after the strategic agent learns between episodes).

---

## Key calibration choices (transparent, adjustable)

- `c_max` — kept tight on purpose so the agent must **rate-limit** the botnet
  rather than brute-force it with servers (the honest way to show the agentic
  advantage; otherwise "more compute" wins trivially).
- `compute_kappa` (optional, default off) — a *principled* version of the same
  constraint: shared-compute contention inflates per-attach processing by the
  processor-sharing factor `1/(1-rho_c)`, so adding servers shows diminishing
  then retrograde returns (peak near `c=kappa/2`) and load reduction wins. Off
  by default so the paper's numbers are exactly reproduced; turn it on for the
  richer, more realistic regime.
- `server_provision_delay_s` (default 0) — new vDU/vCU servers come online
  gradually after a scale-up command, so capacity lags reaction. Larger values
  reward anticipation (pre-provisioning before the storm).
- `lq_max=1500`, `kB=0.004` — utility congestion sensitivity, matched to the
  queue depths this regime actually reaches.
- `u_des` auto-calibrated to the **pre-storm baseline utility** rather than a
  flat 1.0 — the system's healthy steady state is ≈0.83, so a flat target would
  make recovery undetectable. Consistent with eq. (8)'s "ideal conditions."

---

## Roadmap

- [x] Discrete-event simulator with RRC timer/retry amplification
- [x] Utility + A3RT resilience scoring
- [x] Baseline controllers (fixed, Lyapunov, Lyapunov+forecast)
- [x] Validation against the prior paper
- [x] Comparison plots and a live real-time GUI
- [x] Real-time pacing (`realtime`/`rt_factor`) — the clock foundation for the agentic layer
- [x] **MCP server** exposing telemetry (resources) + actuators, the Lyapunov solver, and a
      forecaster (tools) — `mcp_server/`, see `README_mcp_server.md`
- [x] **Three parallel A2A agents**: coordinator (SMO), tactical (Near-RT), strategic (Non-RT) —
      `agents/`, see `README_agents.md`.
- [x] Real-time co-simulation harness so agent A2A/MCP latency is incurred and charged as overhead —
      `scripts/run_agentic.py`
- [x] **Real decision policies**: pluggable LLM backends (local Ollama by default, Anthropic/OpenAI
      via API key) behind the `Reasoner` seam (`agents/base/llm.py`), with deterministic
      stub-fallback on any provider failure — see `README_agents.md`. Decision *quality* is still
      basic prompting on a small local model; not yet tuned.
- [x] **Interactive dashboard** (`streamlit run scripts/dashboard.py`): configure/start/stop a run,
      watch live telemetry + agent decisions, compare past runs' resilience/overhead.
- [x] **Evolution**: Strategic's per-storm memory feeds back into Tactical within one
      `multi_storm_traffic()` episode (`storm_complete`/`policy_update` A2A messages; self-detected
      onset/subside windows, no hardcoded scenario boundaries) — tightens `escalation_threshold`/
      `drop_prob_floor` when a storm's absorption was weak, holds otherwise. See "Evolution" in
      `README_agents.md`. PostgreSQL's `storms` table now holds one row per storm.
- [x] **PostgreSQL persistence**: every run's config/status, MCP/A2A calls, LLM call content, and
      storm evolution records live in PostgreSQL (`instrumentation/db.py`) instead of per-process
      JSON dumps — see `README_database.md`.
- [x] **Docker Compose backend**: mcp_server + the 3 agents can run as containers
      (`--backend docker`) instead of bare subprocesses, alongside a persistent Postgres container
      — see `README_operations.md` "Docker backend". The bare subprocess backend (default) is
      unaffected and needs no Docker at all beyond the now-mandatory Postgres container.
- [ ] Multi-storm experiments with a real (non-stub) LLM backend, and a deliberately weakened
      baseline policy to demonstrate the tightening actually firing end-to-end (the stub policy's
      Lyapunov scaling is strong enough on its own that absorption hasn't dropped below the 0.7
      trigger in testing so far — the tightening logic itself is unit-verified, just not yet
      exercised by a "natural" live run)
- [ ] Tighten `c_max` (~9) for the final ablation comparison so Lyapunov can't brute-force the
      botnet component of `multi_storm_traffic()` — only matters once decision quality is real;
      the demo above still uses the existing `c_max=16` default so `compare_baselines.py` stays
      exactly reproducible
- [ ] Tactical's onset-detection baseline (`baseline_lam`, captured from its first telemetry poll)
      can be corrupted if that first poll lands after a storm has already started — found while
      verifying the database migration: combining a real (non-stub) LLM provider with a low
      `--rt-factor` made this likely enough to reproduce consistently (no `storms` rows recorded
      despite escalation triggering correctly). Not a regression from the database/Docker work;
      `--rt-factor 1.0` with real providers (already the documented default, see "Run the agentic
      demo" above) avoids it. A real fix would capture the baseline from telemetry observed
      *before* `start_episode`, or hold a fixed pre-storm reference window.

---

## Known docs cleanup (not yet done)

There are stale duplicate `README*.md` files directly in the project root
(pre-dating the `realtime`/`compute_kappa`/`server_provision_delay_s` config
additions). The current, maintained docs are the ones in this `ReadMe/`
folder — don't trust the root-level copies.

---

## Reference

Reconstructed delay model, utility function, and resilience metric from
S. Chatzimiltis, M. Shojafar, M. Boloursaz Mashhadi, R. Tafazolli,
*"Surviving the Storm: The Impacts of Open RAN Disaggregation on Latency and
Resilience,"* arXiv:2505.00605.
