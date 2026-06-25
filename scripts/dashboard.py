"""
Streamlit dashboard for the agentic resilience-orchestration system: start an
experiment, watch it run live (telemetry, agent decisions, resilience score),
and compare past runs.

Run from the project root:
    streamlit run scripts/dashboard.py

This is a process *supervisor* + *viewer*, not a reimplementation of the
orchestration logic: clicking "Start" spawns `python -m scripts.run_agentic`
as one subprocess (which itself spawns mcp_server + the three agents, exactly
as it does from the CLI). The dashboard never talks to the simulator's
actuators directly -- it only reads `telemetry://latest` / `sim://status` via
its own short-lived MCP connections (the same read-only resources any
viewer is free to poll) and tails each agent's log file. Only one experiment
can run at a time (mcp_server/agents bind fixed ports), which is also why
Start is disabled while a run is active.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.base.mcp_client import MCPBridge  # noqa: E402
from instrumentation import db  # noqa: E402
from instrumentation.recorder import InstrumentedRecorder  # noqa: E402
from sim.config import multi_storm_traffic, single_storm_traffic  # noqa: E402
from sim.metrics import UtilityParams, utility  # noqa: E402
from sim.simulator import TelemetrySample  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "instrumentation" / "_runs"  # logs/*.log only -- everything else is in PostgreSQL now
MCP_URL = "http://127.0.0.1:8800/mcp"
LQ_MAX = 1500.0
UTIL_PARAMS = UtilityParams(lq_max=LQ_MAX, kB=0.004)
SCENARIO_HORIZON = {"single_storm": single_storm_traffic().horizon(), "multi_storm": multi_storm_traffic().horizon()}

st.set_page_config(page_title="Signaling-Storm Agentic Dashboard", layout="wide")


# ---------------------------------------------------------------------------
# Data access -- PostgreSQL (instrumentation/db.py) for everything structured,
# logs/<agent>.log files (still plain text) for live log tailing only.
# ---------------------------------------------------------------------------
async def _get_run(run_id: str) -> dict:
    async with db.connection() as conn:
        return await db.get_run(conn, run_id) or {}


def load_status(run_id: str) -> dict:
    try:
        return asyncio.run(_get_run(run_id))
    except Exception:  # noqa: BLE001 - Postgres unreachable / run not found yet
        return {}


async def _build_report(run_id: str) -> dict:
    async with db.connection() as conn:
        run = await db.get_run(conn, run_id)
        if not run or run.get("phase") != "done":
            return {}
        overhead = await db.build_overhead_report(conn, run_id)
        storms = await db.get_storms(conn, run_id)
        resilience = storms[-1]["resilience"] if storms else None
        return {"resilience": resilience, "storms": storms, "overhead": overhead}


def load_report(run_id: str) -> dict:
    try:
        return asyncio.run(_build_report(run_id))
    except Exception:  # noqa: BLE001
        return {}


def tail_log(run_id: str, name: str, n: int = 30) -> str:
    path = RUNS_DIR / run_id / "logs" / f"{name}.log"
    if not path.exists():
        return "(no log yet)"
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:]) or "(empty)"


async def _list_runs() -> list:
    async with db.connection() as conn:
        return await db.list_runs(conn)


def list_past_runs() -> list:
    try:
        return [r["run_id"] for r in asyncio.run(_list_runs())]
    except Exception:  # noqa: BLE001 - Postgres unreachable
        return []


# ---------------------------------------------------------------------------
# Live telemetry (the dashboard's own short-lived MCP reads -- independent
# of Tactical's own polling; just observing, never calling actuator tools)
# ---------------------------------------------------------------------------
async def _fetch_live(mcp_url: str) -> dict:
    recorder = InstrumentedRecorder(owner="_dashboard")
    async with MCPBridge(mcp_url, recorder) as mcp:
        status = await mcp.read_resource("sim://status")
        telemetry = await mcp.read_resource("telemetry://latest")
    return {"status": status, "telemetry": telemetry}


def fetch_live(mcp_url: str) -> dict:
    try:
        return asyncio.run(_fetch_live(mcp_url))
    except Exception as exc:  # noqa: BLE001 - mcp_server not up (yet/anymore)
        return {"status": None, "telemetry": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Process supervision
# ---------------------------------------------------------------------------
def start_run(run_id: str, cfg: dict) -> subprocess.Popen:
    args = [
        sys.executable, "-m", "scripts.run_agentic",
        "--run-id", run_id,
        "--backend", cfg["backend"],
        "--scenario", cfg["scenario"],
        "--rt-factor", str(cfg["rt_factor"]),
        "--c-max", str(cfg["c_max"]),
        "--seed", str(cfg["seed"]),
        "--poll-interval-s", str(cfg["poll_interval_s"]),
        "--llm-provider", cfg["llm_provider"],
        "--llm-base-url", cfg["llm_base_url"],
        "--coordinator-model", cfg["coordinator_model"],
        "--tactical-model", cfg["tactical_model"],
        "--strategic-model", cfg["strategic_model"],
    ]
    log_path = RUNS_DIR / run_id
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / "orchestrator_stdout.log", "w")
    return subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdout=log_file, stderr=subprocess.STDOUT)


def stop_run() -> None:
    proc = st.session_state.get("active_proc")
    if proc is not None and proc.poll() is None:
        proc.terminate()


# ---------------------------------------------------------------------------
# Sidebar: experiment configuration
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    st.sidebar.header("New experiment")
    active = st.session_state.get("active_run")

    backend = st.sidebar.selectbox(
        "Backend", ["subprocess", "docker"], disabled=bool(active),
        help="subprocess: bare OS processes against this venv (default). docker: the same 4 "
             "processes as Docker Compose services -- requires `docker compose up -d postgres` "
             "already running.",
    )
    scenario = st.sidebar.selectbox("Scenario", ["single_storm", "multi_storm"], disabled=bool(active))
    rt_factor = st.sidebar.number_input(
        "rt_factor (wall-sec per sim-sec)", min_value=0.01, max_value=2.0, value=1.0, step=0.05,
        disabled=bool(active),
        help="1.0 = true real time (~17 min for single_storm). Lower compresses time but can make "
             "even a fast LLM's latency dominate the storm's response window.",
    )
    llm_provider = st.sidebar.selectbox(
        "LLM provider", ["ollama", "stub", "anthropic", "openai"], disabled=bool(active),
    )
    llm_base_url = st.sidebar.text_input(
        "Ollama base URL", value="http://localhost:11434", disabled=bool(active) or llm_provider != "ollama",
    )
    with st.sidebar.expander("Per-agent model override", expanded=False):
        default_model = "llama3.2:latest" if llm_provider == "ollama" else (
            "claude-haiku-4-5" if llm_provider == "anthropic" else "gpt-4o-mini" if llm_provider == "openai" else "stub"
        )
        coordinator_model = st.text_input("Coordinator model", value=default_model, disabled=bool(active))
        tactical_model = st.text_input("Tactical model", value=default_model, disabled=bool(active))
        strategic_model = st.text_input("Strategic model", value=default_model, disabled=bool(active))

    seed = st.sidebar.number_input("seed", min_value=0, value=3, step=1, disabled=bool(active))
    poll_interval_s = st.sidebar.number_input(
        "Tactical poll interval (s)", min_value=0.5, value=2.0, step=0.5, disabled=bool(active),
    )
    c_max = st.sidebar.number_input("c_max", min_value=1, value=16, step=1, disabled=bool(active))
    run_id = st.sidebar.text_input("run_id", value=f"dash_{int(time.time())}", disabled=bool(active))

    if active:
        st.sidebar.info(f"Run **{active}** is active. Stop it to configure a new one.")
        if st.sidebar.button("Stop current run", type="primary"):
            stop_run()
    else:
        if st.sidebar.button("Start experiment", type="primary"):
            cfg = dict(
                backend=backend, scenario=scenario, rt_factor=rt_factor, c_max=c_max, seed=seed,
                poll_interval_s=poll_interval_s, llm_provider=llm_provider, llm_base_url=llm_base_url,
                coordinator_model=coordinator_model, tactical_model=tactical_model, strategic_model=strategic_model,
            )
            proc = start_run(run_id, cfg)
            st.session_state["active_run"] = run_id
            st.session_state["active_proc"] = proc
            st.session_state["telemetry_history"] = []
            st.rerun()


# ---------------------------------------------------------------------------
# Live Run tab
# ---------------------------------------------------------------------------
@st.fragment(run_every="2s")
def render_live_run() -> None:
    run_id = st.session_state.get("active_run")
    if not run_id:
        last = st.session_state.get("last_finished_run")
        if last:
            st.subheader(f"Last run: {last}")
            render_report(last)
        else:
            st.info("No active run. Configure one in the sidebar and click **Start experiment**.")
        return

    proc = st.session_state.get("active_proc")
    status = load_status(run_id)
    phase = status.get("phase", "unknown")
    proc_alive = proc is not None and proc.poll() is None

    st.subheader(f"Run: {run_id}")
    cols = st.columns(4)
    cols[0].metric("Phase", phase)
    cols[1].metric("Process", "alive" if proc_alive else "exited")

    if not proc_alive:
        # Process has exited (successfully, stopped, or crashed) -- finalize and stop polling.
        st.session_state["active_run"] = None
        st.session_state["last_finished_run"] = run_id
        st.session_state["active_proc"] = None
        st.rerun()
        return

    live = fetch_live(status.get("mcp_url", MCP_URL))
    telemetry = live.get("telemetry") or {}
    sim_status = live.get("status") or {}

    if telemetry:
        horizon = SCENARIO_HORIZON.get(status.get("config", {}).get("scenario", "single_storm"), 1010.0)
        t = telemetry.get("t", 0.0)
        cols[2].metric("Sim time", f"{t:.1f}s")
        cols[3].progress(min(1.0, t / horizon), text=f"{t:.0f}/{horizon:.0f} sim-s")

        sample = TelemetrySample(**telemetry)
        u = utility(sample, sim_status.get("mu_single", 28.37), UTIL_PARAMS)
        history = st.session_state.setdefault("telemetry_history", [])
        if not history or history[-1]["t"] != t:
            history.append({"t": t, "queue_len": telemetry["queue_len"], "servers (c)": telemetry["c"],
                             "arrival_rate": telemetry["lam_target"], "utility": u})
            st.session_state["telemetry_history"] = history[-600:]

        df = pd.DataFrame(st.session_state["telemetry_history"]).set_index("t")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Queue length / servers online")
            st.line_chart(df[["queue_len", "servers (c)"]])
        with c2:
            st.caption("Arrival rate / utility u(t)")
            st.line_chart(df[["arrival_rate", "utility"]])
    else:
        st.caption("Waiting for the sim clock to start / first telemetry sample...")

    with st.expander("Tactical agent activity (triggers + decisions)", expanded=True):
        st.code(tail_log(run_id, "tactical", 25), language=None)
    cexp1, cexp2 = st.columns(2)
    with cexp1.expander("Coordinator log"):
        st.code(tail_log(run_id, "coordinator", 15), language=None)
    with cexp2.expander("Strategic log"):
        st.code(tail_log(run_id, "strategic", 15), language=None)


def render_report(run_id: str) -> None:
    report = load_report(run_id)
    if not report:
        status = load_status(run_id)
        st.warning(f"No final report for **{run_id}** (phase: {status.get('phase', 'unknown')}). "
                   f"Check logs under `instrumentation/_runs/{run_id}/logs/`.")
        return
    resilience = report.get("resilience") or {}
    overhead = report.get("overhead") or {}
    by_channel = overhead.get("by_channel", {})

    cols = st.columns(5)
    cols[0].metric("P (resilience)", f"{resilience.get('P', 0):.3f}")
    cols[1].metric("absorption", f"{resilience.get('absorption', 0):.2f}")
    cols[2].metric("adaptation", f"{resilience.get('adaptation', 0):.2f}")
    cols[3].metric("trec", f"{resilience.get('trec', 0):.2f}")
    cols[4].metric("recovery_time", f"{resilience.get('recovery_time', 0):.0f}s")

    storms = report.get("storms") or []
    if len(storms) > 1:
        st.caption("Evolution across storms (does storm N score better than storm 1?)")
        evo = pd.DataFrame([
            {
                "storm": f"storm-{s['storm_index']}",
                "P": s["resilience"]["P"],
                "absorption": s["resilience"]["absorption"],
                "escalation_threshold": s["policy_after"]["escalation_threshold"],
                "drop_prob_floor": s["policy_after"]["drop_prob_floor"],
            }
            for s in storms
        ]).set_index("storm")
        ec1, ec2 = st.columns(2)
        with ec1:
            st.bar_chart(evo[["P", "absorption"]])
        with ec2:
            st.bar_chart(evo[["escalation_threshold", "drop_prob_floor"]])

    c1, c2 = st.columns(2)
    with c1:
        st.caption("MCP channel")
        mcp = by_channel.get("mcp", {})
        st.write(f"count={mcp.get('count', 0)}  mean_latency={mcp.get('mean_latency_s', 0)*1000:.1f}ms  "
                 f"bytes={mcp.get('total_bytes', 0)}")
    with c2:
        st.caption("A2A channel")
        a2a = by_channel.get("a2a", {})
        st.write(f"count={a2a.get('count', 0)}  mean_latency={a2a.get('mean_latency_s', 0)*1000:.1f}ms "
                 f"(dominated by the long-blocking start_episode call -- see agents/README.md)  "
                 f"bytes={a2a.get('total_bytes', 0)}")

    llm_calls = overhead.get("llm_calls") or []
    if llm_calls:
        llm_summary = overhead.get("llm_summary", {})
        st.caption(
            f"LLM calls: {llm_summary.get('count', 0)} "
            f"({llm_summary.get('fallback_count', 0)} fell back)  "
            f"tokens_in={llm_summary.get('total_tokens_in', 0)}  "
            f"tokens_out={llm_summary.get('total_tokens_out', 0)}  "
            f"mean_latency={llm_summary.get('mean_latency_s', 0):.2f}s"
        )
        with st.expander(f"LLM call log ({len(llm_calls)} calls -- prompt, response, tokens)"):
            llm_df = pd.DataFrame([
                {
                    "owner": c["owner"],
                    "provider": c["provider"],
                    "model": c["model"],
                    "source": c["source"],
                    "tokens_in": c["tokens_in"],
                    "tokens_out": c["tokens_out"],
                    "latency_s": round(c["latency_s"], 2),
                    "prompt_preview": (c["user_prompt"] or "")[:80],
                    "response_preview": (c["response_text"] or "")[:80],
                }
                for c in llm_calls
            ])
            st.dataframe(llm_df, width="stretch")
            idx = st.selectbox(
                "View full prompt/response for call #", range(len(llm_calls)),
                format_func=lambda i: f"{i}: {llm_calls[i]['owner']} ({llm_calls[i]['source']})",
            )
            chosen = llm_calls[idx]
            st.text_area("System prompt", chosen.get("system_prompt") or "", height=80, disabled=True)
            st.text_area("User prompt", chosen.get("user_prompt") or "", height=150, disabled=True)
            st.text_area("Raw response", chosen.get("response_text") or "(none -- exception, see error)",
                         height=100, disabled=True)
            st.json(chosen.get("decision") or {})
            if chosen.get("error"):
                st.error(chosen["error"])

    if storms:
        with st.expander("Strategic agent's per-storm memory (PostgreSQL `storms` table)"):
            st.json(storms)


# ---------------------------------------------------------------------------
# Compare Runs tab
# ---------------------------------------------------------------------------
def render_compare() -> None:
    run_ids = list_past_runs()
    rows = []
    for rid in run_ids:
        report = load_report(rid)
        status = load_status(rid)
        if not report:
            continue
        resilience = report.get("resilience") or {}
        overhead = report.get("overhead") or {}
        by_channel = overhead.get("by_channel", {})
        cfg = status.get("config", {})
        storms = report.get("storms") or []
        row = {
            "run_id": rid,
            "scenario": cfg.get("scenario"),
            "llm_provider": cfg.get("llm_provider"),
            "rt_factor": cfg.get("rt_factor"),
            "P": resilience.get("P"),
            "absorption": resilience.get("absorption"),
            "adaptation": resilience.get("adaptation"),
            "mcp_count": by_channel.get("mcp", {}).get("count"),
            "mcp_mean_lat_ms": round(by_channel.get("mcp", {}).get("mean_latency_s", 0) * 1000, 2),
            "a2a_count": by_channel.get("a2a", {}).get("count"),
        }
        if len(storms) > 1:
            p_first, p_last = storms[0]["resilience"]["P"], storms[-1]["resilience"]["P"]
            row.update({"n_storms": len(storms), "P_first": p_first, "P_last": p_last, "improved": p_last > p_first})
        rows.append(row)

    if not rows:
        st.info("No completed runs in the database yet. Finish an experiment in the Live Run tab first.")
        return

    df = pd.DataFrame(rows)
    st.dataframe(df, width="stretch")

    selected = st.multiselect("Runs to compare", df["run_id"].tolist(), default=df["run_id"].tolist()[:5])
    if selected:
        sub = df[df["run_id"].isin(selected)].set_index("run_id")
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Resilience P")
            st.bar_chart(sub["P"])
        with c2:
            st.caption("MCP call count")
            st.bar_chart(sub["mcp_count"])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.title("Open RAN Signaling-Storm Agentic Dashboard")
render_sidebar()

tab_live, tab_compare = st.tabs(["Live Run", "Compare Runs"])
with tab_live:
    render_live_run()
with tab_compare:
    render_compare()
