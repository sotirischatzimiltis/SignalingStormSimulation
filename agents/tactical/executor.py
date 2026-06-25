"""
Tactical (Near-RT): trigger-driven, owns absorption + adaptation + executes
recovery scale-down. The only agent on a tight(ish) poll loop -- but that
loop runs at `poll_interval_s` (seconds, real time), deliberately MUCH
slower than the simulator's own `sample_dt_s` (0.5s) control tick, per
gotcha #1: LLM-class decision latency exceeds the Near-RT budget, so no
agent reasons every tick. The fast deterministic loop inside the simulator
(the bridge controller) runs continuously regardless; Tactical only
intervenes on triggers.

`run_episode` (from Coordinator) is acked immediately; the actual work runs
as an independent background asyncio task (`_run_episode_loop`), so
Coordinator is never blocked waiting on the whole episode through this call.

Triggers (evaluated client-side from polled telemetry -- MCP has no
push/subscribe primitive):
  - onset:      lam_target rises far above the pre-storm baseline
  - escalation: queue_len exceeds a threshold (debounced -- the A2A
                escalation message to Coordinator is rate-limited so a
                sustained overload doesn't spam it every poll)
  - subside:    lam_target has returned near baseline for several
                consecutive polls -- this is also "recovery scale-down":
                re-querying lyapunov_solve with the now-lower lam_target
                naturally proposes a smaller c, so it is not a separate
                decision branch this milestone.

On every trigger, Tactical immediately calls the Lyapunov solver tool (the
existing drift-plus-penalty math, called not re-derived) for the adaptation
lever -- this never waits on the reasoner. On escalation specifically, it
also asks the reasoner (stub rule or real LLM) for the rate-limiting lever
-- this is the lever Lyapunov alone does not have -- as a background task,
so a multi-second LLM call never delays the fast deterministic lever or the
next poll.

Evolution (multi-storm): onset/subside are self-detected reactively (no
ground-truth scenario boundaries read), so the observed (onset_t, subside_t)
window is reported to Coordinator as `storm_complete` once a storm's subside
holds -- this is also what lets Strategic score that storm and propose a new
`policy_update` for the next one. `self._policy` (escalation_threshold,
drop_prob_floor) is the live, mutable policy Tactical owns and reports
alongside each storm_complete (the sole source of truth for what was
actually active); `policy_update` messages from Coordinator replace it
wholesale for the next storm. Like the rate-limit decision, `storm_complete`
is sent as a background task -- Strategic's reasoning must never delay the
poll loop, which is calm by definition right after a subside.
"""

from __future__ import annotations

import asyncio
import time

from a2a.helpers import new_data_message, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from agents.base.a2a_app import get_request_data, send_data_message
from agents.base.mcp_client import MCPBridge
from agents.base.reason import Reasoner
from instrumentation.recorder import InstrumentedRecorder

ESCALATION_QUEUE_THRESHOLD = 40  # default; live value lives in self._policy, tunable via policy_update
DROP_PROB_FLOOR_DEFAULT = 0.0
ESCALATION_DEBOUNCE_S = 10.0
ONSET_RATIO = 1.5
SUBSIDE_RATIO = 1.2
SUBSIDE_HOLD_POLLS = 3


def default_decision_fn(context: dict) -> dict:
    """Deterministic fallback policy -- used directly when --llm-provider
    stub is selected, and as the safety net an LLMReasoner falls back to if
    the model is unreachable or returns unparseable output. Only the
    escalation trigger engages the rate-limiting lever (and only because
    that's the lever Lyapunov doesn't have -- single_storm_traffic() has no
    botnet, so this mostly exercises the plumbing there; the payoff scenario
    is multi_storm_traffic(), where escalation actually fires)."""
    trigger = context["trigger"]
    if trigger == "escalation":
        return {"drop_prob": 0.3}
    return {"drop_prob": 0.0}


def prompt_fn(context: dict) -> str:
    trigger = context["trigger"]
    t = context["telemetry"]
    return (
        "You are the Near-RT tactical agent absorbing a signaling storm in an Open RAN "
        f"control plane. Trigger detected: {trigger}. Current telemetry: arrival_rate="
        f"{t['lam_target']:.1f} UEs/s, queue_len={t['queue_len']}, servers_online={t['c']}, "
        f"sim_time={t['t']:.1f}s. Decide the fraction of malicious traffic to rate-limit "
        "(drop_prob). Use a higher value when queue_len is large and the trigger is "
        "'escalation'; use 0.0 for 'onset' or 'subside'. Respond with ONLY this JSON, "
        'no other text: {"drop_prob": <float between 0.0 and 1.0>}'
    )


SYSTEM_PROMPT = (
    "You are a fast, terse Near-RT network control agent. Always respond with a single "
    "JSON object matching the requested schema and nothing else -- no prose, no markdown."
)


class TacticalExecutor(AgentExecutor):
    def __init__(
        self,
        mcp_url: str,
        coordinator_url: str,
        recorder: InstrumentedRecorder,
        reasoner: Reasoner,
        poll_interval_s: float = 3.0,
    ):
        self.mcp_url = mcp_url
        self.coordinator_url = coordinator_url
        self.recorder = recorder
        self.reasoner = reasoner
        self.poll_interval_s = poll_interval_s
        self._poll_task = None
        self._policy = {
            "escalation_threshold": ESCALATION_QUEUE_THRESHOLD,
            "drop_prob_floor": DROP_PROB_FLOOR_DEFAULT,
        }

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)

        req = get_request_data(context)
        kind = req.get("kind")
        if kind == "run_episode":
            await updater.start_work()
            if self._poll_task is None or self._poll_task.done():
                self._poll_task = asyncio.create_task(self._run_episode_loop())
            await updater.complete(message=new_data_message({"status": "started"}))
        elif kind == "policy_update":
            data = req.get("data", {})
            for key in ("escalation_threshold", "drop_prob_floor"):
                if key in data:
                    self._policy[key] = data[key]
            print(f"[tactical] policy_update applied: {self._policy}", flush=True)
            await updater.complete(message=new_data_message({"status": "ack", "policy": dict(self._policy)}))
        else:
            await updater.failed(message=new_data_message({"error": f"unknown kind {kind!r}"}))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")

    async def _apply_rate_limit_decision(self, mcp: MCPBridge, trigger: str, telemetry: dict) -> None:
        decision = await self.reasoner.reason({"trigger": trigger, "telemetry": telemetry})
        print(f"[tactical] decision={decision}", flush=True)
        try:
            drop_prob = float(decision.get("drop_prob"))
            drop_prob = max(0.0, min(1.0, drop_prob))
        except (TypeError, ValueError):
            return
        drop_prob = max(drop_prob, self._policy["drop_prob_floor"])
        await mcp.call_tool("set_malicious_drop_prob", p=drop_prob)

    async def _send_storm_complete(self, storm_index: int, onset_t: float, subside_t: float) -> None:
        await send_data_message(
            self.coordinator_url,
            {
                "kind": "storm_complete",
                "data": {
                    "storm_index": storm_index,
                    "onset_t": onset_t,
                    "subside_t": subside_t,
                    "policy_used": dict(self._policy),
                },
            },
            self.recorder,
            "storm_complete",
        )

    async def _run_episode_loop(self) -> None:
        async with MCPBridge(self.mcp_url, self.recorder) as mcp:
            baseline_lam = None
            in_storm = False
            subside_streak = 0
            last_escalation_t = -1e9
            latest = {}
            pending_decisions: list = []
            pending_storm_sends: list = []
            storm_index = 0
            onset_t = None

            while True:
                status = await mcp.read_resource("sim://status")
                if status.get("is_done"):
                    break
                latest = await mcp.read_resource("telemetry://latest")
                if not latest:
                    # sim started but hasn't produced its first sample yet
                    await asyncio.sleep(self.poll_interval_s)
                    continue

                lam = latest["lam_target"]
                queue_len = latest["queue_len"]
                t = latest["t"]
                if baseline_lam is None:
                    baseline_lam = max(lam, 1.0)

                # onset/subside bookkeeping is evaluated unconditionally, independent of
                # whether escalation ALSO fires this same poll -- a severe storm can blow the
                # queue past the escalation threshold in the very first poll after onset, and
                # an if/elif chain with escalation checked first would silently skip marking
                # in_storm (and therefore never detect that storm's subside either) for
                # exactly the storms violent enough to matter most.
                storm_event = None
                if not in_storm and lam > ONSET_RATIO * baseline_lam:
                    in_storm = True
                    onset_t = t
                    storm_event = "onset"
                elif in_storm and lam <= SUBSIDE_RATIO * baseline_lam:
                    subside_streak += 1
                    if subside_streak >= SUBSIDE_HOLD_POLLS:
                        storm_event = "subside"
                        in_storm = False
                        subside_streak = 0
                else:
                    subside_streak = 0

                trigger = "escalation" if queue_len > self._policy["escalation_threshold"] else storm_event

                if storm_event == "subside" and onset_t is not None:
                    storm_index += 1
                    pending_storm_sends.append(
                        asyncio.create_task(self._send_storm_complete(storm_index, onset_t, t))
                    )
                    onset_t = None

                if trigger is not None:
                    masked = f" (storm_event={storm_event})" if storm_event and storm_event != trigger else ""
                    print(
                        f"[tactical] trigger={trigger}{masked} sim_t={t:.1f} lam={lam:.1f} "
                        f"queue_len={queue_len} c={latest.get('c')}",
                        flush=True,
                    )
                    # Adaptation (servers) is deterministic and fast (the existing Lyapunov
                    # solver, no LLM involved) -- apply it immediately. It must NOT wait on
                    # the LLM-backed rate-limit decision below, which can take seconds; gating
                    # the fast lever behind the slow one defeats the point of having a fast
                    # lever at all.
                    lyap = await mcp.call_tool("lyapunov_solve")
                    await mcp.call_tool("set_servers", c=lyap["c"])
                    if trigger == "escalation" and (t - last_escalation_t) >= ESCALATION_DEBOUNCE_S:
                        last_escalation_t = t
                        await send_data_message(
                            self.coordinator_url,
                            {"kind": "escalation", "data": {"t": t, "queue_len": queue_len, "proposed_c": lyap["c"]}},
                            self.recorder,
                            "escalation",
                        )
                    # Absorption (rate-limiting) is the LLM-backed decision -- only worth
                    # reasoning about on escalation (onset/subside always resolve to 0.0 per
                    # the fallback policy, so don't burn an LLM call on them). Runs as a
                    # background task so its latency doesn't delay the next poll either; await
                    # all pending ones before the MCP session closes below. Capped so a
                    # sustained overload (many escalations in a row) can't pile up unbounded
                    # concurrent LLM calls against a single shared local model.
                    if trigger == "escalation" and len(pending_decisions) < 3:
                        pending_decisions = [t for t in pending_decisions if not t.done()]
                        pending_decisions.append(
                            asyncio.create_task(self._apply_rate_limit_decision(mcp, trigger, latest))
                        )

                await asyncio.sleep(self.poll_interval_s)

            if in_storm and onset_t is not None:
                # Episode ended mid-storm (subside never confirmed in time) --
                # still report it so every onset gets a matching storm_complete.
                storm_index += 1
                pending_storm_sends.append(
                    asyncio.create_task(self._send_storm_complete(storm_index, onset_t, latest.get("t")))
                )

            if pending_decisions:
                await asyncio.gather(*pending_decisions, return_exceptions=True)
            if pending_storm_sends:
                await asyncio.gather(*pending_storm_sends, return_exceptions=True)

            await send_data_message(
                self.coordinator_url,
                {"kind": "episode_complete", "data": {"final_t": latest.get("t"), "wall_t": time.time()}},
                self.recorder,
                "episode_complete",
            )
