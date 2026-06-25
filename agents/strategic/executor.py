"""
Strategic (Non-RT): once per storm, recovery validation. Reads the full
telemetry history via MCP, computes the existing resilience_score()
(sim/metrics.py, untouched -- pure post-processing, no MCP needed for the
math itself) over that storm's self-detected (onset_t, subside_t) window,
and writes/updates a memory file under memory/.

Evolution: this is also where Strategic decides the next storm's policy
(escalation_threshold, drop_prob_floor) -- tightened when absorption was
weak (the queue grew too much before Tactical reacted), held otherwise --
relative to `policy_used`, the policy Tactical reports it actually ran with
for that storm (Tactical is the sole source of truth for its own policy;
Strategic never keeps a shadow copy). single_storm_traffic() is just the
N=1 case of the same path: one storm_complete, one storms-table row, one
(no-op, episode already over) policy_update. The per-storm record is
written straight to PostgreSQL's `storms` table (instrumentation/db.py) --
one INSERT per storm replaces the old memory/episode_<run_id>.json
read-modify-write file.
"""

from __future__ import annotations

from a2a.helpers import new_data_message, new_task_from_user_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater

from agents.base.a2a_app import get_request_data
from agents.base.mcp_client import MCPBridge
from agents.base.reason import Reasoner
from instrumentation import db
from instrumentation.recorder import InstrumentedRecorder
from sim.metrics import UtilityParams, resilience_score
from sim.simulator import TelemetrySample

LQ_MAX = 1500.0  # matches scripts/compare_baselines.py and mcp_server/__main__.py
DEFAULT_ESCALATION_THRESHOLD = 40  # fallback only -- policy_used normally carries the real value
ABSORPTION_WEAK = 0.7
THRESHOLD_TIGHTEN_FACTOR = 0.8
THRESHOLD_MIN = 10.0
FLOOR_RAISE_STEP = 0.2


def default_decision_fn(context: dict) -> dict:
    r = context["resilience"]
    policy = context.get("policy_used") or {}
    threshold = policy.get("escalation_threshold", DEFAULT_ESCALATION_THRESHOLD)
    floor = policy.get("drop_prob_floor", 0.0)
    if r["absorption"] < ABSORPTION_WEAK:
        threshold = max(THRESHOLD_MIN, threshold * THRESHOLD_TIGHTEN_FACTOR)
        floor = min(1.0, floor + FLOOR_RAISE_STEP)
    lessons = "recovered_within_threshold" if r["trec"] >= 1.0 else "recovery_slower_than_desired"
    return {
        "classification": "recovered" if r["P"] > 0.6 else "degraded",
        "lessons": lessons,
        "escalation_threshold": threshold,
        "drop_prob_floor": floor,
    }


def prompt_fn(context: dict) -> str:
    r = context["resilience"]
    policy = context.get("policy_used") or {}
    return (
        "You are the Non-RT strategic agent validating recovery after one storm in a "
        "signaling-storm episode. Resilience metrics: P={:.3f}, absorption={:.2f}, "
        "adaptation={:.2f}, trec={:.2f}, recovery_time={:.1f}s. Tactical's policy during this "
        "storm: escalation_threshold={}, drop_prob_floor={}. Classify the episode, note one "
        "lesson, and propose policy values for the NEXT storm: lower escalation_threshold "
        "and/or raise drop_prob_floor if absorption was weak (queue grew too much before "
        "Tactical reacted), otherwise keep them the same. Respond with ONLY this JSON, no "
        'other text: {{"classification": "recovered" or "degraded", "lessons": "<one short '
        'sentence>", "escalation_threshold": <number>, "drop_prob_floor": <float 0.0-1.0>}}'
    ).format(
        r["P"], r["absorption"], r["adaptation"], r["trec"], r["recovery_time"],
        policy.get("escalation_threshold", DEFAULT_ESCALATION_THRESHOLD), policy.get("drop_prob_floor", 0.0),
    )


SYSTEM_PROMPT = (
    "You are a deliberate, analytical Non-RT network resilience agent. Always respond with a "
    "single JSON object matching the requested schema and nothing else -- no prose, no markdown."
)


class StrategicExecutor(AgentExecutor):
    def __init__(self, mcp_url: str, recorder: InstrumentedRecorder, reasoner: Reasoner, run_id: str):
        self.mcp_url = mcp_url
        self.recorder = recorder
        self.reasoner = reasoner
        self.run_id = run_id

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)

        req = get_request_data(context)
        if req.get("kind") != "validate_recovery":
            await updater.failed(message=new_data_message({"error": f"unknown kind {req.get('kind')!r}"}))
            return

        await updater.start_work()
        data = req.get("data", {})
        storm_index = data.get("storm_index", 1)
        t0, td = data["onset_t"], data["subside_t"]
        policy_used = data.get("policy_used") or {}

        async with MCPBridge(self.mcp_url, self.recorder) as mcp:
            payload = await mcp.call_tool("get_telemetry", last_n=1_000_000)

        samples = [TelemetrySample(**s) for s in payload["samples"]]
        mu_single = payload["mu_single"]
        util_p = UtilityParams(lq_max=LQ_MAX, kB=0.004)
        result = resilience_score(samples, mu_single, util_p, t0=t0, td=td)

        decision = await self.reasoner.reason(
            {"resilience": result, "policy_used": policy_used, "storm_index": storm_index}
        )
        new_policy = self._extract_policy(decision, policy_used)
        storm_record = {
            "storm_index": storm_index,
            "t0": t0,
            "td": td,
            "resilience": result,
            "classification": decision.get("classification"),
            "lessons": decision.get("lessons"),
            "decision_source": decision.get("_source"),
            "policy_before": policy_used,
            "policy_after": new_policy,
        }
        await db.insert_storm(
            self.recorder.pool, self.run_id, storm_index, t0, td, result,
            decision.get("classification"), decision.get("lessons"), decision.get("_source"),
            policy_used, new_policy,
        )

        # Coordinator accumulates this directly into its per-episode storms list (so it must
        # carry storm_index/policy_before/policy_after, not just be nested under "memory") and
        # also reads "policy" off it to forward as the next storm's policy_update.
        await updater.complete(message=new_data_message({**storm_record, "policy": new_policy}))

    @staticmethod
    def _extract_policy(decision: dict, policy_used: dict) -> dict:
        try:
            threshold = float(decision.get("escalation_threshold"))
        except (TypeError, ValueError):
            threshold = policy_used.get("escalation_threshold", DEFAULT_ESCALATION_THRESHOLD)
        try:
            floor = max(0.0, min(1.0, float(decision.get("drop_prob_floor"))))
        except (TypeError, ValueError):
            floor = policy_used.get("drop_prob_floor", 0.0)
        return {"escalation_threshold": threshold, "drop_prob_floor": floor}

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")
