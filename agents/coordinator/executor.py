"""
Coordinator (SMO level): A2A hub only, no MCP. Holds operator intent and
routes the episode lifecycle:

  start_episode (from scripts/run_agentic.py)
      -> send run_episode to Tactical (fast ack -- Tactical's poll loop runs
         independently afterward)
      -> wait for Tactical's separate episode_complete message, then for any
         still-in-flight storm_complete handling (below) to finish
      -> complete the original start_episode task with the accumulated
         per-storm results ({"storms": [...]}, one entry per storm Tactical
         detected -- length 1 for single_storm_traffic())

  escalation (from Tactical, fired whenever it detects a trigger needing
      visibility)
      -> calls the reasoner (LLM or stub) for a logged approve/comment
         judgement. With a single agent path and a single scenario there is
         nothing yet for that judgement to actually override (no competing
         proposals, no guardrail threshold breached) -- real conflict
         resolution becomes meaningful once there's something to resolve.
         The LLM is genuinely consulted and its comment surfaced in the ack,
         but control flow doesn't yet branch on "approved".

  episode_complete (from Tactical)
      -> records the payload, sets the event the start_episode handler is
         waiting on, then awaits any still-in-flight storm_complete handling
         (below) before returning the accumulated per-storm results.

  storm_complete (from Tactical, once per storm -- self-detected onset/
      subside window, see agents/tactical/executor.py)
      -> acks immediately, handling the actual work
         (send validate_recovery to Strategic for that storm's window, then
         forward Strategic's proposed policy_update back to Tactical) as a
         background task -- mirrors Tactical's own ack-fast pattern, so
         Strategic's reasoning latency never blocks Tactical's poll loop.
         Tracked in self._pending_storm_tasks and gathered before
         start_episode's response is finalized.
"""

from __future__ import annotations

import asyncio

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.helpers import new_data_message

from agents.base.a2a_app import get_request_data, send_data_message
from agents.base.reason import Reasoner
from instrumentation.recorder import InstrumentedRecorder


def default_decision_fn(context: dict) -> dict:
    return {"approved": True, "comment": "auto-ack (no guardrail breached)"}


def prompt_fn(context: dict) -> str:
    intent = context["intent"]
    esc = context["escalation"]
    return (
        "You are the SMO-level coordinator for an Open RAN signaling-storm resilience system. "
        f"Operator intent: {intent}. The tactical agent reports an escalation: {esc}. "
        "Decide whether to approve its proposed action and give one short reason. Respond with "
        'ONLY this JSON, no other text: {"approved": true or false, "comment": "<one short sentence>"}'
    )


SYSTEM_PROMPT = (
    "You are a calm, decisive SMO-level orchestration agent. Always respond with a single JSON "
    "object matching the requested schema and nothing else -- no prose, no markdown."
)


class CoordinatorExecutor(AgentExecutor):
    def __init__(
        self, tactical_url: str, strategic_url: str, recorder: InstrumentedRecorder, reasoner: Reasoner
    ):
        self.tactical_url = tactical_url
        self.strategic_url = strategic_url
        self.recorder = recorder
        self.reasoner = reasoner
        self.operator_intent = {"priority": "resilience_over_cost"}
        self._episode_done = asyncio.Event()
        self._storm_results: list = []
        self._pending_storm_tasks: list = []

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.helpers import new_task_from_user_message
        from a2a.server.tasks import TaskUpdater

        task = context.current_task or new_task_from_user_message(context.message)
        if not context.current_task:
            await event_queue.enqueue_event(task)
        updater = TaskUpdater(event_queue=event_queue, task_id=task.id, context_id=task.context_id)

        req = get_request_data(context)
        kind = req.get("kind")

        if kind == "start_episode":
            await updater.start_work()
            self._episode_done.clear()
            await send_data_message(
                self.tactical_url,
                {"kind": "run_episode", "intent": self.operator_intent, **req.get("data", {})},
                self.recorder,
                "run_episode",
            )
            await self._episode_done.wait()
            if self._pending_storm_tasks:
                await asyncio.gather(*self._pending_storm_tasks, return_exceptions=True)
            await updater.complete(
                message=new_data_message({"status": "episode_done", "result": {"storms": self._storm_results}})
            )

        elif kind == "episode_complete":
            self._episode_done.set()
            await updater.complete(message=new_data_message({"status": "ack"}))

        elif kind == "storm_complete":
            data = req.get("data", {})
            self._pending_storm_tasks = [t for t in self._pending_storm_tasks if not t.done()]
            self._pending_storm_tasks.append(asyncio.create_task(self._handle_storm_complete(data)))
            await updater.complete(message=new_data_message({"status": "ack"}))

        elif kind == "escalation":
            decision = await self.reasoner.reason(
                {"intent": self.operator_intent, "escalation": req.get("data", {})}
            )
            await updater.complete(message=new_data_message({"status": "ack_escalation", "decision": decision}))

        else:
            await updater.failed(message=new_data_message({"error": f"unknown kind {kind!r}"}))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported")

    async def _handle_storm_complete(self, data: dict) -> None:
        result = await send_data_message(
            self.strategic_url,
            {"kind": "validate_recovery", "data": data},
            self.recorder,
            "validate_recovery",
        )
        self._storm_results.append(result)
        policy = result.get("policy") or {}
        if policy:
            await send_data_message(
                self.tactical_url,
                {"kind": "policy_update", "data": policy},
                self.recorder,
                "policy_update",
            )
