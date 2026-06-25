"""
The pluggable decision seam. Every agent executor owns exactly one Reasoner
and calls `await self.reasoner.reason(context)` at the one point where a real
LLM call will go in a later milestone. StubReasoner stands in for that call
now: it injects a randomized async latency (standing in for real inference
latency -- gotcha #1: this routinely exceeds the Near-RT control budget) but
returns a deterministic decision given the same context, so a demo run stays
reproducible end-to-end.
"""

from __future__ import annotations

import asyncio
import random
from typing import Callable, Protocol, Tuple


class Reasoner(Protocol):
    async def reason(self, context: dict) -> dict:
        """context: structured inputs (telemetry snapshot, trigger type,
        operator intent, peer messages...). Returns an agent-specific
        decision dict. The only place a future LLM call would go."""
        ...


class StubReasoner:
    def __init__(
        self,
        decision_fn: Callable[[dict], dict],
        latency_range_s: Tuple[float, float] = (0.1, 0.5),
        seed: int = None,
    ):
        self.decision_fn = decision_fn
        self.latency_range_s = latency_range_s
        self.rng = random.Random(seed)

    async def reason(self, context: dict) -> dict:
        await asyncio.sleep(self.rng.uniform(*self.latency_range_s))
        return self.decision_fn(context)
