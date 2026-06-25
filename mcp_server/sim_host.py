"""
Owns the one live StormSim for an episode and bridges it safely across the
thread boundary between the sim's own SimPy loop and the MCP server's
asyncio/HTTP loop.

This reuses the exact pattern scripts/gui.py already established: the sim
runs on its own background thread (paced to wall-clock time via
RealtimeEnvironment when cfg.realtime=True); external callers (here, MCP tool
handlers running on the asyncio loop) never call sim.set_servers() /
sim.set_malicious_drop_prob() directly -- they only write a "commanded
target" into CommandState. A bridge controller, installed exactly like any
other controller via sim.run(controller=...), runs ON THE SIM'S OWN THREAD
and is the only thing that actually calls the actuators, polling
CommandState once per sample_dt_s tick. The poll delay is what makes a
late-arriving agent command show up as queue growth -- not an artifact to
work around, but the thing being measured.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import List, Optional

from sim.config import SimConfig
from sim.simulator import StormSim, TelemetrySample


@dataclass
class CommandState:
    """Plain-Python shared state. Scalar read/replace of immutable values
    needs no lock under CPython's GIL -- same assumption gui.py's SimRunner
    already relies on for its slider-driven attributes."""
    target_c: Optional[int] = None
    target_drop_prob: Optional[float] = None
    last_applied_c: Optional[int] = None
    last_applied_drop_prob: Optional[float] = None


class _BridgeController:
    """Same shape as gui.py's _gui_controller: step(sim, s) runs on the sim's
    own thread (invoked by sim._control_loop every sample_dt_s) and is the
    only code path that calls the real actuators."""

    def __init__(self, state: CommandState):
        self.state = state

    def step(self, sim: StormSim, s: TelemetrySample) -> None:
        if self.state.target_c is not None:
            sim.set_servers(self.state.target_c)
            self.state.last_applied_c = self.state.target_c
            self.state.target_c = None
        if self.state.target_drop_prob is not None:
            sim.set_malicious_drop_prob(self.state.target_drop_prob)
            self.state.last_applied_drop_prob = self.state.target_drop_prob
            self.state.target_drop_prob = None


class SimHost:
    """One per MCP-server process. Construct, then call start(); the sim
    runs to completion (the scenario's traffic horizon) on a daemon thread."""

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.sim = StormSim(cfg)
        self.state = CommandState()
        self._bridge = _BridgeController(self.state)
        self._thread: Optional[threading.Thread] = None
        self._done = threading.Event()
        self._started = threading.Event()

    def start(self) -> None:
        """Explicit trigger -- NOT called automatically on construction. The
        sim clock (especially under realtime=True) must not start ticking
        until the orchestrator has confirmed every process is up; otherwise
        the storm window can elapse in real time before agents are even
        wired up to watch it (single_storm_traffic's 60s storm window is
        only 12 real seconds at rt_factor=0.2)."""
        if self._started.is_set():
            return
        self._started.set()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_started(self) -> bool:
        return self._started.is_set()

    def _run(self) -> None:
        try:
            self.sim.run(controller=self._bridge)
        finally:
            self._done.set()

    def is_done(self) -> bool:
        return self._done.is_set()

    def wait_done(self, timeout: Optional[float] = None) -> bool:
        return self._done.wait(timeout=timeout)

    # -- read-only accessors, safe from any thread: sim.telemetry is an
    # append-only list of immutable dataclass instances; reading the tail (or
    # a slice) while the sim thread appends is safe under the GIL. ----------
    def latest_telemetry(self) -> Optional[TelemetrySample]:
        tel = self.sim.telemetry
        return tel[-1] if tel else None

    def telemetry_window(self, last_n: int) -> List[TelemetrySample]:
        tel = self.sim.telemetry
        return tel[-last_n:] if last_n > 0 else list(tel)

    def mu_single(self) -> float:
        return self.sim.mu_single

    # -- command-queueing API used by mcp_server/tools.py; never touches sim
    # actuators directly. -----------------------------------------------------
    def command_set_servers(self, c: int) -> None:
        self.state.target_c = c

    def command_set_drop_prob(self, p: float) -> None:
        self.state.target_drop_prob = p
