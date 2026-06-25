"""
Configuration for the Open RAN signaling-storm simulator.

Delay / overhead values are reconstructed from:
  Chatzimiltis et al., "Surviving the Storm: The Impacts of Open RAN
  Disaggregation on Latency and Resilience", arXiv:2505.00605.

Reverse-engineered from the paper so the rebuilt discrete-event sim
reproduces the published service rates:
  - M (CU-handled control messages in RRC setup) = 3
  - total internal processing  sum(t_p,i)         = 30 ms   (default)
  - one-way RU->BBU delay (monolithic, Split 7)   = 0.25 ms
  - one-way RU->CU  delay (Open RAN, 7 + 2)        = 1.75 ms (0.25 + 1.5)
=> service time monolithic = 30 + 3*0.25 = 30.75 ms -> mu = 32.52 UEs/s
=> service time Open RAN   = 30 + 3*1.75 = 35.25 ms -> mu = 28.37 UEs/s
"""

from dataclasses import dataclass, field
from typing import List, Literal


# ---------------------------------------------------------------------------
# Architecture / delay model
# ---------------------------------------------------------------------------
@dataclass
class ArchConfig:
    """Per-message control-plane delay accounting for the UE attach procedure."""
    n_ctrl_messages: int = 3          # M : CU-handled RRC msgs (Setup Req/Setup/Setup Complete)
    proc_total_ms: float = 30.0       # sum_i t_p,i : total internal processing (Table VII row "30")
    oneway_delay_ms: float = 1.75     # RU->CU (Open RAN). Use 0.25 for monolithic.

    def service_time_ms(self) -> float:
        """Mean service time of one full attach attempt (ms)."""
        return self.proc_total_ms + self.n_ctrl_messages * self.oneway_delay_ms

    def service_rate(self) -> float:
        """Per-server service rate mu (UEs/s)."""
        return 1000.0 / self.service_time_ms()


def open_ran_arch(**kw) -> ArchConfig:
    return ArchConfig(oneway_delay_ms=1.75, **kw)


def monolithic_arch(**kw) -> ArchConfig:
    return ArchConfig(oneway_delay_ms=0.25, **kw)


# ---------------------------------------------------------------------------
# RRC timer / retry behaviour  (the realism upgrade: storm amplification)
# ---------------------------------------------------------------------------
@dataclass
class RRCConfig:
    """
    Explicit RRC connection-setup timer (T300-like) and retry policy.

    A UE that does not complete attachment within `t300_ms` abandons the
    current attempt and retries, up to `max_attempts`. Under overload this
    creates the self-amplifying retry load that defines a signaling storm.
    """
    t300_ms: float = 1000.0           # RRC setup guard timer (T300). 3GPP allows 100..2000 ms.
    max_attempts: int = 5             # attempts before the UE gives up (failure)
    backoff_ms: float = 0.0           # extra wait before a retry (0 = immediate re-attach)


# ---------------------------------------------------------------------------
# Traffic: benign baseline + malicious botnet, multi-storm timeline
# ---------------------------------------------------------------------------
@dataclass
class StormPhase:
    """A piecewise-constant segment of the arrival-rate timeline (seconds)."""
    t_start: float
    t_end: float
    benign_rate: float                # benign UE arrivals/s
    botnet_rate: float = 0.0          # malicious UE arrivals/s (repeated attach)
    label: str = ""


@dataclass
class TrafficConfig:
    phases: List[StormPhase] = field(default_factory=list)

    def horizon(self) -> float:
        return max((p.t_end for p in self.phases), default=0.0)

    def rates_at(self, t: float):
        """Return (benign_rate, botnet_rate) active at time t."""
        for p in self.phases:
            if p.t_start <= t < p.t_end:
                return p.benign_rate, p.botnet_rate
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# Top-level simulation config
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    arch: ArchConfig = field(default_factory=open_ran_arch)
    rrc: RRCConfig = field(default_factory=RRCConfig)
    traffic: TrafficConfig = field(default_factory=TrafficConfig)
    c0: int = 1                       # initial number of servers
    c_max: int = 16                   # max servers the actuator may allocate
    lq_max: float = 7000.0            # queue length at which utility uB -> 0
    sample_dt_s: float = 0.5          # telemetry sampling interval (s)
    seed: int = 0
    # malicious UEs are flagged in telemetry; a rate-limit actuator may drop them
    botnet_attach_period_ms: float = 200.0   # how often a botnet UE re-attaches when admitted
    # --- real-time pacing ---
    # realtime=False -> run as fast as possible (virtual time, for experiments).
    # realtime=True  -> pace the sim clock to wall-clock time.
    #   rt_factor = wall-clock seconds per simulated second:
    #     1.0 -> true real time, 0.1 -> 10x faster, 2.0 -> 2x slower (nice for a GUI)
    realtime: bool = False
    rt_factor: float = 1.0
    # --- shared-compute contention (load-dependent processing time) ---
    # Models the vCU/vDU running on a finite shared compute pool. When enabled,
    # per-attach PROCESSING time inflates by the processor-sharing factor
    # 1/(1 - rho_c), where rho_c = (busy workers)/compute_kappa. The F1/O-FH
    # propagation delay is unaffected.
    #   compute_kappa = None -> contention OFF (recovers the paper's numbers)
    #   compute_kappa = K    -> compute pool can run ~K attach-workers at full speed
    #   compute_rho_cap      -> clamp rho_c < 1 to avoid the infinite pole
    compute_kappa: float = None
    compute_rho_cap: float = 0.98
    # --- server provisioning delay ---
    # Time to bring a newly commanded vDU/vCU server online (image pull / boot /
    # pool attach). New servers come up one at a time, this many seconds apart.
    # 0.0 (default) = instant (preserves the paper's behaviour). Scaling down is
    # always immediate (no preemption of in-flight attaches).
    server_provision_delay_s: float = 0.0


# ---------------------------------------------------------------------------
# Convenience scenario builders
# ---------------------------------------------------------------------------
def single_storm_traffic(normal=20.0, storm=200.0,
                         t_pre=50.0, t_storm=60.0, t_post=900.0) -> TrafficConfig:
    """The prior paper's scenario: 20 -> 200 -> 20 UEs/s."""
    return TrafficConfig(phases=[
        StormPhase(0.0, t_pre, normal, 0.0, "pre"),
        StormPhase(t_pre, t_pre + t_storm, storm, 0.0, "storm"),
        StormPhase(t_pre + t_storm, t_pre + t_storm + t_post, normal, 0.0, "recovery"),
    ])


def multi_storm_traffic() -> TrafficConfig:
    """
    Three storms of growing intensity with a malicious component, used to
    demonstrate the evolution stage (the system should handle storm 3 better
    than storm 1 after the strategic agent updates its policy between storms).
    """
    return TrafficConfig(phases=[
        StormPhase(0,    60,   20, 0,   "calm-1"),
        StormPhase(60,   120,  120, 40, "storm-1"),
        StormPhase(120,  360,  20, 0,   "recover-1"),
        StormPhase(360,  420,  20, 0,   "calm-2"),
        StormPhase(420,  480,  180, 60, "storm-2"),
        StormPhase(480,  720,  20, 0,   "recover-2"),
        StormPhase(720,  780,  20, 0,   "calm-3"),
        StormPhase(780,  840,  220, 80, "storm-3"),
        StormPhase(840, 1100,  20, 0,   "recover-3"),
    ])
