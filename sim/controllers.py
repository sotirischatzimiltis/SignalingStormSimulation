"""
Baseline controllers acting on the simulator's server count c(t).

- FixedController        : constant c (the prior paper's c in {1,2,4,6,...})
- LyapunovController     : drift-plus-penalty optimum c(t) (eqs. 10-14)
- ForecastLyapunov       : Lyapunov + a 1-step arrival-rate forecast (anticipation
                           ablation rung: same single lever, but pre-warms).

These run inside the simulator's control loop (sim._control_loop), invoked every
sample_dt_s. The agentic system will later replace/augment these via MCP tools.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import UtilityParams, utility
from .simulator import StormSim, TelemetrySample


class FixedController:
    def __init__(self, c: int):
        self.c = c

    def step(self, sim: StormSim, s: TelemetrySample):
        sim.set_servers(self.c)


@dataclass
class LyapunovController:
    """
    Chooses c(t) minimising the drift-plus-penalty objective (eq. 14):

      min_c  Lq*(lam - c*mu) + 0.5*(lam - c*mu)^2 - V*u + W*c
      s.t.   1 <= c <= c_max,  c integer

    Solved by integer search over the feasible range each step.
    """
    V: float = 1000.0
    W: float = 1.0
    util_p: UtilityParams = None

    def __post_init__(self):
        if self.util_p is None:
            self.util_p = UtilityParams()

    def _objective(self, sim, s, c, lam):
        mu = sim.mu_single
        drift = s.queue_len * (lam - c * mu) + 0.5 * (lam - c * mu) ** 2
        # utility evaluated as if c were applied (queue/lam from current sample)
        probe = TelemetrySample(**{**s.__dict__, "c": c})
        u = utility(probe, mu, self.util_p)
        return drift - self.V * u + self.W * c

    def _lambda_estimate(self, sim, s):
        return s.lam_target

    def step(self, sim: StormSim, s: TelemetrySample):
        lam = self._lambda_estimate(sim, s)
        best_c, best_obj = sim.c, float("inf")
        for c in range(1, sim.cfg.c_max + 1):
            obj = self._objective(sim, s, c, lam)
            if obj < best_obj:
                best_obj, best_c = obj, c
        sim.set_servers(best_c)


@dataclass
class ForecastLyapunov(LyapunovController):
    """Lyapunov but using a 1-step-ahead arrival-rate forecast (pre-warming)."""
    horizon_s: float = 5.0

    def _lambda_estimate(self, sim, s):
        # peek the traffic schedule horizon_s into the future (idealised forecast;
        # in the agentic system this is replaced by a real predictor MCP tool)
        b, m = sim.cfg.traffic.rates_at(s.t + self.horizon_s)
        return max(s.lam_target, b + m)
