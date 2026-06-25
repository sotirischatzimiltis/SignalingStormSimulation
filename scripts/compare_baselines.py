"""
Reproduce the baseline resilience ladder (fixed vs. Lyapunov vs. forecast)
on the single-storm scenario. This is the bar the agentic system must beat.

Run from the project root:
    python -m scripts.compare_baselines
"""

from sim.config import SimConfig, open_ran_arch, RRCConfig, single_storm_traffic
from sim.simulator import StormSim
from sim.controllers import FixedController, LyapunovController, ForecastLyapunov
from sim.metrics import UtilityParams, resilience_score

LQMAX = 1500.0
T0, TD = 50.0, 110.0      # storm window for the single-storm scenario


def run(controller, label, c0=1, seed=3):
    cfg = SimConfig(
        arch=open_ran_arch(),
        rrc=RRCConfig(t300_ms=1000, max_attempts=5),
        c0=c0, c_max=16, lq_max=LQMAX,
        traffic=single_storm_traffic(), seed=seed,
    )
    sim = StormSim(cfg)
    sim.run(controller=controller)
    up = UtilityParams(lq_max=LQMAX, kB=0.004)
    r = resilience_score(sim.telemetry, sim.mu_single, up, t0=T0, td=TD)
    peakq = max(s.queue_len for s in sim.telemetry)
    avgc = sum(s.c for s in sim.telemetry) / len(sim.telemetry)
    print(f"{label:18s} P={r['P']:.3f}  absorb={r['absorption']:.2f} "
          f"adapt={r['adaptation']:.2f} trec={r['trec']:.2f} "
          f"recov={r['recovery_time']:4.0f}s  peakQ={peakq:5d}  avgC={avgc:.1f}  "
          f"fail={sim.stats.failed}")


if __name__ == "__main__":
    print("Single-storm resilience ladder (Open RAN, 20->200->20 UEs/s)\n")
    run(FixedController(1), "Fixed c=1")
    run(FixedController(2), "Fixed c=2", c0=2)
    run(FixedController(8), "Fixed c=8", c0=8)
    run(LyapunovController(V=1000, W=1), "Lyapunov V=1000")
    run(ForecastLyapunov(V=1000, W=1, horizon_s=8), "Lyap+Forecast")
