"""
Generate comparison plots across controllers on the single-storm scenario.

    python -m scripts.plot_compare

Produces:
    plots/timeseries.png  - queue, utility, servers, arrival rate over time
    plots/summary.png     - resilience P and average servers per controller
"""

import os
import matplotlib
matplotlib.use("Agg")           # headless: write PNGs, no display needed
import matplotlib.pyplot as plt

from sim.config import SimConfig, open_ran_arch, RRCConfig, single_storm_traffic
from sim.simulator import StormSim
from sim.controllers import FixedController, LyapunovController, ForecastLyapunov
from sim.metrics import UtilityParams, utility_series, resilience_score

LQMAX = 1500.0
T0, TD = 50.0, 110.0
UP = UtilityParams(lq_max=LQMAX, kB=0.004)

CONTROLLERS = [
    ("Fixed c=1",       lambda: FixedController(1),              1),
    ("Fixed c=8",       lambda: FixedController(8),              8),
    ("Lyapunov",        lambda: LyapunovController(V=1000, W=1), 1),
    ("Lyap+Forecast",   lambda: ForecastLyapunov(V=1000, W=1, horizon_s=8), 1),
]


def run_one(make_ctrl, c0, seed=3):
    cfg = SimConfig(arch=open_ran_arch(), rrc=RRCConfig(t300_ms=1000, max_attempts=5),
                    c0=c0, c_max=16, lq_max=LQMAX, traffic=single_storm_traffic(), seed=seed)
    sim = StormSim(cfg)
    sim.run(controller=make_ctrl())
    return sim


def main():
    os.makedirs("plots", exist_ok=True)
    runs = {}
    for label, make_ctrl, c0 in CONTROLLERS:
        sim = run_one(make_ctrl, c0)
        ts = [s.t for s in sim.telemetry]
        runs[label] = {
            "t": ts,
            "queue": [s.queue_len for s in sim.telemetry],
            "c": [s.c for s in sim.telemetry],
            "lam": [s.lam_target for s in sim.telemetry],
            "u": utility_series(sim.telemetry, sim.mu_single, UP),
            "P": resilience_score(sim.telemetry, sim.mu_single, UP, T0, TD)["P"],
            "avgc": sum(s.c for s in sim.telemetry) / len(sim.telemetry),
        }

    # ---- Figure 1: time series ----
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("Controller comparison - single signaling storm (Open RAN, 20->200->20 UEs/s)",
                 fontsize=13)
    panels = [
        (ax[0][0], "queue", "Queue length (waiting UEs)", "log"),
        (ax[0][1], "u",     "Utility u(t)", None),
        (ax[1][0], "c",     "Active servers c(t)", None),
        (ax[1][1], "lam",   "Arrival rate lambda(t) (UEs/s)", None),
    ]
    for a, key, title, yscale in panels:
        for label in runs:
            a.plot(runs[label]["t"], runs[label][key], label=label, linewidth=1.6)
        a.axvspan(T0, TD, color="orange", alpha=0.15, label="storm")
        a.set_title(title)
        a.set_xlabel("time (s)")
        if yscale:
            a.set_yscale(yscale)
        a.grid(True, alpha=0.3)
    ax[0][0].legend(fontsize=8, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig("plots/timeseries.png", dpi=130)
    print("wrote plots/timeseries.png")

    # ---- Figure 2: summary bars ----
    fig2, (b1, b2) = plt.subplots(1, 2, figsize=(11, 4))
    labels = list(runs.keys())
    Ps = [runs[l]["P"] for l in labels]
    Cs = [runs[l]["avgc"] for l in labels]
    b1.bar(labels, Ps, color="#3b6ea5")
    b1.set_title("Resilience score P"); b1.set_ylim(0, 1)
    for i, v in enumerate(Ps):
        b1.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    b2.bar(labels, Cs, color="#a5573b")
    b2.set_title("Average servers used (efficiency)")
    for i, v in enumerate(Cs):
        b2.text(i, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    for b in (b1, b2):
        b.tick_params(axis="x", rotation=20)
        b.grid(True, axis="y", alpha=0.3)
    fig2.tight_layout()
    fig2.savefig("plots/summary.png", dpi=130)
    print("wrote plots/summary.png")


if __name__ == "__main__":
    main()
