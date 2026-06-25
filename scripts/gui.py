"""
Real-time interactive dashboard for the signaling-storm simulator.

Run on your own machine (needs a display / interactive matplotlib backend):

    python -m scripts.gui

What you get:
  - The simulator runs in REAL TIME on a background thread.
  - Live-updating plots: queue length, utility u(t), active servers, arrival rate.
  - Sliders you can drag WHILE IT RUNS:
      * Benign rate   - benign UE arrivals/s  (push it past c*mu to start a storm)
      * Botnet rate   - malicious UE arrivals/s
      * Servers (c)   - manual server allocation (the adaptation lever)
      * Drop malicious- fraction of malicious attempts rate-limited (absorption lever)
      * T300 (ms)     - RRC setup timer (lower -> faster, more aggressive retries)
  - Buttons: Pause / Resume, Reset.

This is a teaching / exploration tool and the visual front-end the agentic
system will later stream into. It drives the real simulator (single source of
truth) via the same actuators the controllers use.

Note: matplotlib sliders are fine for exploration. For a polished web dashboard
(nicer styling, remote access, agent panels), a Streamlit or FastAPI+websocket
version is a natural next step - see README roadmap.
"""

import threading
import time
from collections import deque

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from sim.config import SimConfig, open_ran_arch, RRCConfig, TrafficConfig, StormPhase
from sim.simulator import StormSim
from sim.metrics import UtilityParams, utility

WINDOW_S = 120.0        # seconds of history shown
LQMAX = 1500.0
UP = UtilityParams(lq_max=LQMAX, kB=0.004)


class SimRunner:
    """Runs a StormSim in real time on a background thread, restartable."""

    def __init__(self):
        self.lock = threading.Lock()
        self.thread = None
        self.stop_flag = False
        self.paused = False
        # shared, slider-controlled parameters
        self.benign = 20.0
        self.botnet = 0.0
        self.target_c = 1
        self.drop = 0.0
        self.t300 = 1000.0
        self.sim = None

    def _make_sim(self):
        cfg = SimConfig(
            arch=open_ran_arch(),
            rrc=RRCConfig(t300_ms=self.t300, max_attempts=5),
            c0=self.target_c, c_max=16, lq_max=LQMAX,
            traffic=TrafficConfig([StormPhase(0, 10_000, self.benign, self.botnet)]),
            realtime=True, rt_factor=1.0, seed=int(time.time()) % 10000,
        )
        sim = StormSim(cfg)
        sim.live_rate_override = (self.benign, self.botnet)
        return sim

    def _gui_controller(self):
        runner = self

        class _Ctrl:
            def step(self, sim, s):
                # push slider values into the live sim every control step
                sim.live_rate_override = (runner.benign, runner.botnet)
                sim.cfg.rrc.t300_ms = runner.t300
                sim.set_malicious_drop_prob(runner.drop)
                sim.set_servers(runner.target_c)
                while runner.paused and not runner.stop_flag:
                    time.sleep(0.05)
        return _Ctrl()

    def _run(self):
        self.sim = self._make_sim()
        # run in chunks so stop_flag is checked; RealtimeEnvironment paces it
        ctrl = self._gui_controller()
        self.sim.env.process(self.sim._control_loop(ctrl))
        try:
            # step the env in small real-time slices
            while not self.stop_flag and self.sim.env.peek() < 10_000:
                self.sim.env.step()
        except Exception:
            pass

    def start(self):
        self.stop_flag = False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def reset(self):
        self.stop_flag = True
        if self.thread:
            self.thread.join(timeout=1.0)
        self.start()


def main():
    runner = SimRunner()
    runner.start()

    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    plt.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.34, hspace=0.35)
    fig.suptitle("Open RAN signaling-storm simulator - live", fontsize=13)

    titles = ["Queue length (waiting UEs)", "Utility u(t)",
              "Active servers c(t)", "Arrival rate lambda (UEs/s)"]
    axes = [ax[0][0], ax[0][1], ax[1][0], ax[1][1]]
    lines = []
    for a, t in zip(axes, titles):
        (ln,) = a.plot([], [], linewidth=1.6)
        a.set_title(t); a.set_xlabel("time (s)"); a.grid(True, alpha=0.3)
        lines.append(ln)
    axes[1].set_ylim(0, 1.05)

    hist = {k: deque(maxlen=2000) for k in ["t", "q", "u", "c", "lam"]}

    def refresh(_frame):
        sim = runner.sim
        if sim is None or not sim.telemetry:
            return lines
        s = sim.telemetry[-1]
        hist["t"].append(s.t)
        hist["q"].append(s.queue_len)
        hist["u"].append(utility(s, sim.mu_single, UP))
        hist["c"].append(s.c)
        hist["lam"].append(s.lam_target)
        tmax = hist["t"][-1]
        tmin = max(0.0, tmax - WINDOW_S)
        xs = list(hist["t"])
        for ln, key, a in zip(lines, ["q", "u", "c", "lam"], axes):
            ln.set_data(xs, list(hist[key]))
            a.set_xlim(tmin, tmax + 1)
            if key != "u":
                ys = [y for x, y in zip(xs, hist[key]) if x >= tmin] or [0]
                a.set_ylim(0, max(ys) * 1.2 + 1)
        return lines

    # ----- sliders -----
    def add_slider(y, label, lo, hi, init, step=None):
        sax = fig.add_axes([0.10, y, 0.36, 0.03])
        return Slider(sax, label, lo, hi, valinit=init, valstep=step)

    s_benign = add_slider(0.22, "Benign rate", 0, 400, runner.benign, 1)
    s_botnet = add_slider(0.17, "Botnet rate", 0, 200, runner.botnet, 1)
    s_servers = add_slider(0.12, "Servers c", 1, 16, runner.target_c, 1)
    s_drop = add_slider(0.07, "Drop malicious", 0.0, 1.0, runner.drop, 0.05)
    s_t300 = add_slider(0.02, "T300 (ms)", 100, 2000, runner.t300, 50)

    def on_change(_):
        runner.benign = s_benign.val
        runner.botnet = s_botnet.val
        runner.target_c = int(s_servers.val)
        runner.drop = s_drop.val
        runner.t300 = s_t300.val
    for s in (s_benign, s_botnet, s_servers, s_drop, s_t300):
        s.on_changed(on_change)

    # ----- buttons -----
    bax_pause = fig.add_axes([0.62, 0.10, 0.12, 0.05])
    bax_reset = fig.add_axes([0.76, 0.10, 0.12, 0.05])
    b_pause = Button(bax_pause, "Pause/Resume")
    b_reset = Button(bax_reset, "Reset")

    def toggle_pause(_):
        runner.paused = not runner.paused
    def do_reset(_):
        for h in hist.values():
            h.clear()
        runner.reset()
    b_pause.on_clicked(toggle_pause)
    b_reset.on_clicked(do_reset)

    # animation
    from matplotlib.animation import FuncAnimation
    _anim = FuncAnimation(fig, refresh, interval=300, blit=False, cache_frame_data=False)
    fig._anim = _anim  # keep a reference alive
    plt.show()
    runner.stop_flag = True


if __name__ == "__main__":
    main()
