# `controllers.py` — Baseline Controllers

These are the **non-agentic baselines** the agentic system will be compared
against. Each one is a closed-loop controller that acts on the running
simulator's server count `c(t)`. They form the lower rungs of the ablation
ladder; the agentic orchestrator is the top rung.

---

## The controller interface (contract)

Every controller implements one method:

```python
def step(self, sim: StormSim, s: TelemetrySample) -> None: ...
```

The simulator's control loop calls `step` once every `sample_dt_s`, passing the
live simulator (so the controller can call `sim.set_servers(...)`,
`sim.set_malicious_drop_prob(...)`) and the latest telemetry sample. This is the
same hook the agentic layer will use — the agents are just much more elaborate
`step` implementations that happen to run as parallel services and reason with
an LLM.

---

## `FixedController(c)`
Holds a constant server count. This is the "no adaptation" reference and
reproduces the prior paper's fixed-`c` experiments (`c ∈ {1, 2, 4, 6, ...}`).
- Small `c` → overwhelmed by the storm (timeouts, retries, failures).
- Large `c` → handles the storm but wastes capacity by running flat-out always.

## `LyapunovController(V, W, util_p)`
The prior paper's **drift-plus-penalty** adaptive mechanism (eqs. 10–14), and
the strongest non-agentic baseline. Each step it picks the integer `c` that
minimizes:

```
min_c   Lq·(λ − c·µ) + ½·(λ − c·µ)²  −  V·u  +  W·c
s.t.    1 ≤ c ≤ c_max
```

- the first two terms are the **Lyapunov drift** (queue stability),
- `−V·u` rewards higher utility (`V` prioritizes performance),
- `+W·c` penalizes using more servers (`W` prioritizes resource thrift).

Solved by a simple integer search over `[1, c_max]` every step. `_objective`
evaluates the expression for a candidate `c` (probing utility as if that `c`
were applied); `_lambda_estimate` returns the arrival rate it optimizes against
(here, the current measured rate — purely **reactive**).

Tuning `V` and `W` trades performance against efficiency, reproducing the
paper's `(V, W)` sweep.

## `ForecastLyapunov(V, W, horizon_s)`
Identical to `LyapunovController` except `_lambda_estimate` looks
`horizon_s` seconds **into the future** and optimizes against the larger of the
current and forecast rate — i.e. it **pre-warms** servers before a surge. This
is the **anticipation** ablation rung: same single lever, but proactive instead
of reactive.

> The forecast here reads the traffic schedule directly (an idealized oracle).
> In the agentic system this is replaced by a real predictor exposed as an MCP
> tool, so the anticipation quality — and its error — become part of the study.

---

## The ablation ladder these define

| Rung | Controller | Levers | Behaviour |
|---|---|---|---|
| 1 | `FixedController` | none | static provisioning |
| 2 | `LyapunovController` | servers | reactive adaptation |
| 3 | `ForecastLyapunov` | servers | reactive + anticipation |
| 4 | *agentic orchestrator* | servers **+ rate-limiting** + memory | coordinated, multi-stage, learns across storms |

The 3 → 4 jump is the key comparison: it isolates the contribution of
**multi-lever coordination and evolution** from mere information (forecast) or
compute (more servers), which is what the survey's thesis needs to demonstrate.

---

## Typical use
```python
from sim.controllers import LyapunovController
from sim.metrics import UtilityParams, resilience_score

sim.run(controller=LyapunovController(V=1000, W=1))
r = resilience_score(sim.telemetry, sim.mu_single, UtilityParams(lq_max=1500, kB=0.004),
                     t0=50, td=110)
```

## Note on fairness
For an honest "agentic beats Lyapunov" claim, give the forecast to **both**
sides (rung 3 vs. rung 4). Then any remaining advantage is attributable to
coordination and learning, not to the agent simply having more information.
