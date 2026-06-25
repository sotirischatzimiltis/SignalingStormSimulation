# `metrics.py` — Utility and Resilience Scoring

This module turns a stream of `TelemetrySample`s into numbers: an instantaneous
**utility** `u(t)` and an episode-level **resilience score** `P`. Both are
reconstructed from the prior paper (*"Surviving the Storm"*, arXiv:2505.00605),
equations (9) and (8) respectively.

It is pure post-processing — it reads telemetry, never touches the simulator.

---

## Utility function `u(t)` — eq. (9)

A value in `[0, 1]`; higher means a more stable / healthier system. It blends
two sigmoid terms:

```
u(t) = wA·uA(t) + wB·uB(t)
uA(t) = 1 / (1 + exp(kA·(λ(t) − mA)))      # arrival-rate health
uB(t) = 1 / (1 + exp(kB·(Lq(t) − mB)))      # queue-length health
```

### `UtilityParams`
| Field | Meaning | Default |
|---|---|---|
| `wA`, `wB` | weights on the two terms (sum to 1) | 0.5 / 0.5 |
| `kA`, `kB` | steepness of each sigmoid transition | 0.5 / 0.01 |
| `mfracA` | midpoint of `uA` as a fraction of `c·µ` (so `mA = c·µ·mfracA`) | 0.75 |
| `lq_max` | queue length at which `uB → 0` | 7000 |
| `mfracB` | midpoint of `uB` as a fraction of `lq_max` (`mB = lq_max·mfracB`) | 0.5 |

Note `mA` depends on the **current server count `c`**, so adding servers
*raises* the arrival-rate the system considers "healthy" — adaptation directly
improves utility.

### Functions
- `utility(sample, mu_single, params)` → `u(t)` for one sample.
- `utility_series(telemetry, mu_single, params)` → list of `u(t)`.

---

## Resilience score `P` — eq. (8) (the A3RT metric)

`P` summarizes how well the system weathered one storm episode, combining three
components over the disruption window:

```
P = w1·(absorption) + w2·(adaptation) + w3·(trec)
```

- **absorption** — area under `u(t)` during the storm `[t0, td]`, normalized by
  the desired utility. How well service was held *during* the hit.
- **adaptation** — same ratio over the recovery window `[td, tr]`. How well
  partial function was restored *after* the hit.
- **trec** — a time-to-recovery term: `1` if recovery happened within the
  desired threshold `dt_des`, otherwise scaled down as `dt_des / (tr − t0)`.

### `resilience_score(...)` arguments
| Argument | Meaning |
|---|---|
| `t0`, `td` | storm start and end (known from the traffic schedule) |
| `u_des` | desired/ideal utility. **If `None` (recommended), auto-calibrated to the mean pre-storm baseline utility over `[0, t0]`.** |
| `dt_des` | desired recovery-time threshold (default 60 s) |
| `recovery_frac` | fraction of `u_des` that counts as "recovered" (default 0.95) |
| `weights` | `ResilienceWeights(w1, w2, w3)` — defaults 0.4 / 0.4 / 0.2 |

Returns a dict: `P`, `absorption`, `adaptation`, `trec`, `tr`,
`recovery_time`.

### The `u_des` calibration (important modelling choice)
The paper defines `u_des` as the "desired utility under ideal conditions." A
flat `u_des = 1.0` is wrong in practice: a healthy steady state with one server
serving 20 UEs/s already sits at `u ≈ 0.83` (because 20 is close to the `uA`
midpoint `0.75·µ`). With a flat target of 1.0, nothing ever registers as
recovered. Setting `u_des` to the **measured pre-storm baseline** makes the
metric self-calibrating and recovery detection meaningful. This is the default.

### Recovery detection (`tr`)
`tr` is the first time after `td` at which `u(t)` rises above
`recovery_frac · u_des` **and stays above it for a sustained 30 s window** — a
robust check that ignores brief late dips.

### `_trapz`
Internal trapezoidal-integration helper for the absorption/adaptation areas.

---

## Typical use
```python
from sim.metrics import UtilityParams, resilience_score

up = UtilityParams(lq_max=1500, kB=0.004)
r = resilience_score(sim.telemetry, sim.mu_single, up, t0=50, td=110)
print(r["P"], r["recovery_time"])
```

## Calibration notes
- `lq_max` and `kB` should match the queue depths your scenario actually
  reaches (≈1000–1500 in the current single-storm setup).
- Leave `u_des=None` unless you have a specific external "ideal" to compare
  against.
- `weights` let you emphasise absorption vs. recovery per the paper's guidance
  (e.g. higher `w1`/`w2` for high-risk services).
