# `config.py` — Parameters and Scenarios

This module holds **everything tunable** in the simulator. It contains no
logic that *runs* a simulation — only the data structures that describe one.
Everything else in the package consumes a `SimConfig` produced here.

The delay/overhead numbers are reverse-engineered from the prior paper
(*"Surviving the Storm"*, arXiv:2505.00605) so the rebuilt discrete-event
simulator reproduces its published service rates exactly.

---

## What's inside

### `ArchConfig` — the control-plane delay model
Describes how long one UE attach attempt takes, broken into the components
the prior paper modelled.

| Field | Meaning | Default |
|---|---|---|
| `n_ctrl_messages` (M) | CU-handled RRC messages in the setup exchange (Setup Request, Setup, Setup Complete) | 3 |
| `proc_total_ms` (Σ tₚ,ᵢ) | total internal processing time across those messages | 30 ms |
| `oneway_delay_ms` | one-way delay from RU to the processing unit | 1.75 ms (Open RAN) |

Two methods:
- `service_time_ms()` = `proc_total_ms + n_ctrl_messages × oneway_delay_ms`
- `service_rate()` = `1000 / service_time_ms()` → the per-server **µ** in UEs/s

**How the numbers were recovered.** The paper reports µ = 32.52 (monolithic)
and 28.37 (Open RAN). Working backwards:
- difference in service time = `1/28.37 − 1/32.52 = 4.5 ms`
- Open RAN adds `(1.75 − 0.25) = 1.5 ms` per message, so `4.5 / 1.5 = 3` messages → **M = 3**
- monolithic service time `1/32.52 = 30.75 ms = 30 + 3×0.25` → **Σ tₚ,ᵢ = 30 ms**

This is why `service_rate()` returns 32.52 / 28.37 and matches Table VII at
other processing times (10 ms → 93.02/65.57, 50 ms → 19.70/18.10).

Helpers: `open_ran_arch(**kw)` (one-way 1.75 ms) and `monolithic_arch(**kw)`
(one-way 0.25 ms, Split Option 7).

### `RRCConfig` — the timer/retry behaviour (the realism upgrade)
This is what makes a storm *amplify* rather than merely congest. The
analytical M/M/c model in the prior paper cannot capture it.

| Field | Meaning | Default |
|---|---|---|
| `t300_ms` | RRC setup guard timer (T300). If an attach doesn't complete in this time, the UE abandons and retries. 3GPP allows 100–2000 ms. | 1000 ms |
| `max_attempts` | attempts before the UE gives up (counts as a failure) | 5 |
| `backoff_ms` | extra wait before a retry | 0 |

Under overload, attaches time out → retry → add load → cause more timeouts.
That feedback loop is the signaling storm.

### `StormPhase` and `TrafficConfig` — the arrival timeline
`StormPhase` is one piecewise-constant segment `[t_start, t_end)` with a
`benign_rate` and a `botnet_rate` (malicious UEs that re-attach aggressively).
`TrafficConfig` is an ordered list of phases with two helpers:
- `horizon()` — total simulated duration
- `rates_at(t)` — `(benign, botnet)` rates active at time `t`

### `SimConfig` — the top-level object
Bundles `arch`, `rrc`, `traffic`, plus runtime knobs:
- `c0` / `c_max` — initial and maximum server count (the actuator's range)
- `lq_max` — queue length at which the utility's congestion term → 0
- `sample_dt_s` — telemetry/control sampling interval (default 0.5 s)
- `seed` — RNG seed (set per run for reproducibility / averaging)
- `botnet_attach_period_ms` — how aggressively malicious UEs re-attach
- `realtime` / `rt_factor` — pace the sim clock to wall-clock time.
  `realtime=False` (default) runs as fast as possible (virtual time, for
  experiments). `realtime=True` paces it, with `rt_factor` = wall-clock seconds
  per simulated second (`1.0` = true real time, `0.1` = 10x faster, `2.0` = 2x
  slower). Used by the GUI and, later, by the agentic layer so agent decision
  latency is incurred against the live clock.
- `compute_kappa` / `compute_rho_cap` — **shared-compute contention** (optional,
  default off). Models the vCU/vDU running on a finite shared compute pool. When
  `compute_kappa = K` is set, each attach's *processing* time inflates by the
  processor-sharing factor `1/(1 - rho_c)`, where `rho_c = busy/K` and `busy` is
  the number of attaches in service. `compute_rho_cap` (default 0.98) clamps
  `rho_c` below 1 to avoid the infinite pole. `compute_kappa = None` (default)
  disables contention and recovers the prior paper's constant-rate numbers. See
  README_simulator.md for the model and its justification.
- `server_provision_delay_s` — **server provisioning delay** (default `0.0` =
  instant). Time to bring a newly commanded vDU/vCU server online (image pull /
  boot / pool attach). New servers come up one at a time, this many seconds
  apart, so actual capacity lags the controller's command during a scale-up.
  Scaling down is always immediate (in-flight attaches are never preempted).

### Scenario builders
- `single_storm_traffic(...)` — the prior paper's scenario: 20 → 200 → 20 UEs/s.
- `multi_storm_traffic()` — three storms of growing intensity **with a botnet
  component**, used to demonstrate the **evolution** stage (storm 3 should be
  handled better than storm 1 after the strategic agent learns between storms).

---

## Typical use
```python
from sim.config import SimConfig, open_ran_arch, RRCConfig, single_storm_traffic

cfg = SimConfig(
    arch=open_ran_arch(),
    rrc=RRCConfig(t300_ms=1000, max_attempts=5),
    traffic=single_storm_traffic(),
    c0=1, c_max=10, lq_max=1500, seed=3,
)
```

## Knobs that affect headline numbers
- `c_max` — tight values force the agent to *rate-limit* instead of brute-force
  scaling (the intended way to make the agentic advantage real).
- `compute_kappa` — when set, gives a *principled* version of the same idea:
  shared-compute contention makes server scaling hit diminishing then retrograde
  returns (peak throughput near `c = kappa/2`), so reducing load (rate-limiting)
  beats adding servers. Leave `None` to reproduce the paper.
- `lq_max` — calibrates the congestion sensitivity of the utility function.
- `t300_ms` / `max_attempts` — control how violently the storm amplifies.
- `server_provision_delay_s` — how slowly new capacity arrives after a scale-up
  command; larger values reward anticipation (pre-provisioning before a storm).
