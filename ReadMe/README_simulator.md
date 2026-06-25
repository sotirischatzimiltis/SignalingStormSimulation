# `simulator.py` — The Discrete-Event Engine

This is the heart of the system: a [SimPy](https://simpy.readthedocs.io)
discrete-event simulation of Open RAN UE initial attachment under load. It
models **individual UEs** performing the control-plane attach procedure against
a pool of CU-processing servers, with **explicit RRC timers and retries** so
that signaling storms amplify realistically.

It replaces the analytical M/M/1 / M/M/c treatment in the prior paper with an
event-driven model that (a) captures retry-driven amplification and (b) exposes
runtime **actuators** so external controllers and agents can intervene.

---

## Data structures

### `TelemetrySample`
One snapshot of system state, recorded every `sample_dt_s`. Fields:
`t`, `lam_target` (instantaneous arrival rate), `queue_len`, `busy`,
`in_system`, `c` (servers **online and serving**), `c_target` (servers
**commanded** by the controller; exceeds `c` during a scale-up warm-up), and
cumulative counters `completed`, `failed`, `retries`, `arrivals`. This is the
observable stream a controller or agent reads.

### `Stats`
Cumulative run totals plus `completion_delays` (per-UE attach latency in ms),
used for validation against M/M/1 theory.

### `_Attempt`
One attach attempt. A single UE may produce several `_Attempt`s across retries.
Carries a `served_evt` (fired when service completes), an `abandoned` flag (set
on timeout), and an `in_service` flag.

---

## `StormSim` — the simulator

Constructed from a `SimConfig`. On init it starts three SimPy processes:
arrivals, the dispatcher, and telemetry sampling.

### Runtime actuators (what controllers/agents call)
- `set_servers(c)` — set the **commanded** server count, clamped to `[1, c_max]`.
  This is the **adaptation** lever (the only one Lyapunov has). The command takes
  effect on `c` immediately, but the number actually serving (`c_online`) only
  catches up after the provisioning delay (see provisioning manager below).
- `set_malicious_drop_prob(p)` — drop a fraction `p` of *malicious* attach
  attempts at admission. This is the **absorption / rate-limiting** lever
  (the extra lever the agentic system has and Lyapunov does not).
- `live_rate_override` — set to `(benign_rate, botnet_rate)` to override the
  traffic schedule on the fly (used by the GUI sliders); set back to `None` to
  resume the scheduled traffic.
- `mu_single` (property) — the per-server service rate, for controllers.

### Real-time vs. virtual time
On construction the simulator picks its SimPy environment from the config: a
plain `simpy.Environment` (virtual time, runs as fast as possible) when
`cfg.realtime` is `False`, or a `simpy.rt.RealtimeEnvironment(factor=rt_factor,
strict=False)` when `True`. `strict=False` is deliberate: if a control step
(e.g. an agent's LLM call) overruns its real-time slot, the sim does not crash —
the action simply lands late, and that lateness is exactly the agentic overhead
acting on the queue.

### How it works internally
- **`_arrival_process`** — generates Poisson arrivals at the time-varying total
  rate `benign + botnet` from the traffic schedule; each arrival is tagged
  benign or malicious in proportion to those rates.
- **`_ue_attach`** — a single UE's lifecycle, and the source of amplification:
  1. (optional) admission control drops the attempt if malicious and unlucky;
  2. the attempt joins the waiting list and starts a **T300 timer**;
  3. it waits for *either* service completion *or* timer expiry;
  4. on completion → success; on timeout → abandon, count a retry, and loop;
  5. after `max_attempts` exhausted → failure.
  Malicious UEs additionally re-attach every `botnet_attach_period_ms`.
- **`_dispatcher`** — assigns waiting attempts to free servers whenever
  `busy < c_online`. Because `c_online` is mutable, this is a hand-rolled
  **dynamic-capacity** queue (SimPy's built-in `Resource` has a fixed capacity,
  which would not let an actuator resize the pool mid-run). A wake-event
  (`_signal` / `_wake`) re-checks the queue whenever an arrival, completion, or
  newly-online server occurs.
- **`_provisioning_manager`** — reconciles the online count `c_online` toward
  the commanded target `c`. Scaling **up** is gradual: each new server takes
  `server_provision_delay_s` to come online (image pull / boot / pool attach of
  a vDU/vCU instance), brought up one at a time. Scaling **down** is immediate —
  no preemption: any busy server finishes its current attach, the dispatcher
  just stops starting new work once `busy >= c_online`. With the default delay
  of 0 this is instant and `c_online == c` always.
- **`_serve`** — holds a server for an exponential service time (see
  `_service_time` below), then frees it and fires the attempt's `served_evt`
  (unless the UE already abandoned).
- **`_service_time`** — the mean service time of one attach. With contention
  off (`compute_kappa is None`) it is the constant value from `ArchConfig`
  (`proc_total + M·oneway`), reproducing the prior paper. With a compute budget
  `kappa` set, only the **processing** component inflates:

  ```
  service_time = proc_total / (1 - rho_c)  +  M · oneway
  rho_c = min(busy / kappa, rho_cap)
  ```

  This is the processor-sharing (PS) model: `proc_total / (1 - rho_c)` is the
  exact mean sojourn time of an M/M/1-PS server processing a job of demand
  `proc_total` at utilization `rho_c` (equivalently the heavy-traffic / Kingman
  `1/(1-rho)` law). The F1/O-FH propagation term `M·oneway` is fixed link
  physics and is *not* inflated. The slowdown is keyed on `busy` (concurrent
  compute demand), not on queue length, so it does not double-count the queuing
  delay that the waiting line already produces. Consequence: server scaling
  shows diminishing then **retrograde** returns (peak throughput near
  `c = kappa/2`), which is why reducing load (rate-limiting) beats adding
  servers under shared compute.
- **`_telemetry_process`** — appends a `TelemetrySample` every `sample_dt_s`.

### Running
```python
sim = StormSim(cfg)
sim.run(controller=my_controller)   # controller may be None for open-loop
telemetry = sim.telemetry
```
`run(until, controller)` advances the clock to `until` (defaults to the traffic
horizon). If a `controller` is given, `_control_loop` calls
`controller.step(sim, latest_sample)` every `sample_dt_s` — this is the hook the
baseline controllers and (later) the agentic layer plug into.

---

## Validation status
- Steady state matches **M/M/1 theory** and the prior paper's Table VI:
  at ρ = 0.5 (Open RAN) measured W ≈ 70.7 ms vs. theoretical/published 70.52 ms.
- Throughput tracks offered load when stable.
- Retry amplification behaves correctly: with too few servers, attaches time
  out, retry, and fail en masse; with enough servers the storm is absorbed
  with zero failures.

## Where this is heading
The actuators (`set_servers`, `set_malicious_drop_prob`) and the telemetry
stream will be exposed as **MCP tools/resources**, so the parallel A2A agents
act on the simulator exactly the way the baseline controllers do today.
