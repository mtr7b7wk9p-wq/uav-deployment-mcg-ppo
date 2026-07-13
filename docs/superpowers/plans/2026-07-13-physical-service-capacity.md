# Physical Service Capacity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the normalized distance service proxy with path-loss, SINR, interference, and Shannon-rate driven service capacity for resource cognition.

**Architecture:** Reuse the existing `envs/channel.py` air-to-ground path-loss implementation. Add a small physical-capacity calculation in `ResourceCognitionEnv` that evaluates every scheduled UAV-task link, computes same-band interference from other scheduled links, and returns capacity/SINR/path-loss/outage arrays. Keep the task-level queue and discrete task-selection action space unchanged.

**Tech Stack:** Python 3, NumPy, existing air-to-ground channel utilities, standard-library `unittest`, current PPO runner.

## Global Constraints

- Queue, arrivals, and served data use Mbit; rate uses Mbit/s; service duration uses seconds.
- Fixed bandwidth and transmit power are configuration parameters, not new policy actions.
- Hidden path loss, SINR, and interference must not enter local observations in this change.
- Preserve legacy `mcg_ppo` behavior and all existing resource cognition action indices.
- Keep `remaining_time` and `cognition_service_energy_cost` as the current energy proxy.
- Use standard-library `unittest` because `pytest` is not installed in the current environment.
- Every behavior change gets a failing contract test before implementation code.

## File Map

- Modify: `tests/test_dynamic_business_queue.py` - physical capacity and environment contracts.
- Modify: `configs/scenario_config.py` - physical channel and service parameters.
- Modify: `envs/channel.py` - reusable link-budget and Shannon-rate helper.
- Modify: `envs/resource_cognition_env.py` - physical capacity, interference, outage, and service integration.
- Modify: `envs/metrics.py` - physical service episode and aggregate metrics.
- Modify: `README.md` and `train.md` - physical service semantics and corrected observation dimension.

### Task 1: Add physical link-budget configuration and helper

**Files:**
- Modify: `tests/test_dynamic_business_queue.py`
- Modify: `configs/scenario_config.py`
- Modify: `envs/channel.py`

**Interfaces:**
- `ScenarioConfig` exposes validated fields `cognition_bandwidth_mhz`, `cognition_tx_power_w`, `cognition_noise_power_w`, `cognition_service_duration_s`, `cognition_channel_carrier_freq_ghz`, `cognition_channel_los_a`, `cognition_channel_los_b`, `cognition_channel_eta_los_db`, `cognition_channel_eta_nlos_db`, and `cognition_outage_sinr_threshold`.
- `channel_gain_from_path_loss_db(path_loss_db: np.ndarray) -> np.ndarray` converts dB path loss to linear gain.
- `shannon_rate_mbps(bandwidth_mhz: float, sinr: np.ndarray) -> np.ndarray` returns non-negative Mbit/s rates.

- [ ] **Step 1: Write failing helper tests**

```python
def test_path_loss_gain_and_shannon_rate_are_monotonic(self):
    gain = channel_gain_from_path_loss_db(np.array([80.0, 100.0], dtype=np.float32))
    rate = shannon_rate_mbps(1.0, np.array([1.0, 3.0], dtype=np.float32))
    self.assertGreater(gain[0], gain[1])
    self.assertGreater(rate[1], rate[0])


def test_physical_config_rejects_non_positive_parameters(self):
    with self.assertRaises(ValueError):
        ScenarioConfig(use_resource_cognition=True, cognition_bandwidth_mhz=0.0).validate()
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m unittest tests.test_dynamic_business_queue.DynamicBusinessQueueTests.test_path_loss_gain_and_shannon_rate_are_monotonic tests.test_dynamic_business_queue.DynamicBusinessQueueTests.test_physical_config_rejects_non_positive_parameters -v`

Expected: FAIL because the helper functions and new configuration validation do not exist.

- [ ] **Step 3: Implement the link-budget primitives and config validation**

Add the configuration fields with defaults from the design document. Clamp values in `__post_init__` and reject non-positive bandwidth, transmit power, noise power, service duration, and carrier frequency in `validate`.

Add these ASCII-safe helpers to `envs/channel.py`:

```python
def channel_gain_from_path_loss_db(path_loss_db: np.ndarray) -> np.ndarray:
    path_loss = np.asarray(path_loss_db, dtype=np.float32)
    return np.power(10.0, -path_loss / 10.0).astype(np.float32)


def shannon_rate_mbps(bandwidth_mhz: float, sinr: np.ndarray) -> np.ndarray:
    if bandwidth_mhz <= 0.0:
        raise ValueError("bandwidth_mhz must be positive.")
    values = np.maximum(np.asarray(sinr, dtype=np.float32), 0.0)
    return (float(bandwidth_mhz) * np.log2(1.0 + values)).astype(np.float32)
```

- [ ] **Step 4: Run the helper tests and verify they pass**

Run: `python -m unittest tests.test_dynamic_business_queue.DynamicBusinessQueueTests.test_path_loss_gain_and_shannon_rate_are_monotonic tests.test_dynamic_business_queue.DynamicBusinessQueueTests.test_physical_config_rejects_non_positive_parameters -v`

Expected: PASS.

- [ ] **Step 5: Commit the helper layer**

```bash
git add tests/test_dynamic_business_queue.py configs/scenario_config.py envs/channel.py
git commit -m "feat: add physical channel service primitives"
```

### Task 2: Replace normalized scheduling capacity with physical SINR capacity

**Files:**
- Modify: `tests/test_dynamic_business_queue.py`
- Modify: `envs/resource_cognition_env.py`

**Interfaces:**
- `ResourceCognitionEnv._physical_service_capacity(assignments, conflict_counts, use_truth)` returns `(capacity_by_agent, sinr_by_agent, path_loss_by_agent, outage_by_agent)`.
- `_evaluate_schedule` continues returning `(utility, service_by_agent, conflict_counts, capacity_by_agent)`, but its capacity now comes from the physical helper.
- `_execute_scheduling` returns physical arrays and outage/interference statistics in its result dictionary.

- [ ] **Step 1: Write failing environment tests**

```python
def test_physical_service_capacity_decreases_with_noise(self):
    env = make_env()
    env.reset(seed=11)
    env.task_truth.true_states[:] = 0.0
    assignments = np.array([0, -1], dtype=np.int64)
    conflicts = np.zeros(env.num_agents, dtype=np.float32)
    first = env._physical_service_capacity(assignments, conflicts, use_truth=True)
    env.cfg.cognition_noise_power_w *= 100.0
    second = env._physical_service_capacity(assignments, conflicts, use_truth=True)
    self.assertLessEqual(second[1][0], first[1][0])
    self.assertLessEqual(second[0][0], first[0][0])


def test_same_band_interference_reduces_physical_capacity(self):
    env = make_env()
    env.reset(seed=12)
    env.task_truth.true_states[:] = 0.0
    env.task_truth.band_ids[:] = 0
    env.task_truth.positions_xy[1] = env.task_truth.positions_xy[0]
    assignments = np.array([0, 1], dtype=np.int64)
    conflicts = np.zeros(env.num_agents, dtype=np.float32)
    with_interference = env._physical_service_capacity(assignments, conflicts, use_truth=True)
    without_interference = env._physical_service_capacity(
        np.array([0, -1], dtype=np.int64), conflicts, use_truth=True
    )
    self.assertLess(with_interference[1][0], without_interference[1][0])
    self.assertLess(with_interference[0][0], without_interference[0][0])
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m unittest tests.test_dynamic_business_queue.DynamicBusinessQueueTests.test_physical_service_capacity_decreases_with_noise tests.test_dynamic_business_queue.DynamicBusinessQueueTests.test_same_band_interference_reduces_physical_capacity -v`

Expected: FAIL because `_physical_service_capacity` does not exist.

- [ ] **Step 3: Implement the physical capacity helper**

Build an `AirToGroundChannelConfig` from scenario fields and call `average_atg_path_loss_db` with task positions and UAV positions. For each assigned agent, compute desired received power. For each other assigned agent with the same task band, compute the other transmitter's received power at the current task location and add it to interference. Use:

```python
sinr = desired_power / (noise_power + interference_power)
rate = shannon_rate_mbps(cfg.cognition_bandwidth_mhz, sinr)
capacity = rate * cfg.cognition_service_duration_s
capacity = np.minimum(capacity, cfg.cognition_max_service_per_step)
outage = sinr < cfg.cognition_outage_sinr_threshold
capacity[outage] = 0.0
```

For `use_truth=False`, use the local spectrum estimate only to decide spectrum availability while keeping geometry and channel truth available to the environment-level estimated utility calculation. Do not put path loss or SINR into `get_local_obs`.

- [ ] **Step 4: Connect physical capacity to queue service**

Replace the current exponential-distance `raw_capacity` calculation in `_evaluate_schedule` with `_physical_service_capacity`. Preserve queue capping, conflict counts, difference rewards, service energy deductions, and action masks. Add `mean_path_loss_db`, `mean_sinr`, `mean_service_capacity`, `service_outage_count`, `service_outage_rate`, and `total_interference_power_w` to scheduling stats and step `info`.

- [ ] **Step 5: Run service and physical tests**

Run: `python -m unittest tests.test_dynamic_business_queue -v`

Expected: all existing queue/message/service contracts and the new physical capacity contracts pass.

- [ ] **Step 6: Commit physical scheduling**

```bash
git add tests/test_dynamic_business_queue.py envs/resource_cognition_env.py
git commit -m "feat: use physical SINR for resource service"
```

### Task 3: Record physical service metrics and update documentation

**Files:**
- Modify: `tests/test_dynamic_business_queue.py`
- Modify: `envs/metrics.py`
- Modify: `README.md`
- Modify: `train.md`

**Interfaces:**
- `EpisodeMetrics` records path loss, SINR, physical capacity, outages, and interference.
- `MetricTracker.aggregate()` returns mean physical service metrics.

- [ ] **Step 1: Write a failing physical metrics test**

```python
def test_metrics_collect_physical_service_fields(self):
    metrics = EpisodeMetrics()
    metrics.update(1.0, {
        "mean_path_loss_db": 95.0,
        "mean_sinr": 4.0,
        "mean_service_capacity": 1.5,
        "service_outage_count": 1,
        "service_outage_rate": 0.5,
        "total_interference_power_w": 2e-13,
    })
    summary = metrics.summary()
    self.assertEqual(summary["mean_path_loss_db"], 95.0)
    self.assertEqual(summary["mean_sinr"], 4.0)
    self.assertEqual(summary["total_service_outages"], 1)
```

- [ ] **Step 2: Run the test and verify the expected failure**

Run: `python -m unittest tests.test_dynamic_business_queue.DynamicBusinessQueueTests.test_metrics_collect_physical_service_fields -v`

Expected: FAIL because the new physical fields are not collected.

- [ ] **Step 3: Implement metric collection and documentation**

Add per-step lists, final summary fields, and aggregate means for path loss, SINR, capacity, outage count/rate, and interference. Update README and `train.md` to state that queue units are Mbit, service capacity is computed from the existing air-to-ground path-loss model and Shannon rate, and the default resource observation dimension is `176` rather than `210`.

- [ ] **Step 4: Run all tests and static checks**

Run:

```bash
python -m unittest tests.test_dynamic_business_queue -v
python -m py_compile configs/scenario_config.py envs/channel.py envs/resource_cognition_env.py envs/metrics.py agents/ppo/models.py
git diff --check
```

Expected: all tests pass, compilation succeeds, and no whitespace errors are reported.

- [ ] **Step 5: Run both one-update smoke tests**

Run:

```bash
python runners/train_ppo_deployment.py --method-name mcg_ppo_resource_cognition --total-updates 1 --rollout-episodes-per-update 1 --eval-interval 1 --eval-episodes 1 --save-interval 9999 --device cpu --output-root results/smoke_physical_capacity
python runners/train_ppo_deployment.py --method-name mcg_ppo --total-updates 1 --rollout-episodes-per-update 1 --eval-interval 1 --eval-episodes 1 --save-interval 9999 --device cpu --output-root results/smoke_physical_capacity
```

Expected: both methods finish one update; the resource run prints service metrics and legacy coverage behavior remains unchanged.

- [ ] **Step 6: Commit metrics and docs**

```bash
git add tests/test_dynamic_business_queue.py envs/metrics.py README.md train.md
git commit -m "feat: report physical service assurance metrics"
```

## Self-Review Checklist

- Spec coverage: units, path loss, received power, interference, SINR, Shannon rate, outage, queue capping, metrics, compatibility, and smoke tests are covered by Tasks 1-3.
- Type consistency: channel helpers use NumPy arrays; environment helper returns four per-agent arrays; scheduling stats and info use scalar summaries plus per-agent arrays.
- Scope: no user-level queue, continuous action, or physical propulsion-energy redesign is included.
- Documentation: default resource observation dimension is explicitly corrected to `176`.
