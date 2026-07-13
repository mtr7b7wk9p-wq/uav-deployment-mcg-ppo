# Dynamic Business Queue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add dynamic aggregate business queues to resource cognition and make scheduling actions change actual service rate, queue backlog, and communication assurance.

**Architecture:** Keep each resource task as a region-band pair. Extend `TaskTruthBatch` with hidden arrival, queue, and service state; extend `LocalBeliefBatch` with per-UAV queue and arrival beliefs; execute scheduling from hidden truth while reporting estimated utility from local beliefs. Keep the legacy coverage environment and `mcg_ppo` path unchanged.

**Tech Stack:** Python 3, NumPy, dataclasses, PyTorch PPO, pytest, existing `ScenarioConfig` and `ResourceCognitionEnv` interfaces.

## Global Constraints

- Use task-level aggregate queues; do not introduce explicit user association in this change.
- Hidden queue and arrival truth must never enter local observations directly.
- Only accepted sensing or neighbor messages may update local queue beliefs.
- Preserve the legacy coverage environment and `mcg_ppo` action/observation behavior.
- Resource cognition checkpoints are incompatible after the resource observation dimension changes.
- Every behavior change must have a failing contract test before production code is written.

## File Map

- Create: `tests/test_dynamic_business_queue.py` - focused environment and model contracts.
- Modify: `configs/scenario_config.py` - queue, arrival, service, and reward parameters plus validation.
- Modify: `envs/task_model.py` - hidden dynamic task truth and local queue beliefs.
- Modify: `envs/communication_model.py` - queue and arrival fields in cognition messages.
- Modify: `envs/resource_cognition_env.py` - arrivals, queue update, service capacity, observations, reward, and info.
- Modify: `agents/ppo/models.py` - structured resource observation slot dimensions.
- Modify: `envs/metrics.py` - service and queue episode/aggregate metrics.
- Modify: `README.md` and `train.md` - resource action/observation and metric documentation.

### Task 1: Add the dynamic truth and belief state contracts

**Files:**
- Create: `tests/test_dynamic_business_queue.py`
- Modify: `envs/task_model.py`

**Interfaces:**
- `TaskTruthBatch.advance_business(arrivals: np.ndarray) -> dict[str, float]` updates hidden queues and returns total arrivals and overflow.
- `TaskTruthBatch.apply_service(served: np.ndarray) -> float` subtracts per-task served data and returns the actual total served amount.
- `LocalBeliefBatch` exposes `queue_estimates`, `queue_uncertainties`, `queue_aoi`, `queue_confidence`, and `arrival_estimates` with shape `[num_agents, num_tasks]`.
- `LocalBeliefBatch.apply_local_sensing(...)` accepts `queue_observations` and `arrival_observations` and updates all five queue-related belief fields for selected agent-task pairs.

- [ ] **Step 1: Write failing tests for queue state transitions**

```python
def test_queue_adds_arrivals_without_service():
    truth = make_truth(queue_lengths=[1.0, 0.0], queue_capacity=10.0)
    stats = truth.advance_business(np.array([2.0, 3.0], dtype=np.float32))
    assert np.allclose(truth.queue_lengths, [3.0, 3.0])
    assert stats["total_arrivals"] == 5.0
    assert stats["queue_overflow"] == 0.0


def test_service_is_capped_by_queue_and_capacity():
    truth = make_truth(queue_lengths=[2.0, 10.0], queue_capacity=10.0)
    truth.advance_business(np.zeros(2, dtype=np.float32))
    served = truth.apply_service(np.array([5.0, 4.0], dtype=np.float32))
    assert np.allclose(truth.queue_lengths, [0.0, 6.0])
    assert served == 6.0
```

- [ ] **Step 2: Run the focused tests and verify the expected failure**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "queue_adds or service_is_capped" -q`

Expected: FAIL because `TaskTruthBatch` has no queue transition methods and no queue arrays.

- [ ] **Step 3: Implement the smallest hidden-state extension**

Add `arrival_rates`, `queue_lengths`, and `queue_capacity` to `TaskTruthBatch.__init__`, validate their shapes, and implement:

```python
def advance_business(self, arrivals: np.ndarray) -> dict[str, float]:
    values = np.asarray(arrivals, dtype=np.float32)
    if values.shape != self.queue_lengths.shape:
        raise ValueError("arrivals must match queue_lengths shape.")
    before = self.queue_lengths.copy()
    raw = before + np.maximum(values, 0.0)
    self.queue_lengths[:] = np.minimum(raw, self.queue_capacity)
    return {
        "total_arrivals": float(np.sum(self.queue_lengths - before)),
        "queue_overflow": float(np.sum(np.maximum(raw - self.queue_capacity, 0.0))),
    }

def apply_service(self, served: np.ndarray) -> float:
    values = np.asarray(served, dtype=np.float32)
    if values.shape != self.queue_lengths.shape:
        raise ValueError("served must match queue_lengths shape.")
    actual = np.minimum(np.maximum(values, 0.0), self.queue_lengths)
    self.queue_lengths[:] -= actual
    return float(np.sum(actual))
```

Initialize local queue estimates to zero, queue uncertainty to one, queue AoI to the configured initial AoI, queue confidence to zero, and arrival estimates to zero. Add queue observations to `apply_local_sensing` and update queue uncertainty/AoI/confidence with the same reduction pattern as demand beliefs.

- [ ] **Step 4: Run the focused tests and verify they pass**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "queue_adds or service_is_capped" -q`

Expected: PASS.

- [ ] **Step 5: Commit the isolated state-model change**

```bash
git add tests/test_dynamic_business_queue.py envs/task_model.py
git commit -m "feat: add dynamic task business queues"
```

### Task 2: Extend cognition messages without leaking hidden truth

**Files:**
- Modify: `tests/test_dynamic_business_queue.py`
- Modify: `envs/task_model.py`
- Modify: `envs/communication_model.py`

**Interfaces:**
- `CognitionMessage` adds `queue_estimate`, `queue_uncertainty`, `queue_confidence`, `queue_aoi`, and `arrival_estimate` with backward-compatible defaults.
- `LocalBeliefBatch.fuse_neighbor_message(...)` accepts the same queue fields and updates the receiver only if the queue message passes the existing confidence and freshness checks.

- [ ] **Step 1: Add a failing message-fusion test**

```python
def test_queue_belief_changes_only_after_accepted_message():
    beliefs = make_beliefs(num_agents=2, num_tasks=1)
    before = beliefs.queue_estimates[1, 0]
    result = beliefs.fuse_neighbor_message(
        receiver_id=1, task_id=0,
        estimate=0.0, uncertainty=0.2, confidence=0.8, message_aoi=0.0,
        queue_estimate=7.0, queue_uncertainty=0.2,
        queue_confidence=0.8, queue_aoi=0.0, arrival_estimate=2.0,
        source_update_step=1, current_step=1,
        confidence_threshold=0.05, freshness_decay=0.1,
    )
    assert result["queue_accepted"] == 1.0
    assert beliefs.queue_estimates[1, 0] != before
    assert beliefs.queue_estimates[1, 0] > 0.0
```

- [ ] **Step 2: Run the test and verify it fails for the missing queue message fields**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "message" -q`

Expected: FAIL because `fuse_neighbor_message` does not accept queue arguments and `CognitionMessage` cannot carry them.

- [ ] **Step 3: Implement queue message transport and fusion**

Add the five fields to `CognitionMessage`. In `LocalBeliefBatch.fuse_neighbor_message`, call `_fuse_dimension` for queue estimates and queue arrival estimates, return `queue_accepted`, and update `last_update_step` when any queue field is accepted. Keep the existing spectrum/demand behavior unchanged.

- [ ] **Step 4: Run the test and verify it passes**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "message" -q`

Expected: PASS.

- [ ] **Step 5: Commit the message contract**

```bash
git add tests/test_dynamic_business_queue.py envs/task_model.py envs/communication_model.py
git commit -m "feat: propagate queue beliefs through cognition messages"
```

### Task 3: Connect arrivals and real service to scheduling

**Files:**
- Modify: `tests/test_dynamic_business_queue.py`
- Modify: `configs/scenario_config.py`
- Modify: `envs/resource_cognition_env.py`

**Interfaces:**
- `ResourceCognitionEnv._sample_business_arrivals() -> np.ndarray` returns non-negative per-task arrivals.
- `ResourceCognitionEnv._evaluate_schedule(assignments, use_truth: bool) -> tuple[float, np.ndarray, np.ndarray]` returns normalized utility, service capacity by agent, and conflict counts.
- `ResourceCognitionEnv._execute_scheduling(...)` returns `served_by_task`, `service_rate`, `weighted_demand_satisfaction`, `high_priority_service_rate`, and service energy fields in addition to existing scheduling fields.

- [ ] **Step 1: Add failing service tests**

```python
def test_scheduling_reduces_queue_and_reports_service_rate():
    env = make_env(num_tasks=2, max_steps=5)
    env.reset(seed=7)
    env.task_truth.queue_lengths[:] = [8.0, 0.0]
    env.task_truth.arrival_rates[:] = 0.0
    env._slot_task_indices[0] = np.array([0], dtype=np.int64)
    env._slot_task_indices[1] = np.array([], dtype=np.int64)
    actions = np.array([env._schedule_action_start(), 0], dtype=np.int64)
    _, _, _, info = env.step(actions)
    assert info["scheduling_served_data"] > 0.0
    assert env.task_truth.queue_lengths[0] < 8.0
    assert 0.0 < info["service_rate"] <= 1.0


def test_same_band_conflict_reduces_service():
    env = make_env(num_tasks=2, max_steps=5)
    env.reset(seed=8)
    env.task_truth.band_ids[:] = 0
    env.task_truth.queue_lengths[:] = 8.0
    env.task_truth.arrival_rates[:] = 0.0
    no_conflict = env._evaluate_schedule(np.array([0, 1]), use_truth=True)[1].sum()
    env.task_truth.positions_xy[1] = env.task_truth.positions_xy[0]
    conflict = env._evaluate_schedule(np.array([0, 1]), use_truth=True)[1].sum()
    assert conflict < no_conflict
```

- [ ] **Step 2: Run the tests and verify the expected failure**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "scheduling_reduces or same_band" -q`

Expected: FAIL because scheduling currently computes static demand utility and does not mutate task queues or report service metrics.

- [ ] **Step 3: Add validated scenario parameters**

Add and clamp these `ScenarioConfig` fields:

```python
cognition_queue_capacity: float = 20.0
cognition_initial_queue_min: float = 0.0
cognition_initial_queue_max: float = 4.0
cognition_arrival_rate_min: float = 0.1
cognition_arrival_rate_max: float = 0.8
cognition_arrival_noise_std: float = 0.05
cognition_base_service_rate: float = 4.0
cognition_max_service_per_step: float = 4.0
cognition_queue_reward_weight: float = 1.0
cognition_service_reward_weight: float = 5.0
cognition_priority_service_weight: float = 2.0
```

Require non-negative bounds, `max >= min`, and positive queue capacity/service limits.

- [ ] **Step 4: Implement the environment state transition**

At reset, sample per-task arrival rates and initial queues and pass them into `TaskTruthBatch`. At the start of `step`, sample arrivals from a clipped normal distribution around each task arrival rate, call `advance_business`, and store arrival/overflow statistics.

Change `_evaluate_schedule` so `use_truth=True` reads hidden queue, spectrum, demand, and priority; `use_truth=False` reads local queue and demand beliefs. Compute a service capacity per assigned agent using:

```python
capacity = cfg.cognition_base_service_rate * link_quality
capacity *= spectrum_availability * conflict_factor * energy_factor
capacity = min(capacity, cfg.cognition_max_service_per_step)
```

In `_execute_scheduling`, convert capacities to per-task actual service, call `truth.apply_service`, calculate weighted satisfaction and high-priority satisfaction, and deduct service energy. Preserve existing scheduling difference rewards, but base the difference on the new service utility.

- [ ] **Step 5: Update reward and info fields**

Add actual service and backlog components to the shared and per-agent scheduling reward. Add these info keys:

```python
"total_arrivals", "queue_overflow", "total_queue_length",
"scheduling_served_data", "service_rate",
"weighted_demand_satisfaction", "high_priority_service_rate",
"service_energy_consumption", "per_agent_served_data"
```

Use `service_rate = total_served / max(total_arrivals + queue_before_service, 1e-6)` and clamp it to `[0, 1]`.

- [ ] **Step 6: Run the service tests and verify they pass**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "scheduling_reduces or same_band" -q`

Expected: PASS.

- [ ] **Step 7: Commit the service loop**

```bash
git add tests/test_dynamic_business_queue.py configs/scenario_config.py envs/resource_cognition_env.py
git commit -m "feat: connect scheduling to dynamic service queues"
```

### Task 4: Extend local observations and structured encoder dimensions

**Files:**
- Modify: `tests/test_dynamic_business_queue.py`
- Modify: `envs/resource_cognition_env.py`
- Modify: `configs/scenario_config.py`
- Modify: `agents/ppo/models.py`

**Interfaces:**
- Resource task slot dimension changes from `12` to `17`.
- Resource message slot dimension changes from `12` to `17`.
- `ResourceCognitionEnv._compute_local_obs_dim()` and `ScenarioConfig.get_resource_cognition_local_obs_dim()` return the same dimension.

- [ ] **Step 1: Add a failing observation contract**

```python
def test_dynamic_queue_observation_has_fixed_dimension_without_truth_leak():
    env = make_env(num_tasks=2, max_steps=3)
    first = env.reset(seed=9)["local_obs"].copy()
    env.task_truth.queue_lengths[:] = 10.0
    second = env.get_local_obs(0)
    assert second.shape == first[0].shape
    assert np.allclose(second, first[0])
```

- [ ] **Step 2: Run the test and verify it fails**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "observation" -q`

Expected: FAIL because the existing resource slot dimensions do not include queue fields and the test fixture expects the new fixed layout.

- [ ] **Step 3: Implement the fixed observation layout**

Append normalized queue estimate, queue uncertainty, queue AoI, queue confidence, and arrival estimate to each visible task slot. Append the same queue fields and arrival estimate to each message slot. Update both config and environment dimension calculations to `6 + 17 * max_task_slots + 17 * max_obs_uavs`.

Update `ResourceCognitionObsSliceSpec` defaults and any structured slicing in `agents/ppo/models.py` to use task/message slot dimensions of `17`. Keep action dimensions unchanged.

- [ ] **Step 4: Add sensing and accepted-message observation tests**

Assert that changing hidden `queue_lengths` alone leaves local observations unchanged, while a selected sensing action changes the queue belief and a delivered accepted message changes the receiver queue belief.

- [ ] **Step 5: Run the observation and model-shape tests**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "observation or model" -q`

Expected: PASS.

- [ ] **Step 6: Commit the observation contract**

```bash
git add tests/test_dynamic_business_queue.py envs/resource_cognition_env.py configs/scenario_config.py agents/ppo/models.py
git commit -m "feat: expose dynamic queue beliefs to resource policies"
```

### Task 5: Add metrics, documentation, and compatibility verification

**Files:**
- Modify: `tests/test_dynamic_business_queue.py`
- Modify: `envs/metrics.py`
- Modify: `README.md`
- Modify: `train.md`

**Interfaces:**
- `EpisodeMetrics` records arrivals, served data, queue length, service rate, satisfaction, priority assurance, overflow, and service energy.
- `MetricTracker.aggregate()` returns mean versions of the new final and total metrics.

- [ ] **Step 1: Add a failing metrics test**

```python
def test_metrics_collect_dynamic_service_fields():
    metrics = EpisodeMetrics()
    metrics.update(1.0, {
        "total_arrivals": 3.0,
        "scheduling_served_data": 2.0,
        "total_queue_length": 4.0,
        "service_rate": 0.5,
        "weighted_demand_satisfaction": 0.25,
        "high_priority_service_rate": 0.75,
        "queue_overflow": 1.0,
        "service_energy_consumption": 2.0,
    })
    summary = metrics.summary()
    assert summary["total_arrivals"] == 3.0
    assert summary["total_served_data"] == 2.0
    assert summary["final_total_queue"] == 4.0
    assert summary["weighted_demand_satisfaction"] == 0.25
```

- [ ] **Step 2: Run the metrics test and verify it fails**

Run: `python -m pytest tests/test_dynamic_business_queue.py -k "metrics_collect" -q`

Expected: FAIL because `EpisodeMetrics` does not yet collect the new fields.

- [ ] **Step 3: Implement metrics and documentation**

Add per-step lists and final values to `EpisodeMetrics`, include them in `summary`, and include mean values in `MetricTracker.aggregate`. Update README and `train.md` with the resource action layout, queue semantics, service metrics, and checkpoint incompatibility warning.

- [ ] **Step 4: Run all contract tests**

Run: `python -m pytest tests/test_dynamic_business_queue.py -q`

Expected: PASS.

- [ ] **Step 5: Run static and runtime verification**

Run:

```bash
python -m py_compile configs/scenario_config.py envs/task_model.py envs/communication_model.py envs/resource_cognition_env.py envs/metrics.py agents/ppo/models.py
python runners/train_ppo_deployment.py --method-name mcg_ppo_resource_cognition --num-updates 1 --episodes-per-update 1 --eval-episodes 1 --device cpu
python runners/train_ppo_deployment.py --method-name mcg_ppo --num-updates 1 --episodes-per-update 1 --eval-episodes 1 --device cpu
git diff --check
```

Expected: compilation succeeds, both methods complete one update, and `git diff --check` reports no whitespace errors.

- [ ] **Step 6: Commit the completed feature**

```bash
git add tests/test_dynamic_business_queue.py envs/metrics.py README.md train.md
git commit -m "feat: report dynamic service assurance metrics"
```

## Self-Review Checklist

- Spec coverage: hidden dynamic truth, local queue beliefs, message fusion, service capacity, queue transition, reward, observation shape, metrics, compatibility, and legacy smoke tests are covered by Tasks 1-5.
- Placeholder scan: all implementation steps contain concrete files, interfaces, commands, and expected results.
- Type consistency: task truth methods operate on `[num_tasks]`; local beliefs and messages use `[num_agents, num_tasks]`; environment service output uses per-task and per-agent float arrays.
- Scope: explicit user queues and bandwidth allocation are intentionally excluded from this implementation.
