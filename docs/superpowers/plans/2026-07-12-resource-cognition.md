# Resource Cognition Implementation Plan

> **For agentic workers:** Implement this plan task-by-task with verification checkpoints. The user explicitly waived failing-test-first; use focused behavior checks instead.

**Goal:** Add an isolated resource-cognition environment that supports local trusted-state observation and explicit sensing actions while preserving the existing coverage experiments.

**Architecture:** Keep `DisasterDeploymentEnv` as the compatibility environment. Add a small task-state model and a dedicated cognition environment, then connect configuration, registry, training, metrics, and documentation through the existing runner interfaces.

**Tech Stack:** Python, NumPy, dataclasses, existing PPO runner, existing geometry utilities, JSON summaries.

## Global Constraints

- Do not change default `ppo_main` or `mcg_ppo` behavior.
- Do not add runtime dependencies.
- Do not expose global task truth in local policy observations.
- Do not treat movement as automatic sensing.
- Keep old coverage metric keys compatible.
- Validate every batch with `py_compile`, a focused behavior script, and a one-update CPU smoke run.

### Task 1: Task State Model

**Files:**
- Create: `envs/task_model.py`
- Modify: none
- Verify: inline Python behavior check

**Interfaces:**
- `TaskState`: immutable task metadata plus mutable cognitive state.
- `TaskStateBatch`: vectorized task state with `reset`, `age`, `sense`, and summary methods.

- [ ] Define task fields for position, band id, true state, estimate, uncertainty, AoI, priority, and confidence.
- [ ] Implement bounded uncertainty reduction and AoI aging.
- [ ] Implement weighted cognitive quality without coupling to the environment.
- [ ] Run a deterministic state transition check with one task and confirm bounds.

### Task 2: Resource Cognition Configuration

**Files:**
- Modify: `configs/scenario_config.py`
- Modify: `configs/ablation_config.py`
- Modify: `baselines/method_registry.py`

**Interfaces:**
- Add `resource_cognition` parameters without changing coverage defaults.
- Register `mcg_ppo_resource_cognition` as a separate method.

- [ ] Add task count, sensing slot count, sensing radius, band count, observation noise, AoI increment, and uncertainty target.
- [ ] Add action dimension helpers for the cognition mode.
- [ ] Register the method with a separate output/checkpoint name.
- [ ] Validate default coverage and new cognition configurations independently.

### Task 3: Dedicated Cognition Environment

**Files:**
- Create: `envs/resource_cognition_env.py`
- Modify: `envs/__init__.py`

**Interfaces:**
- `reset(seed=None) -> Dict[str, Any]`
- `step(actions) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]`
- `get_local_obs(agent_id) -> np.ndarray`
- `get_global_state() -> np.ndarray`

- [ ] Reuse annulus sampling and UAV movement without importing coverage reward logic.
- [ ] Build local task slots with visibility masks and neighbor summaries.
- [ ] Map explicit sensing actions to visible task slots.
- [ ] Apply sensing once per selected task, aggregate repeat count, and return reward breakdown.
- [ ] Add max-step, uncertainty-target, timeout, and stagnation termination.
- [ ] Run a deterministic two-UAV behavior check for movement, unique sensing, and repeated sensing.

### Task 4: Runner Integration

**Files:**
- Modify: `runners/train_ppo_deployment.py`
- Modify: `train.md`

**Interfaces:**
- Select the cognition environment only for `mcg_ppo_resource_cognition`.
- Preserve existing shared PPO rollout and evaluation APIs.

- [ ] Build method-specific environment factory and observation/action dimensions.
- [ ] Route cognition metrics to the best-checkpoint criterion.
- [ ] Add a one-update CPU command to `train.md`.
- [ ] Run the one-update CPU smoke command.

### Task 5: Metrics and Comparison Safety

**Files:**
- Modify: `envs/metrics.py`
- Modify: `baselines/compare_adapter.py`
- Modify: `utils/experiment_schema.py` only if required by schema validation

- [ ] Preserve coverage aggregate keys.
- [ ] Add cognition aggregate fields with zero-compatible fallbacks.
- [ ] Prevent coverage-only baselines from being silently compared as cognition baselines.
- [ ] Run existing structure smoke checks and parse one generated summary.

### Task 6: Verification and Documentation

**Files:**
- Modify: `README.md`
- Modify: `train.md`

- [ ] Run `python -m compileall` on edited modules.
- [ ] Run structure smoke checks.
- [ ] Run focused task-model and environment behavior checks.
- [ ] Run one-update CPU training for both old `mcg_ppo` and new cognition method.
- [ ] Record remaining research limitations: single-band prototype, simplified observation noise, no downstream allocator yet.
