# Resource Scheduling Implementation Plan

> **For agentic workers:** Execute inline with verification checkpoints. The user waived failing-test-first for this workstream.

**Goal:** Connect trusted resource cognition to explicit spectrum-demand service scheduling with energy and interference constraints.

**Architecture:** Add masked schedule-task actions beside the existing movement and sensing actions. Evaluate service using hidden environment truth for the team objective, compute local-belief estimates for diagnostics, and add exact assignment-removal counterfactual contributions to formal MCG rewards.

**Tech Stack:** Python, NumPy, PyTorch, existing PPO runner.

## Global Constraints

- Do not change legacy coverage action or reward behavior.
- Do not expose hidden scheduling truth in local observations.
- Keep the public resource environment step tuple unchanged.
- Do not add dependencies.

### Task 1: Scheduling Configuration and Action Layout

**Files:** `configs/scenario_config.py`, `envs/resource_cognition_env.py`

- [x] Add scheduling utility, conflict, and service-energy parameters.
- [x] Add schedule-action offsets and expand only the resource action dimension.
- [x] Decode and mask schedule actions using visible task slots.

### Task 2: Resource Service Evaluation

**Files:** `envs/resource_cognition_env.py`

- [x] Compute priority-weighted served demand from hidden demand, spectrum availability, link quality, and energy.
- [x] Penalize same-band nearby assignments.
- [x] Consume service energy and expose team, estimated, and per-UAV scheduling metrics.

### Task 3: PPO Reward Integration

**Files:** `envs/resource_cognition_env.py`, `envs/metrics.py`, `README.md`, `train.md`

- [x] Add shared scheduling reward to the resource baseline.
- [x] Add assignment-removal counterfactual scheduling rewards to formal MCG.
- [x] Record scheduling utility, conflict, and service-energy metrics.
- [x] Document the new action semantics and checkpoint incompatibility.

### Task 4: Verification

**Files:** no new test file required

- [x] Verify schedule action masks and action dimensions.
- [x] Verify high-demand available tasks score above low-demand occupied tasks.
- [x] Verify removing one assignment changes only that UAV's counterfactual contribution and conflict behavior.
- [x] Run one-update CPU training for both resource methods and legacy `mcg_ppo`.
- [x] Run `git diff --check`.
