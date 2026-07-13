# Multi-Resource Cognition Implementation Plan

> **For agentic workers:** Execute inline with verification checkpoints. The user waived failing-test-first for this workstream.

**Goal:** Align the resource-cognition environment with the thesis scenario by modeling spectrum and demand beliefs while removing true-priority leakage.

**Architecture:** Extend the hidden task truth and each UAV's local belief with a demand dimension. Keep link quality as a locally computable geometric feature and keep UAV energy in self state. Expand task and message observation slots to carry both belief dimensions without changing actions, PPO rollout APIs, or coverage methods.

**Tech Stack:** Python, NumPy, PyTorch, existing PPO runner.

## Global Constraints

- Do not expose true task priority, spectrum state, or demand state in local observations.
- Do not change movement or action semantics.
- Do not modify legacy coverage environments.
- Do not add dependencies.

### Task 1: Truth and Belief Model

**Files:** `envs/task_model.py`, `envs/communication_model.py`, `configs/scenario_config.py`

- [x] Add hidden demand intensity to task truth.
- [x] Add per-UAV demand estimate, uncertainty, age, and confidence arrays.
- [x] Update sensing and fusion to process spectrum and demand together.
- [x] Add normalized quality and error metrics for both dimensions.

### Task 2: Resource Environment Contract

**Files:** `envs/resource_cognition_env.py`

- [x] Sample hidden demand independently from hidden spectrum state.
- [x] Generate noisy demand observations during explicit sensing.
- [x] Remove true priority from local task slots.
- [x] Add estimated demand and local link quality to observations.
- [x] Expand message construction and fusion to the two-resource payload.

### Task 3: Encoder and Configuration Integration

**Files:** `agents/ppo/models.py`, `configs/scenario_config.py`, `runners/train_ppo_deployment.py`, `README.md`, `train.md`

- [x] Change resource task/message slot dimensions from 8 to 12.
- [x] Keep flat and structured resource policies compatible with the new dimension.
- [x] Document the thesis-aligned task semantics and observation layout.

### Task 4: Verification

**Files:** no new test file required

- [ ] Run syntax checks and direct belief-shape checks.
- [x] Verify no true-priority feature appears in local observations.
- [x] Verify sensing and message fusion affect both belief dimensions.
- [x] Run one-update CPU training for both resource methods and legacy `mcg_ppo`.
- [x] Run `git diff --check`.
