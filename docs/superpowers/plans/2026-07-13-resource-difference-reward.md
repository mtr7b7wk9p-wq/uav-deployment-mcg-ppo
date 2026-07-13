# Resource Cognition Difference Reward Implementation Plan

> **For agentic workers:** Execute inline with verification checkpoints. The user waived failing-test-first for this workstream.

**Goal:** Train formal resource-cognition MCG-PPO with exact per-UAV counterfactual rewards.

**Architecture:** Compute exact sensing difference rewards from per-UAV belief-quality changes, attribute communication fusion to message senders, and pass the resulting vector through the existing PPO rollout path. Preserve scalar rewards for all baselines and public environment compatibility.

**Tech Stack:** Python, NumPy, PyTorch, existing shared PPO runner.

## Global Constraints

- Do not change the legacy coverage reward.
- Do not change the resource baseline reward.
- Do not replay stochastic environment transitions.
- Do not add dependencies or redundant reward classes.

### Task 1: Reward Configuration

**Files:** `configs/scenario_config.py`, `configs/ablation_config.py`

- [x] Add a disabled-by-default per-agent cognition reward flag and contribution weight.
- [x] Enable the flag only for `mcg_ppo_resource_cognition`.

### Task 2: Counterfactual Reward

**Files:** `envs/resource_cognition_env.py`

- [x] Capture per-UAV quality before and after local sensing.
- [x] Compute exact team-quality differences without environment replay.
- [x] Attribute accepted fusion gain and attempted communication cost to senders.
- [x] Allocate local movement, sensing, and repeated-task costs.
- [x] Return the mean reward and expose component vectors in `info`.

### Task 3: PPO Rollout Integration

**Files:** `agents/ppo/buffer.py`, `runners/train_ppo_deployment.py`

- [x] Accept scalar or shape-`[num_agents]` rewards in the buffer.
- [x] Use `info["per_agent_rewards"]` when present and preserve scalar fallback.

### Task 4: Verification

**Files:** no production file changes

- [x] Run syntax and direct counterfactual identity checks.
- [x] Run one-update CPU training for both resource methods and legacy MCG-PPO.
- [x] Run `git diff --check` and inspect the final diff.
