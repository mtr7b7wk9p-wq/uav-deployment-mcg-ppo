# Resource Cognition MCG Encoder Implementation Plan

> **For agentic workers:** Execute inline with verification checkpoints. The user waived failing-test-first for this workstream.

**Goal:** Separate the flat PPO resource baseline from a formal MCG-PPO resource method with task and received-message aggregation.

**Architecture:** Extend the resource observation with delivered-message slots. Add a dedicated PyTorch encoder that uses self-conditioned masked attention plus max pooling over task and message sets. Select the encoder through PPO configuration while keeping the flat baseline unchanged.

**Tech Stack:** Python, NumPy, PyTorch, existing shared PPO runner.

## Global Constraints

- Do not reuse the coverage structured encoder.
- Do not expose undelivered neighbor state.
- Do not add dependencies.
- Do not modify legacy coverage method behavior.

### Task 1: Method Split

**Files:** `configs/ablation_config.py`, `baselines/method_registry.py`, `README.md`, `train.md`

- [x] Register `ppo_resource_cognition` with the flat PPO configuration.
- [x] Keep `mcg_ppo_resource_cognition` and enable the new encoder only for it.
- [x] Document checkpoint incompatibility and both training commands.

### Task 2: Message Observation Contract

**Files:** `envs/resource_cognition_env.py`, `configs/scenario_config.py`

- [x] Store the latest delivered message per receiver-sender pair.
- [x] Replace geometric neighbor slots with 8-value delivered-message slots.
- [x] Update local observation dimensions.
- [x] Verify that only the addressed receiver sees a message slot.

### Task 3: Resource Encoder

**Files:** `agents/ppo/models.py`, `agents/ppo/ppo_agent.py`

- [x] Add resource observation slicing with strict dimension validation.
- [x] Add self-conditioned masked task aggregation.
- [x] Add self-conditioned masked message aggregation.
- [x] Add resource actor and critic networks.
- [x] Route `LocalActorCritic` through the resource encoder flag.

### Task 4: Runner Integration and Verification

**Files:** `runners/train_ppo_deployment.py`

- [x] Pass resource slot dimensions into `PPOConfig`.
- [x] Confirm baseline and MCG network classes differ.
- [x] Run syntax and structure checks.
- [x] Run one-update CPU training for both resource methods and legacy MCG-PPO.
