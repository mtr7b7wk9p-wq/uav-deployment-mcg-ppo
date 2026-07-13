# Limited Neighbor Communication Implementation Plan

> **For agentic workers:** Execute this plan task-by-task with verification checkpoints. The user waived failing-test-first for this workstream.

**Goal:** Add event-triggered, radius-limited, delayed, lossy messages and confidence-aware belief fusion to the resource-cognition environment.

**Architecture:** Keep message transport in `envs/communication_model.py`, belief fusion in `envs/task_model.py`, and step orchestration in `envs/resource_cognition_env.py`. Preserve the coverage environments and method defaults outside resource cognition.

**Tech Stack:** Python, NumPy, dataclasses, existing PPO runner.

## Global Constraints

- No global belief synchronization.
- No receiver-private information in sender message selection.
- No new policy action in this phase.
- No runtime dependency additions.
- Preserve old coverage training behavior.

### Task 1: Communication Configuration and Transport

**Files:**
- Modify: `configs/scenario_config.py`
- Create: `envs/communication_model.py`

- [x] Add radius, delay, loss, cost, message cap, value threshold, fusion threshold, and freshness decay configuration.
- [x] Implement immutable `CognitionMessage` records.
- [x] Implement reproducible packet loss, pending queues, and due-message delivery.
- [x] Verify radius-independent transport behavior for delay zero, delay one, and full loss.

### Task 2: Receiver-Side Belief Fusion

**Files:**
- Modify: `envs/task_model.py`

- [x] Add confidence- and freshness-aware receiver fusion.
- [x] Reject stale, weak, and non-improving messages.
- [x] Return accepted flag and local quality gain.
- [x] Verify that one receiver update never changes another UAV's belief.

### Task 3: Environment Integration

**Files:**
- Modify: `envs/resource_cognition_env.py`

- [x] Build sender-side message values from local sensing results.
- [x] Restrict recipients to the communication radius and nearest message cap.
- [x] Deliver queued messages before new sends and zero-delay messages after sends.
- [x] Subtract attempted communication cost and add fusion gain reward.
- [x] Expose communication counts, pending queue size, acceptance ratio, and fusion gain.

### Task 4: Metrics and Verification

**Files:**
- Modify: `envs/metrics.py`
- Modify: `README.md`

- [x] Aggregate message attempts, deliveries, acceptance ratio, communication cost, and fusion gain.
- [x] Document event-triggered communication semantics and command.
- [x] Run syntax, structure, focused communication behavior, resource-cognition training, and legacy MCG-PPO regression checks.
