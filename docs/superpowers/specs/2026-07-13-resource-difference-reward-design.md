# Resource Cognition Difference Reward Design

## Scope

- `ppo_resource_cognition` remains the flat-network, shared-team-reward baseline.
- `mcg_ppo_resource_cognition` enables per-UAV resource-cognition rewards.
- Legacy coverage methods and the environment step tuple remain unchanged.

## Counterfactual Contribution

For UAV `i`, the sensing contribution is

`D_i = Q_after_all_sensing - Q_after_sensing_without_i`.

Local sensing changes only UAV `i`'s belief, so the counterfactual is computed exactly by replacing its post-sensing local quality with its pre-sensing quality. This avoids copying or replaying the stochastic environment.

The raw difference is retained in metrics. Its training component is multiplied by the task count so that adding more task units does not dilute the reward scale.

Delivered-message fusion gain is attributed to the original sender. Each UAV pays its own movement, sensing, repeated-task, and attempted-message costs.

## Compatibility

`ResourceCognitionEnv.step` still returns one scalar reward for metrics and compatibility. When enabled, `info["per_agent_rewards"]` contains the training reward vector and the scalar reward is its mean. `PPOBuffer` accepts either a scalar, which it broadcasts, or a vector with one value per UAV.

## Validation

- The vector has shape `[num_agents]` and finite values.
- The reported counterfactual contribution equals the direct team-quality difference.
- A delayed fused message rewards its sender, not its receiver.
- Scalar reward equals the mean vector reward when difference rewards are enabled.
- Resource baseline and coverage methods still use scalar reward broadcasting.
