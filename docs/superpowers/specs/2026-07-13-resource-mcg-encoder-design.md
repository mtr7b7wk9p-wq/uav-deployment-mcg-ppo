# Resource Cognition MCG Encoder Design

## Method Identity

- `ppo_resource_cognition` preserves the current flattened-observation shared PPO as the resource-cognition baseline.
- `mcg_ppo_resource_cognition` keeps the formal method name but switches to a dedicated structured encoder.
- Existing checkpoints written under `mcg_ppo_resource_cognition` by the old MLP prototype are architecture-incompatible and are not silently loaded or migrated.

## Observation Contract

The local observation has three blocks:

1. Self state: 6 values.
2. Visible task slots: `cognition_max_task_slots x 8` values.
3. Received neighbor-message slots: `max_obs_uavs x 8` values.

Message slots contain only delivered communication summaries: sender identity, task identity, estimate, uncertainty, confidence, message age, fusion acceptance, and a valid marker. Nearby UAV ground truth is not used as a substitute for communication.

## Network Architecture

`ResourceCognitionEncoder` parses the fixed observation layout and creates:

- a self embedding;
- per-task embeddings aggregated by self-conditioned masked attention and masked max pooling;
- per-message embeddings aggregated by self-conditioned masked attention and masked max pooling.

The three contexts are fused into one latent vector for the actor or critic. Actor and critic retain separate encoders, matching the existing PPO architecture. No external graph-learning dependency is added.

## Fair Comparison

Both methods use the same environment, local observations, action masks, reward, communication, rollout protocol, and PPO hyperparameters. The baseline uses a flat MLP; the formal MCG method changes only the observation encoder.

## Validation

- Both method names resolve to the resource-cognition environment.
- The baseline constructs `MaskedActorNet` and `LocalCriticNet`.
- The MCG method constructs resource-specific actor and critic encoders.
- Padded task and message slots contribute zero context.
- Delivered message summaries appear only in the addressed receiver observation.
- Both resource methods and legacy `mcg_ppo` complete one-update CPU smoke training.
