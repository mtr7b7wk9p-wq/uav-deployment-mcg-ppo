# Limited Neighbor Communication Design

## Goal

Extend `mcg_ppo_resource_cognition` from isolated local beliefs to limited, on-demand neighbor cognition. A UAV may change another UAV's belief only through a delivered and accepted message.

## Communication Protocol

- Communication is event-triggered after explicit local sensing; it is not a new policy action in this phase.
- A sender can address only UAVs inside `cognition_communication_radius`.
- Each message carries sender, receiver, task, estimate, uncertainty, confidence, AoI, creation step, and arrival step.
- Messages are subject to deterministic step delay, stochastic packet loss, and a per-sender message limit.
- Sending decisions use only sender-side information: task priority, confidence, and freshness. They never inspect the receiver's private belief.

## Fusion Protocol

- Delivery alone does not force an update. The receiver rejects stale, low-confidence, or non-improving information.
- Effective confidence decays exponentially with message age.
- Accepted messages combine the receiver estimate and message estimate by confidence weights.
- Only the addressed receiver-task entry changes; other UAV beliefs remain untouched.
- Fusion gain is the receiver's local cognitive-quality improvement caused by accepted messages.

## Environment Step Order

1. Decode actions against the previous local task slots.
2. Move UAVs and age every local belief.
3. Apply each UAV's explicit local sensing update.
4. Deliver messages whose arrival step is now due and fuse accepted messages.
5. Generate new event-triggered messages from this step's local sensing results.
6. Deliver zero-delay messages, if configured.
7. Compute sensing, fusion, communication, repeat, and movement reward terms.
8. Build the next local observations.

## Metrics

Record attempted, dropped, delivered, accepted, and pending message counts, communication cost, fusion gain, and acceptance ratio. Existing coverage metrics and old environments remain unchanged.

## Validation

- Out-of-range UAVs never receive messages.
- Delayed messages do not update a receiver before arrival.
- Packet-loss rate `1.0` prevents every delivery.
- A delivered, fresher message updates only its receiver.
- A stale or weaker message is rejected.
- Resource-cognition and legacy MCG-PPO smoke training both complete.
