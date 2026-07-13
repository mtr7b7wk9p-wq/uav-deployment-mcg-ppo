# Dynamic Business Queue and Service Assurance Design

## Goal

Upgrade the resource-cognition environment from static demand scalars to dynamic business demand. Scheduling actions must affect served data, queue backlog, and communication assurance, while preserving the current region-band tasks, local cognition, limited neighbor communication, and PPO action interface.

## Modeling Boundary

Use a task-level aggregate queue in this stage. Each task represents a spatial region and band pair `q=(g,b)`. Its queue is the aggregate business demand of users in that region-band. Explicit user association, user-level queues, and bandwidth allocation actions are deferred to a later extension.

The hidden environment task state contains:

- `spectrum_state`: spectrum occupancy;
- `demand_level`: current business intensity;
- `arrival_rate`: per-step business arrival amount;
- `queue_length`: pending business amount;
- `priority`: service priority.

Each UAV observes only demand, queue, age, and confidence estimates obtained through sensing or accepted neighbor messages. The true queue may be used for environment execution and evaluation, but never as a local observation input.

## State Transition

Each environment step follows this order:

1. Generate business arrivals and update the hidden queue, clipped by the queue capacity.
2. Apply movement, explicit sensing, and message transmission/fusion.
3. Build schedule assignments and calculate link quality, spectrum availability, same-band interference, and remaining-energy constraints.
4. Calculate service capacity for each scheduled task and subtract actual served data from the hidden queue.
5. Update service rate, demand satisfaction, priority assurance, and service energy metrics.
6. Expose the updated queue summary only through the next local observation after the cognition rules are applied.

The queue transition is:

```text
queue_next = clip(queue_before + arrivals - served, 0, queue_capacity)
```

Served data cannot exceed the current queue, effective link capacity, or the configured per-step service limit. The implementation may account for arrivals before service in the same step, but must use one consistent ordering in both code and metrics.

## Service Model

For task `q` and UAV `i`, normalized service capacity is:

```text
capacity(i,q) = base_rate
                * link_quality(i,q)
                * spectrum_availability(q)
                * conflict_factor(i,q)
                * energy_factor(i)
```

Actual service is:

```text
served(i,q) = min(queue(q), capacity(i,q), per_step_service_limit)
```

Only one UAV may serve one task in a step. Different tasks using the same band within the interference radius reduce capacity according to their conflict count. Service energy is deducted from UAV remaining resources; insufficient resources reduce capacity or make service zero.

Scheduling utility is based on actual service rather than static demand alone:

- served data;
- weighted demand satisfaction;
- high-priority service assurance;
- queue backlog penalty;
- interference, communication, and service-energy penalties.

## Observation Changes

Keep fixed-size resource task slots. Add these aggregate business fields after the existing demand cognition fields:

- `queue_estimate`;
- `queue_uncertainty`;
- `queue_aoi`;
- `queue_confidence`;
- `arrival_estimate`.

All fields use fixed upper-bound normalization. Padded task slots remain zero, and the existing action mask prevents selection of invisible tasks.

## Cognition and Communication

Sensing updates spectrum, demand intensity, and queue estimates together. Neighbor messages carry demand-related summaries. A receiver fuses queue information only after a message is delivered and passes confidence/freshness checks. The true queue is used only for environment service execution and evaluation, never for action observations.

## Metrics

Add episode and aggregate metrics:

- `total_arrivals`;
- `total_served_data`;
- `mean_queue_length`;
- `final_total_queue`;
- `service_rate`;
- `weighted_demand_satisfaction`;
- `high_priority_service_rate`;
- `queue_overflow`;
- `service_energy_consumption`.

Keep existing cognition quality, AoI, message, conflict, and movement metrics so cognition improvement can be separated from service improvement.

## Compatibility

Only the resource-cognition environment state, reward, and fixed observation dimension change. The legacy coverage environment and `mcg_ppo` remain unaffected. Resource cognition model input and action dimensions will change, so old resource checkpoints are incompatible; README and `train.md` must state this explicitly.

## Validation

Before implementation, add failing contract tests for:

1. Without scheduling, queues increase only by arrivals.
2. Valid scheduling produces positive service and reduces a non-empty queue.
3. Service cannot exceed queue, capacity, or the per-step limit.
4. Same-band interference reduces service rate.
5. Hidden queue changes do not directly change local observations; sensing or accepted messages are required to update queue beliefs.
6. One-update training and the legacy `mcg_ppo` smoke test still run.
