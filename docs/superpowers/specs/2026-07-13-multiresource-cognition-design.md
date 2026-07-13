# Multi-Resource Cognition Design

## Scenario Contract

The resource-cognition environment models a post-disaster area without a stable center, preset cluster head, or fixed cluster boundary. Each cognition task is a local `(area, band)` unit. Spectrum occupancy and demand intensity are hidden task states; link quality is a local geometric feature; remaining UAV time is the UAV's local energy proxy.

## Belief Contract

Every UAV maintains independent spectrum and demand beliefs for every task. Each belief has an estimate, uncertainty, age, and confidence. Sensing updates both resource dimensions. Delivered messages carry both dimensions and only update the addressed receiver.

True task priority remains an environment-side scheduling weight. It is never placed in a local observation or message. Local observations use the UAV's estimated demand and local link quality instead.

## Compatibility

- Movement and explicit task-slot sensing actions remain unchanged.
- The public environment step tuple remains `(observation, reward, done, info)`.
- The resource observation expands from 8-value task/message slots to 12-value slots.
- Coverage environments and legacy method behavior remain unchanged.

## Validation

- Local observations contain no true task priority, true spectrum state, or true demand state.
- Per-UAV belief arrays have shape `[num_agents, num_tasks]` for both resource dimensions.
- A sensing action reduces both selected uncertainties and resets both selected ages.
- Message fusion updates only the addressed receiver's two belief dimensions.
- Flat PPO and structured MCG-PPO both complete one CPU update with the expanded observation.

