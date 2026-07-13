# Resource Scheduling Design

## Action Contract

Resource cognition keeps the existing movement and sensing actions, then adds one scheduling action for each visible task slot:

- `0..4`: stay / move;
- `5..(5+K-1)`: sense visible task slot;
- `(5+K)..(5+2K-1)`: schedule visible task slot.

The selected task identifies both an area and its band. An invalid or padded slot is masked. Scheduling is one-step service allocation; there is no center, cluster head, or persistent global assignment.

## Service Objective

The environment evaluates each scheduled task using hidden demand, hidden spectrum occupancy, local link quality, and the UAV's remaining energy. Same-band nearby assignments create interference and reduce service quality. The team scheduling utility is the priority-weighted served-demand quality after conflict reduction.

The shared resource reward adds scheduling utility, energy cost, and conflict penalty. Formal MCG resource cognition receives a per-UAV scheduling difference reward: the team's utility with all assignments minus the utility after removing that UAV's assignment.

## Local Information Boundary

The policy only sees its local spectrum and demand beliefs, local link quality, and own remaining energy. Hidden truth is used only by the environment objective and diagnostics. Scheduling utility estimated from local beliefs is reported for analysis but is not copied into another UAV's observation.

## Compatibility

- Coverage environments and `mcg_ppo` action semantics are unchanged.
- The resource environment keeps the `(obs, reward, done, info)` step interface.
- Resource action dimension becomes `5 + 2 * cognition_max_task_slots`.
- Existing resource checkpoints are incompatible because observation, action, and reward semantics change.
