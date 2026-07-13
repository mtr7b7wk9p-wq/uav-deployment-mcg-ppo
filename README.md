# UAV Fuxian Deployment

Multi-UAV deployment and decentralized PPO experiments for post-disaster scenarios.

## Quick Start

Install the runtime dependencies:

```powershell
python -m pip install numpy torch matplotlib
```

Train the baseline PPO:

```powershell
python runners/train_ppo_deployment.py --method-name ppo_main
```

Train the original MCG-PPO coverage prototype:

```powershell
python runners/train_ppo_deployment.py --method-name mcg_ppo
```

Train the trusted-sensing prototype with uncertainty, information age, task priority, and repeated-sensing penalties:

```powershell
python runners/train_ppo_deployment.py --method-name mcg_ppo_sensing
```

Train the flat-MLP resource-cognition PPO baseline:

```powershell
python runners/train_ppo_deployment.py --method-name ppo_resource_cognition
```

Train the resource-cognition MCG-PPO with task/message aggregation and per-UAV difference rewards:

```powershell
python runners/train_ppo_deployment.py --method-name mcg_ppo_resource_cognition
```

`mcg_ppo` remains the coverage prototype. `mcg_ppo_sensing` is the earlier automatic-sensing prototype. `ppo_resource_cognition` is the flat-network, shared-reward cognition baseline, while `mcg_ppo_resource_cognition` uses the dedicated structured cognition encoder and per-UAV counterfactual contribution rewards. Old resource checkpoints are not compatible after the structured-observation, dual-resource, and scheduling-action changes.

The resource-cognition environment keeps one belief map per UAV. A local sensing result reaches another UAV only through an event-triggered message within the configured communication radius. Messages are delayed, may be dropped, and are fused only when their confidence and freshness improve the receiver's local belief.

Each resource task represents an area-band pair, with 3 bands by default. Spectrum occupancy and demand intensity are hidden states with independent local estimates, uncertainty, age, and confidence. Each task now also has a hidden dynamic business queue and stochastic arrival rate. Sensing and accepted neighbor messages update local queue beliefs; the true queue is never exposed directly. Resource local observations use 6 self features, 8 task slots with 17 features each, and 2 received-message slots with 17 features each, for a default dimension of 176. Resource actions are movement, sensing a visible task slot, or scheduling a visible task slot. Scheduling computes actual served data in Mbit from queue backlog, air-to-ground path loss, received power, same-band interference, noise, SINR, Shannon rate, and service energy. It reports service rate, demand satisfaction, priority assurance, queue overflow, path loss, SINR, physical capacity, and outage metrics. Old resource checkpoints are incompatible with this observation layout.

Training outputs are written to `results/train` and are intentionally excluded from version control.
