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

Train the resource-cognition MCG-PPO with task and received-message aggregation:

```powershell
python runners/train_ppo_deployment.py --method-name mcg_ppo_resource_cognition
```

`mcg_ppo` remains the coverage prototype. `mcg_ppo_sensing` is the earlier automatic-sensing prototype. `ppo_resource_cognition` is the flat-network cognition baseline, while `mcg_ppo_resource_cognition` uses the dedicated structured cognition encoder. Old MLP checkpoints previously written under `mcg_ppo_resource_cognition` are not compatible with the new encoder.

The resource-cognition environment keeps one belief map per UAV. A local sensing result reaches another UAV only through an event-triggered message within the configured communication radius. Messages are delayed, may be dropped, and are fused only when their confidence and freshness improve the receiver's local belief.

Training outputs are written to `results/train` and are intentionally excluded from version control.
