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

Training outputs are written to `results/train` and are intentionally excluded from version control.
