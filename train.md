## Quick Start
### 1. ppo
```
python runners/train_ppo_deployment.py --method-name ppo_main
```
### 2. mcg_ppo
```
python runners/train_ppo_deployment.py --method-name mcg_ppo
```
### 3. mcg_ppo_sensing
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_sensing
```
### 4. ppo_resource_cognition
```
python runners/train_ppo_deployment.py --method-name ppo_resource_cognition
```
### 5. mcg_ppo_resource_cognition
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_resource_cognition
```
Resource cognition uses aggregate business queues for each area-band task. Queues receive stochastic arrivals and are reduced only by actual scheduled service. The service rate is constrained by link quality, spectrum availability, same-band interference, and UAV service energy. Local task and message slots now contain 17 features each, so the default resource observation dimension is 210; old resource checkpoints are not compatible.
资源认知任务使用“区域-频段”单元，同时维护频谱占用和需求强度的局部认知；链路质量来自局部几何关系，真实任务优先级不进入局部观测。正式 `mcg_ppo_resource_cognition` 还启用逐 UAV 反事实贡献奖励。
资源动作中，前 5 个为移动动作，随后是感知可见任务槽，最后是调度可见任务槽。调度奖励由需求满足、频谱可用性、链路质量、同频冲突和服务能耗共同决定。
### 6. maddpg
```
python runners/train_ppo_deployment.py --method-name maddpg
```
### 7. ippo
```
python runners/train_ppo_deployment.py --method-name ippo
```
### 7. mcg_ppo_no_graph
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_no_graph
```
### 8. mcg_ppo_no_mc_reward
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_no_mc_reward
```
### 9. mcg_ppo_no_overlap_penalty
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_no_overlap_penalty
```
### 10. mcg_ppo_no_guidance
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_no_guidance
```
### 7.主方法与ppo对比试验
```
python runners/compare_deployment_methods.py --methods ppo_main mcg_ppo ippo greedy_local
```

### 8.所有对比实验图
```
python runners/compare_deployment_methods.py --methods random_masked greedy_local ppo_main mcg_ppo ippo
```
### 9. 消融实验
```
python runners/compare_deployment_methods.py --methods ppo_main mcg_ppo mcg_ppo_no_graph mcg_ppo_no_mc_reward mcg_ppo_no_overlap_penalty mcg_ppo_no_guidance
```
### 10. 单结果绘图
```
python runners/plot_single_experiment.py --exp_dir runners/results/train/20260329_140943_train_ppo_main_ppo_main_cfgppo_main_m5_u20/logs/training_log.json"
```
### 11. 总绘图
```
python runners/replot_from_update_logs.py --compare-dirs results/train/ppo_main results/train/ippo results/train/mcg_ppo --compare-labels MCG-PPO IPPO PPO --ablation-dirs results/train/mcg_ppo results/train/mcg_ppo_no_graph results/train/mcg_ppo_no_mc_reward results/train/mcg_ppo_no_overlap_penalty results/train/mcg_ppo_no_guidance --ablation-labels MCG-PPO "w/o Graph" "w/o Reward" "w/o Overlap Penalty" "w/o Guidance" --output-dir results/train/paper_training_curves --smooth-window 30
```
### 12. 绘图
```
python runners/replot_from_update_logs.py --compare-dirs results/train/ppo_main results/train/ippo results/train/mcg_ppo  --compare-labels MCG_PPO IPPO PPO --ablation-dirs results/train/mcg_ppo_no_graph results/train/mcg_ppo_no_overlap_penalty results/train/mcg_ppo_no_guidance --ablation-labels MCG-PPO "w/o Graph" "w/o Overlap Penalty" "w/o Guidance"
```
