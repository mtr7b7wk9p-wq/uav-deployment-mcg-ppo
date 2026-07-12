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
### 3. maddpg
```
python runners/train_ppo_deployment.py --method-name maddpg
```
### 4. ippo
```
python runners/train_ppo_deployment.py --method-name ippo
```
### 3. mcg_ppo_no_graph
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_no_graph
```
### 4. mcg_ppo_no_mc_reward
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_no_mc_reward
```
### 5. mcg_ppo_no_overlap_penalty
```
python runners/train_ppo_deployment.py --method-name mcg_ppo_no_overlap_penalty
```
### 6. mcg_ppo_no_guidance
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
