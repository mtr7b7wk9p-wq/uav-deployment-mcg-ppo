from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from agents.ippo_agent import IPPOConfig, IndependentPPOAgent
from agents.ippo_buffer import IPPOBuffer
from agents.maddpg_agent import MADDPGAgent, MADDPGConfig
from agents.maddpg_buffer import MADDPGReplayBuffer
from agents.ppo.buffer import PPOBuffer
from agents.ppo.ppo_agent import PPOConfig, SharedPPOAgent
from configs.ablation_config import (
    AblationSpec,
    apply_ablation_to_scenario,
    build_base_scenario_config,
    build_method_ppo_config_kwargs,
    get_ablation_spec,
    resolve_method_name,
)
from envs.disaster_deployment_env import DisasterDeploymentEnv
from envs.resource_cognition_env import ResourceCognitionEnv
from envs.metrics import EpisodeMetrics, MetricTracker
from plotting.plot_scene import plot_training_history
from utils.experiment_schema import (
    SCHEMA_VERSION,
    SUMMARY_FILENAME,
    TRAIN_LOG_FILENAME,
    TRAIN_PLOT_FILENAME,
    build_paper_metric_row,
    build_reward_tail_mean,
    build_training_plot_series,
    make_ippo_train_stats_block,
    make_maddpg_train_stats_block,
    make_method_identity,
    make_paths_block,
    make_train_stats_block,
)
from utils.io import save_json
from utils.run_manager import build_run_dirs, build_run_name, save_manifest


@dataclass
class TrainConfig:
    total_updates: int = 500
    rollout_episodes_per_update: int = 8
    eval_interval: int = 20
    eval_episodes: int = 5
    save_interval: int = 50
    log_flush_interval: int = 20
    output_root: str = "results/train"
    exp_tag: str = ""
    seed: int = 42
    device: str = "cuda"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train registered deployment methods.")
    parser.add_argument("--method-name", type=str, default=None, help="Registered method_name, e.g. ppo_main / mcg_ppo / ippo / maddpg.")
    parser.add_argument("--config-name", type=str, default=None, help="Registered config_name. When provided, it will resolve to the matching method_name.")
    parser.add_argument("--output-root", type=str, default="results/train")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", type=str, default=None, help="cuda / cpu. Default is auto selection.")
    parser.add_argument("--total-updates", type=int, default=None, help="Total training updates. Default: 1500.")
    parser.add_argument("--rollout-episodes-per-update", type=int, default=None, help="Episodes collected per update. Default: 8.")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluate every N updates. Default: 120.")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Evaluation episodes per eval. Default: 5.")
    parser.add_argument("--save-interval", type=int, default=None, help="Save checkpoint every N updates. Default: 300.")
    parser.add_argument("--log-flush-interval", type=int, default=None, help="Flush training log every N updates. Default: 30.")
    return parser.parse_args()


def set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _copy_if_exists(src_path: str, dst_path: str) -> None:
    if not os.path.exists(src_path):
        return
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)


def build_training_env(scenario_cfg):
    """Select an environment without changing the legacy coverage path."""
    if bool(getattr(scenario_cfg, "use_resource_cognition", False)):
        return ResourceCognitionEnv(scenario_cfg)
    return DisasterDeploymentEnv(scenario_cfg)


def get_training_action_dim(scenario_cfg) -> int:
    if bool(getattr(scenario_cfg, "use_resource_cognition", False)):
        return int(scenario_cfg.get_resource_cognition_action_dim())
    return int(scenario_cfg.action_size)


def ensure_method_ckpt_alias(output_root: str, method_dir_name: str, src_path: str, filename: str) -> str:
    alias_dir = os.path.join(output_root, method_dir_name, "checkpoints")
    os.makedirs(alias_dir, exist_ok=True)
    dst_path = os.path.join(alias_dir, filename)
    shutil.copy2(src_path, dst_path)
    return dst_path


def sync_train_alias_artifacts(output_root: str, method_dir_name: str, dirs: Dict[str, str]) -> None:
    alias_root = os.path.join(output_root, method_dir_name)
    _copy_if_exists(os.path.join(dirs["run_dir"], "manifest.json"), os.path.join(alias_root, "manifest.json"))
    _copy_if_exists(os.path.join(dirs["run_dir"], SUMMARY_FILENAME), os.path.join(alias_root, SUMMARY_FILENAME))
    _copy_if_exists(os.path.join(dirs["log_dir"], TRAIN_LOG_FILENAME), os.path.join(alias_root, "logs", TRAIN_LOG_FILENAME))
    _copy_if_exists(os.path.join(dirs["plot_dir"], TRAIN_PLOT_FILENAME), os.path.join(alias_root, "plots", TRAIN_PLOT_FILENAME))


def run_one_episode_collect_shared(
    env: Any,
    agent: SharedPPOAgent,
    buffer: PPOBuffer,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    obs_dict = env.reset(seed=seed)
    local_obs = obs_dict["local_obs"]
    action_mask = obs_dict["action_mask"]

    num_agents = local_obs.shape[0]
    ep_metrics = EpisodeMetrics()
    done = False

    last_info: Dict[str, Any] = {
        "coverage_ratio": float(getattr(env, "coverage_ratio", 0.0)),
        "covered_users": int(getattr(env, "covered_mask", np.zeros((0,), dtype=bool)).sum()),
        "total_distance_per_uav": env.total_distance_per_uav.copy(),
        "remaining_time": env.remaining_time.copy(),
        "step": 0,
        "active_uav_count": int(np.sum(env.active_mask)) if hasattr(env, "active_mask") else int(num_agents),
        "termination_reason": "running",
    }

    while not done:
        actions, log_probs, values = agent.act(
            local_obs_batch=local_obs,
            action_mask_batch=action_mask,
        )

        next_obs_dict, reward, done, info = env.step(actions)

        buffer.add_multi_agent_step(
            local_obs_batch=local_obs,
            action_mask_batch=action_mask,
            action_batch=actions,
            log_prob_batch=log_probs,
            reward=reward,
            done=done,
            value_batch=values,
        )

        ep_metrics.update(reward, info)
        local_obs = next_obs_dict["local_obs"]
        action_mask = next_obs_dict["action_mask"]
        last_info = info

    last_values = np.zeros((num_agents,), dtype=np.float32)
    return {
        "episode_metrics": ep_metrics,
        "last_values": last_values,
        "last_info": last_info,
    }


def run_one_episode_collect_ippo(
    env: Any,
    agent: IndependentPPOAgent,
    buffer: IPPOBuffer,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    obs_dict = env.reset(seed=seed)
    local_obs = obs_dict["local_obs"]
    action_mask = obs_dict["action_mask"]

    num_agents = local_obs.shape[0]
    ep_metrics = EpisodeMetrics()
    done = False

    last_info: Dict[str, Any] = {
        "coverage_ratio": float(getattr(env, "coverage_ratio", 0.0)),
        "covered_users": int(getattr(env, "covered_mask", np.zeros((0,), dtype=bool)).sum()),
        "total_distance_per_uav": env.total_distance_per_uav.copy(),
        "remaining_time": env.remaining_time.copy(),
        "step": 0,
        "active_uav_count": int(np.sum(env.active_mask)) if hasattr(env, "active_mask") else int(num_agents),
        "termination_reason": "running",
    }

    while not done:
        actions, log_probs, values = agent.act(
            local_obs_batch=local_obs,
            action_mask_batch=action_mask,
        )

        next_obs_dict, reward, done, info = env.step(actions)

        buffer.add_step(
            local_obs_batch=local_obs,
            action_mask_batch=action_mask,
            action_batch=actions,
            log_prob_batch=log_probs,
            reward=reward,
            done=done,
            value_batch=values,
        )

        ep_metrics.update(reward, info)
        local_obs = next_obs_dict["local_obs"]
        action_mask = next_obs_dict["action_mask"]
        last_info = info

    last_values = np.zeros((num_agents,), dtype=np.float32)
    return {
        "episode_metrics": ep_metrics,
        "last_values": last_values,
        "last_info": last_info,
    }


def run_one_episode_collect_maddpg(
    env: Any,
    agent: MADDPGAgent,
    replay_buffer: MADDPGReplayBuffer,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    obs_dict = env.reset(seed=seed)
    local_obs = obs_dict["local_obs"]
    action_mask = obs_dict["action_mask"]

    num_agents = local_obs.shape[0]
    ep_metrics = EpisodeMetrics()
    done = False
    episode_update_stats: List[Dict[str, Any]] = []

    last_info: Dict[str, Any] = {
        "coverage_ratio": float(getattr(env, "coverage_ratio", 0.0)),
        "covered_users": int(getattr(env, "covered_mask", np.zeros((0,), dtype=bool)).sum()),
        "total_distance_per_uav": env.total_distance_per_uav.copy(),
        "remaining_time": env.remaining_time.copy(),
        "step": 0,
        "active_uav_count": int(np.sum(env.active_mask)) if hasattr(env, "active_mask") else int(num_agents),
        "termination_reason": "running",
    }

    while not done:
        actions, _, _ = agent.act(
            local_obs_batch=local_obs,
            action_mask_batch=action_mask,
            explore=True,
        )

        next_obs_dict, reward, done, info = env.step(actions)

        replay_buffer.add(
            obs=local_obs,
            action_mask=action_mask,
            actions=actions,
            reward=float(reward),
            next_obs=next_obs_dict["local_obs"],
            next_action_mask=next_obs_dict["action_mask"],
            done=done,
        )

        ep_metrics.update(reward, info)
        local_obs = next_obs_dict["local_obs"]
        action_mask = next_obs_dict["action_mask"]
        last_info = info

        if agent.can_update(replay_buffer):
            update_stats = agent.update(replay_buffer)
            episode_update_stats.append(update_stats)

    return {
        "episode_metrics": ep_metrics,
        "last_info": last_info,
        "episode_update_stats": episode_update_stats,
    }


@torch.no_grad()
def evaluate_policy(
    env_cfg,
    agent: Any,
    eval_episodes: int,
    base_seed: int = 10000,
) -> Dict[str, Any]:
    tracker = MetricTracker()

    for ep in range(eval_episodes):
        env = build_training_env(env_cfg)
        obs_dict = env.reset(seed=base_seed + ep)

        local_obs = obs_dict["local_obs"]
        action_mask = obs_dict["action_mask"]

        done = False
        ep_metrics = EpisodeMetrics()

        while not done:
            if hasattr(agent, "greedy_act"):
                actions = agent.greedy_act(
                    local_obs_batch=local_obs,
                    action_mask_batch=action_mask,
                )
            else:
                actions, _, _ = agent.act(
                    local_obs_batch=local_obs,
                    action_mask_batch=action_mask,
                )

            next_obs_dict, reward, done, info = env.step(actions)
            ep_metrics.update(reward, info)

            local_obs = next_obs_dict["local_obs"]
            action_mask = next_obs_dict["action_mask"]

        tracker.add_episode(ep_metrics)

    return tracker.aggregate()


def get_eval_selection_metric(scenario_cfg) -> str:
    return (
        "mean_final_cognitive_quality"
        if scenario_cfg.use_trusted_sensing or scenario_cfg.use_resource_cognition
        else "mean_final_coverage_ratio"
    )


def get_eval_selection_score(eval_summary: Dict[str, Any], scenario_cfg) -> float:
    return float(eval_summary.get(get_eval_selection_metric(scenario_cfg), 0.0))


def build_train_config(
    args: argparse.Namespace,
    ablation_spec: AblationSpec,
) -> TrainConfig:
    default_total_updates = 1500
    default_rollout_episodes_per_update = 8
    default_eval_interval = 120
    default_eval_episodes = 5
    default_save_interval = 300
    default_log_flush_interval = 30

    total_updates = args.total_updates if args.total_updates is not None else default_total_updates
    rollout_episodes_per_update = (
        args.rollout_episodes_per_update
        if args.rollout_episodes_per_update is not None
        else default_rollout_episodes_per_update
    )
    eval_interval = args.eval_interval if args.eval_interval is not None else default_eval_interval
    eval_episodes = args.eval_episodes if args.eval_episodes is not None else default_eval_episodes
    save_interval = args.save_interval if args.save_interval is not None else default_save_interval
    log_flush_interval = (
        args.log_flush_interval if args.log_flush_interval is not None else default_log_flush_interval
    )

    if total_updates <= 0:
        raise ValueError(f"--total-updates must be > 0, got {total_updates}")
    if rollout_episodes_per_update <= 0:
        raise ValueError(
            f"--rollout-episodes-per-update must be > 0, got {rollout_episodes_per_update}"
        )
    if eval_interval <= 0:
        raise ValueError(f"--eval-interval must be > 0, got {eval_interval}")
    if eval_episodes <= 0:
        raise ValueError(f"--eval-episodes must be > 0, got {eval_episodes}")
    if save_interval <= 0:
        raise ValueError(f"--save-interval must be > 0, got {save_interval}")
    if log_flush_interval <= 0:
        raise ValueError(f"--log-flush-interval must be > 0, got {log_flush_interval}")

    return TrainConfig(
        total_updates=int(total_updates),
        rollout_episodes_per_update=int(rollout_episodes_per_update),
        eval_interval=int(eval_interval),
        eval_episodes=int(eval_episodes),
        save_interval=int(save_interval),
        log_flush_interval=int(log_flush_interval),
        output_root=args.output_root,
        exp_tag=ablation_spec.config_name,
        seed=args.seed,
        device=args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
    )


def should_flush_training_log(update_idx: int, train_cfg: TrainConfig) -> bool:
    if update_idx <= 0:
        return False
    if update_idx == train_cfg.total_updates:
        return True
    if train_cfg.eval_interval > 0 and update_idx % train_cfg.eval_interval == 0:
        return True
    if train_cfg.save_interval > 0 and update_idx % train_cfg.save_interval == 0:
        return True
    if train_cfg.log_flush_interval > 0 and update_idx % train_cfg.log_flush_interval == 0:
        return True
    return False


def resolve_training_backend_note(method_name: str, ablation_spec: AblationSpec) -> str:
    if method_name == "ippo":
        return "Independent PPO: each UAV owns independent actor-critic parameters, buffer, and PPO update."
    if method_name == "maddpg":
        return (
            "Discrete-action MADDPG-style minimum implementation: decentralized actors output logits over discrete "
            "actions, centralized critics consume joint observations and joint one-hot actions, with replay buffer "
            "and target networks."
        )
    return "Unified shared-parameter PPO-family training entry."


def _build_common_run_context(
    method_name: str,
    train_cfg: TrainConfig,
    ablation_spec: AblationSpec,
):
    base_scenario_cfg = build_base_scenario_config()
    scenario_cfg = apply_ablation_to_scenario(base_scenario_cfg, method_name)

    set_global_seed(train_cfg.seed)

    if scenario_cfg.use_resource_cognition:
        run_tag = (
            f"rc_cfg{ablation_spec.config_name}"
            f"_m{scenario_cfg.max_candidate_uavs}"
            f"_t{scenario_cfg.num_cognition_tasks}"
        )
    else:
        run_tag = (
            f"{method_name}"
            f"_cfg{ablation_spec.config_name}"
            f"_m{scenario_cfg.max_candidate_uavs}"
            f"_u{scenario_cfg.num_users}"
        )
    if scenario_cfg.use_resource_cognition:
        run_name = build_run_name(prefix="train_rc", method="ppo", tag=run_tag)
    else:
        run_name = build_run_name(prefix="train", method=method_name, tag=run_tag)
    dirs = build_run_dirs(train_cfg.output_root, run_name)

    if method_name == "ippo":
        agent_type = "ippo"
    elif method_name == "maddpg":
        agent_type = "maddpg"
    else:
        agent_type = "shared_ppo"

    method_identity = make_method_identity(
        method_name=method_name,
        display_name=ablation_spec.effective_display_name,
        config_name=ablation_spec.config_name,
        checkpoint_dir_name=ablation_spec.effective_checkpoint_dir_name,
        output_dir_name=ablation_spec.effective_output_dir_name,
        checkpoint_name=ablation_spec.default_checkpoint_name,
        method_label=ablation_spec.method_label,
        trainer_family=ablation_spec.trainer_family,
        policy_family=ablation_spec.policy_family,
        agent_type=agent_type,
    )
    stable_output_dir = os.path.join(train_cfg.output_root, method_identity["output_dir_name"])
    paths_block = make_paths_block(
        run_dir=dirs["run_dir"],
        ckpt_dir=dirs["ckpt_dir"],
        plot_dir=dirs["plot_dir"],
        log_dir=dirs["log_dir"],
        stable_output_dir=stable_output_dir,
    )
    return scenario_cfg, run_name, dirs, method_identity, paths_block


def run_registered_shared_training(
    method_name: str,
    train_cfg: TrainConfig,
    ablation_spec: AblationSpec,
) -> None:
    scenario_cfg, run_name, dirs, method_identity, paths_block = _build_common_run_context(
        method_name=method_name,
        train_cfg=train_cfg,
        ablation_spec=ablation_spec,
    )

    local_obs_dim = int(scenario_cfg.get_local_obs_dim(method_name))
    ppo_cfg_kwargs = build_method_ppo_config_kwargs(
        method_name=method_name,
        local_obs_dim=local_obs_dim,
        action_dim=get_training_action_dim(scenario_cfg),
        device=train_cfg.device,
        max_obs_users=int(scenario_cfg.max_obs_users),
        max_obs_uavs=int(scenario_cfg.max_obs_uavs),
        num_direction_sectors=int(scenario_cfg.num_direction_sectors),
        num_radial_bins=int(scenario_cfg.num_radial_bins),
        resource_num_task_slots=int(scenario_cfg.cognition_max_task_slots),
        resource_num_message_slots=int(scenario_cfg.max_obs_uavs),
    )
    ppo_cfg = PPOConfig(**ppo_cfg_kwargs)

    save_manifest(
        run_dir=dirs["run_dir"],
        run_type="train",
        run_name=run_name,
        note=resolve_training_backend_note(method_name, ablation_spec),
        schema_version=SCHEMA_VERSION,
        method=method_identity,
        paths=paths_block,
        ablation_spec=ablation_spec.to_dict(),
        scenario_cfg=scenario_cfg,
        train_cfg=train_cfg,
        ppo_cfg=ppo_cfg,
        training_backend={
            "trainer_family": ablation_spec.trainer_family,
            "policy_family": ablation_spec.policy_family,
            "backend_impl": "shared_ppo_loop",
            "placeholder_backend": False,
        },
    )

    agent = SharedPPOAgent(ppo_cfg)
    buffer = PPOBuffer()

    update_logs: List[Dict[str, Any]] = []
    train_episode_history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []

    global_episode_idx = 0
    best_eval_metric = get_eval_selection_metric(scenario_cfg)
    best_eval_score = -1.0
    best_eval_summary: Dict[str, Any] = {}

    for update_idx in range(1, train_cfg.total_updates + 1):
        buffer.reset()
        rollout_tracker = MetricTracker()

        for _ in range(train_cfg.rollout_episodes_per_update):
            env = build_training_env(scenario_cfg)
            collected = run_one_episode_collect_shared(
                env=env,
                agent=agent,
                buffer=buffer,
                seed=train_cfg.seed + global_episode_idx,
            )

            ep_summary = rollout_tracker.add_episode(collected["episode_metrics"])
            ep_summary["global_episode_idx"] = global_episode_idx
            ep_summary["update_idx"] = update_idx
            ep_summary["method_name"] = method_identity["method_name"]
            ep_summary["display_name"] = method_identity["display_name"]
            ep_summary["config_name"] = method_identity["config_name"]
            ep_summary["trainer_family"] = method_identity.get("trainer_family")
            ep_summary["policy_family"] = method_identity.get("policy_family")
            ep_summary["agent_type"] = method_identity.get("agent_type")
            train_episode_history.append(ep_summary)

            global_episode_idx += 1

        last_values = np.zeros((scenario_cfg.max_candidate_uavs,), dtype=np.float32)
        raw_train_stats = agent.update(
            buffer=buffer,
            last_values=last_values,
            num_agents=scenario_cfg.max_candidate_uavs,
        )
        train_stats = make_train_stats_block(raw_train_stats)
        rollout_summary = rollout_tracker.aggregate()

        log_record = {
            "update_idx": update_idx,
            "global_episode_idx": global_episode_idx,
            "method_name": method_identity["method_name"],
            "display_name": method_identity["display_name"],
            "config_name": method_identity["config_name"],
            "trainer_family": method_identity.get("trainer_family"),
            "policy_family": method_identity.get("policy_family"),
            "agent_type": method_identity.get("agent_type"),
            "rollout_summary": rollout_summary,
            "train_stats": train_stats,
            "backend_impl": "shared_ppo_loop",
            "placeholder_backend": False,
            **train_stats,
        }
        update_logs.append(log_record)

        if scenario_cfg.use_resource_cognition:
            print(
                f"[{method_name}][Update {update_idx:04d}] "
                f"quality={rollout_summary.get('mean_final_cognitive_quality', 0.0):.4f}  "
                f"uncertainty={rollout_summary.get('mean_final_task_uncertainty', 0.0):.4f}  "
                f"aoi={rollout_summary.get('mean_final_task_aoi', 0.0):.2f}  "
                f"messages={rollout_summary.get('mean_total_messages_attempted', 0.0):.1f}  "
                f"fused={rollout_summary.get('mean_total_messages_fused', 0.0):.1f}  "
                f"return={rollout_summary.get('mean_episode_return', 0.0):.3f}"
            )
        elif scenario_cfg.use_trusted_sensing:
            print(
                f"[{method_name}][Update {update_idx:04d}] "
                f"quality={rollout_summary.get('mean_final_cognitive_quality', 0.0):.4f}  "
                f"uncertainty={rollout_summary.get('mean_final_task_uncertainty', 0.0):.4f}  "
                f"aoi={rollout_summary.get('mean_final_task_aoi', 0.0):.2f}  "
                f"repeat={rollout_summary.get('mean_mean_repeat_sensing_ratio', 0.0):.3f}  "
                f"move={rollout_summary.get('mean_total_move_distance', 0.0):.2f}  "
                f"return={rollout_summary.get('mean_episode_return', 0.0):.3f}"
            )
        else:
            print(
                f"[{method_name}][Update {update_idx:04d}] "
                f"coverage={rollout_summary.get('mean_final_coverage_ratio', 0.0):.4f}  "
                f"covered_users={rollout_summary.get('mean_final_covered_users', 0.0):.2f}  "
                f"move={rollout_summary.get('mean_total_move_distance', 0.0):.2f}  "
                f"overlap={rollout_summary.get('mean_mean_overlap_users_step', 0.0):.3f}  "
                f"return={rollout_summary.get('mean_episode_return', 0.0):.3f}"
            )

        if update_idx % train_cfg.eval_interval == 0 or update_idx == train_cfg.total_updates:
            eval_summary = evaluate_policy(
                env_cfg=scenario_cfg,
                agent=agent,
                eval_episodes=train_cfg.eval_episodes,
                base_seed=10000 + update_idx * 100,
            )
            eval_summary["update_idx"] = update_idx
            eval_summary["method_name"] = method_identity["method_name"]
            eval_summary["display_name"] = method_identity["display_name"]
            eval_summary["config_name"] = method_identity["config_name"]
            eval_summary["trainer_family"] = method_identity.get("trainer_family")
            eval_summary["policy_family"] = method_identity.get("policy_family")
            eval_summary["agent_type"] = method_identity.get("agent_type")
            eval_history.append(eval_summary)

            eval_score = get_eval_selection_score(eval_summary, scenario_cfg)
            eval_summary["selection_metric"] = best_eval_metric
            eval_summary["selection_score"] = eval_score
            if scenario_cfg.use_resource_cognition:
                print(
                    f"  [Eval-{method_name}] {best_eval_metric}={eval_score:.4f}  "
                    f"uncertainty={eval_summary.get('mean_final_task_uncertainty', 0.0):.4f}  "
                    f"messages={eval_summary.get('mean_total_messages_attempted', 0.0):.1f}  "
                    f"accept={eval_summary.get('mean_message_acceptance_ratio', 0.0):.3f}"
                )
            elif scenario_cfg.use_trusted_sensing:
                print(
                    f"  [Eval-{method_name}] {best_eval_metric}={eval_score:.4f}  "
                    f"uncertainty={eval_summary.get('mean_final_task_uncertainty', 0.0):.4f}  "
                    f"aoi={eval_summary.get('mean_final_task_aoi', 0.0):.2f}  "
                    f"repeat={eval_summary.get('mean_mean_repeat_sensing_ratio', 0.0):.3f}"
                )
            else:
                print(
                    f"  [Eval-{method_name}] {best_eval_metric}={eval_score:.4f}  "
                    f"covered_users={eval_summary.get('mean_final_covered_users', 0.0):.2f}  "
                    f"move={eval_summary.get('mean_total_move_distance', 0.0):.2f}  "
                    f"success={eval_summary.get('full_coverage_success_rate', 0.0):.3f}"
                )

            if eval_score >= best_eval_score:
                best_eval_score = eval_score
                best_eval_summary = dict(eval_summary)
                best_model_path = os.path.join(dirs["ckpt_dir"], "best_model.pt")
                agent.save(best_model_path)
                ensure_method_ckpt_alias(
                    output_root=train_cfg.output_root,
                    method_dir_name=method_identity["checkpoint_dir_name"],
                    src_path=best_model_path,
                    filename="best_model.pt",
                )

        if update_idx % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(dirs["ckpt_dir"], f"update_{update_idx:05d}.pt")
            agent.save(ckpt_path)

        if should_flush_training_log(update_idx=update_idx, train_cfg=train_cfg):
            training_log_payload = {
                "schema_version": SCHEMA_VERSION,
                "run_type": "train",
                "run_name": run_name,
                "method": method_identity,
                "paths": paths_block,
                "ablation_spec": ablation_spec.to_dict(),
                "scenario_cfg": asdict(scenario_cfg),
                "train_cfg": asdict(train_cfg),
                "ppo_cfg": asdict(ppo_cfg),
                "update_logs": update_logs,
                "eval_history": eval_history,
                "train_episode_history": train_episode_history,
                "best_eval_metric": best_eval_metric,
                "best_eval_score": best_eval_score,
                "best_eval_coverage": float(best_eval_summary.get("mean_final_coverage_ratio", 0.0)),
                "training_backend": {
                    "trainer_family": ablation_spec.trainer_family,
                    "policy_family": ablation_spec.policy_family,
                    "backend_impl": "shared_ppo_loop",
                    "placeholder_backend": False,
                },
            }
            save_json(training_log_payload, os.path.join(dirs["log_dir"], TRAIN_LOG_FILENAME))

    final_model_path = os.path.join(dirs["ckpt_dir"], "final_model.pt")
    agent.save(final_model_path)
    ensure_method_ckpt_alias(
        output_root=train_cfg.output_root,
        method_dir_name=method_identity["checkpoint_dir_name"],
        src_path=final_model_path,
        filename="final_model.pt",
    )

    plot_series = build_training_plot_series(
        {
            "update_logs": update_logs,
            "train_episode_history": train_episode_history,
        }
    )
    plot_training_history(
        episode_returns=plot_series["episode_returns"],
        final_coverages=plot_series["final_coverages"],
        total_move_distances=plot_series["total_move_distances"],
        mean_overlap_users_step=plot_series["mean_overlap_users_step"],
        title=f"{method_identity['display_name']} Training\n{run_name}",
        save_path=os.path.join(dirs["plot_dir"], TRAIN_PLOT_FILENAME),
        show=False,
    )

    tail = train_episode_history[-10:] if len(train_episode_history) >= 10 else train_episode_history
    last_rollout_summary = update_logs[-1]["rollout_summary"] if update_logs else {}
    train_tail_aggregate = MetricTracker()
    for item in tail:
        train_tail_aggregate.episode_summaries.append(item)
    tail_summary = train_tail_aggregate.aggregate()

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "summary_type": "train_run_summary",
        "run_name": run_name,
        "method": method_identity,
        "paths": {
            **paths_block,
            "best_checkpoint_path": os.path.join(dirs["ckpt_dir"], "best_model.pt"),
            "final_checkpoint_path": final_model_path,
        },
        "ablation_spec": ablation_spec.to_dict(),
        "train_config": asdict(train_cfg),
        "scenario_config": asdict(scenario_cfg),
        "ppo_config": asdict(ppo_cfg),
        "training_backend": {
            "trainer_family": ablation_spec.trainer_family,
            "policy_family": ablation_spec.policy_family,
            "backend_impl": "shared_ppo_loop",
            "placeholder_backend": False,
        },
        "num_total_updates": train_cfg.total_updates,
        "num_total_episodes": global_episode_idx,
        "best_eval_metric": best_eval_metric,
        "best_eval_score": best_eval_score,
        "best_eval_coverage": float(best_eval_summary.get("mean_final_coverage_ratio", 0.0)),
        "best_eval_summary": best_eval_summary,
        "last_rollout_summary": last_rollout_summary,
        "tail_train_summary": tail_summary,
        "reward_component_episode_total_means_last10": build_reward_tail_mean(train_episode_history, tail_size=10),
        "paper_metric_row": build_paper_metric_row(method_identity, best_eval_summary or last_rollout_summary),
    }
    save_json(summary_payload, os.path.join(dirs["run_dir"], SUMMARY_FILENAME))

    sync_train_alias_artifacts(
        output_root=train_cfg.output_root,
        method_dir_name=method_identity["output_dir_name"],
        dirs=dirs,
    )

    print("\n===== Training Finished =====")
    print("run_name:", run_name)
    print("run_dir:", dirs["run_dir"])
    print("method_name:", method_identity["method_name"])
    print("display_name:", method_identity["display_name"])
    print("trainer_family:", method_identity.get("trainer_family"))
    print("policy_family:", method_identity.get("policy_family"))
    print("agent_type:", method_identity.get("agent_type"))
    print("best_eval_metric:", best_eval_metric)
    print("best_eval_score:", best_eval_score)
    print("final_model_path:", final_model_path)
    print(f"Total updates: {train_cfg.total_updates}")
    print(f"Total episodes: {global_episode_idx}")


def run_registered_ippo_training(
    method_name: str,
    train_cfg: TrainConfig,
    ablation_spec: AblationSpec,
) -> None:
    scenario_cfg, run_name, dirs, method_identity, paths_block = _build_common_run_context(
        method_name=method_name,
        train_cfg=train_cfg,
        ablation_spec=ablation_spec,
    )

    local_obs_dim = int(scenario_cfg.get_local_obs_dim(method_name))
    ippo_cfg_kwargs = build_method_ppo_config_kwargs(
        method_name=method_name,
        local_obs_dim=local_obs_dim,
        action_dim=get_training_action_dim(scenario_cfg),
        device=train_cfg.device,
        max_obs_users=int(scenario_cfg.max_obs_users),
        max_obs_uavs=int(scenario_cfg.max_obs_uavs),
        num_direction_sectors=int(scenario_cfg.num_direction_sectors),
        num_radial_bins=int(scenario_cfg.num_radial_bins),
    )
    ippo_cfg = IPPOConfig(
        num_agents=int(scenario_cfg.max_candidate_uavs),
        **ippo_cfg_kwargs,
    )

    save_manifest(
        run_dir=dirs["run_dir"],
        run_type="train",
        run_name=run_name,
        note=resolve_training_backend_note(method_name, ablation_spec),
        schema_version=SCHEMA_VERSION,
        method=method_identity,
        paths=paths_block,
        ablation_spec=ablation_spec.to_dict(),
        scenario_cfg=scenario_cfg,
        train_cfg=train_cfg,
        ippo_cfg=ippo_cfg,
        training_backend={
            "trainer_family": ablation_spec.trainer_family,
            "policy_family": ablation_spec.policy_family,
            "backend_impl": "independent_ppo_loop",
            "placeholder_backend": False,
        },
    )

    agent = IndependentPPOAgent(ippo_cfg)

    update_logs: List[Dict[str, Any]] = []
    train_episode_history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []

    global_episode_idx = 0
    best_eval_coverage = -1.0
    best_eval_summary: Dict[str, Any] = {}

    for update_idx in range(1, train_cfg.total_updates + 1):
        buffer = IPPOBuffer(num_agents=scenario_cfg.max_candidate_uavs)
        rollout_tracker = MetricTracker()

        for _ in range(train_cfg.rollout_episodes_per_update):
            env = build_training_env(scenario_cfg)
            collected = run_one_episode_collect_ippo(
                env=env,
                agent=agent,
                buffer=buffer,
                seed=train_cfg.seed + global_episode_idx,
            )

            ep_summary = rollout_tracker.add_episode(collected["episode_metrics"])
            ep_summary["global_episode_idx"] = global_episode_idx
            ep_summary["update_idx"] = update_idx
            ep_summary["method_name"] = method_identity["method_name"]
            ep_summary["display_name"] = method_identity["display_name"]
            ep_summary["config_name"] = method_identity["config_name"]
            ep_summary["trainer_family"] = method_identity.get("trainer_family")
            ep_summary["policy_family"] = method_identity.get("policy_family")
            ep_summary["agent_type"] = method_identity.get("agent_type")
            train_episode_history.append(ep_summary)

            global_episode_idx += 1

        last_values = np.zeros((scenario_cfg.max_candidate_uavs,), dtype=np.float32)
        raw_train_stats = agent.update(
            buffer=buffer,
            last_values=last_values,
        )
        train_stats = make_ippo_train_stats_block(raw_train_stats)
        rollout_summary = rollout_tracker.aggregate()

        log_record = {
            "update_idx": update_idx,
            "global_episode_idx": global_episode_idx,
            "method_name": method_identity["method_name"],
            "display_name": method_identity["display_name"],
            "config_name": method_identity["config_name"],
            "trainer_family": method_identity.get("trainer_family"),
            "policy_family": method_identity.get("policy_family"),
            "agent_type": method_identity.get("agent_type"),
            "rollout_summary": rollout_summary,
            "train_stats": train_stats,
            "backend_impl": "independent_ppo_loop",
            "placeholder_backend": False,
            **{k: v for k, v in train_stats.items() if k != "independent_agent_stats"},
        }
        update_logs.append(log_record)

        print(
            f"[ippo][Update {update_idx:04d}] "
            f"coverage={rollout_summary.get('mean_final_coverage_ratio', 0.0):.4f}  "
            f"covered_users={rollout_summary.get('mean_final_covered_users', 0.0):.2f}  "
            f"move={rollout_summary.get('mean_total_move_distance', 0.0):.2f}  "
            f"overlap={rollout_summary.get('mean_mean_overlap_users_step', 0.0):.3f}  "
            f"return={rollout_summary.get('mean_episode_return', 0.0):.3f}"
        )

        if update_idx % train_cfg.eval_interval == 0 or update_idx == train_cfg.total_updates:
            eval_summary = evaluate_policy(
                env_cfg=scenario_cfg,
                agent=agent,
                eval_episodes=train_cfg.eval_episodes,
                base_seed=10000 + update_idx * 100,
            )
            eval_summary["update_idx"] = update_idx
            eval_summary["method_name"] = method_identity["method_name"]
            eval_summary["display_name"] = method_identity["display_name"]
            eval_summary["config_name"] = method_identity["config_name"]
            eval_summary["trainer_family"] = method_identity.get("trainer_family")
            eval_summary["policy_family"] = method_identity.get("policy_family")
            eval_summary["agent_type"] = method_identity.get("agent_type")
            eval_history.append(eval_summary)

            eval_cov = float(eval_summary.get("mean_final_coverage_ratio", 0.0))
            print(
                f"  [Eval-ippo] coverage={eval_cov:.4f}  "
                f"covered_users={eval_summary.get('mean_final_covered_users', 0.0):.2f}  "
                f"move={eval_summary.get('mean_total_move_distance', 0.0):.2f}  "
                f"success={eval_summary.get('full_coverage_success_rate', 0.0):.3f}"
            )

            if eval_cov >= best_eval_coverage:
                best_eval_coverage = eval_cov
                best_eval_summary = dict(eval_summary)
                best_model_path = os.path.join(dirs["ckpt_dir"], "best_model.pt")
                agent.save(best_model_path)
                ensure_method_ckpt_alias(
                    output_root=train_cfg.output_root,
                    method_dir_name=method_identity["checkpoint_dir_name"],
                    src_path=best_model_path,
                    filename="best_model.pt",
                )

        if update_idx % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(dirs["ckpt_dir"], f"update_{update_idx:05d}.pt")
            agent.save(ckpt_path)

        if should_flush_training_log(update_idx=update_idx, train_cfg=train_cfg):
            training_log_payload = {
                "schema_version": SCHEMA_VERSION,
                "run_type": "train",
                "run_name": run_name,
                "method": method_identity,
                "paths": paths_block,
                "ablation_spec": ablation_spec.to_dict(),
                "scenario_cfg": asdict(scenario_cfg),
                "train_cfg": asdict(train_cfg),
                "ippo_cfg": asdict(ippo_cfg),
                "update_logs": update_logs,
                "eval_history": eval_history,
                "train_episode_history": train_episode_history,
                "best_eval_coverage": best_eval_coverage,
                "training_backend": {
                    "trainer_family": ablation_spec.trainer_family,
                    "policy_family": ablation_spec.policy_family,
                    "backend_impl": "independent_ppo_loop",
                    "placeholder_backend": False,
                },
            }
            save_json(training_log_payload, os.path.join(dirs["log_dir"], TRAIN_LOG_FILENAME))

    final_model_path = os.path.join(dirs["ckpt_dir"], "final_model.pt")
    agent.save(final_model_path)
    ensure_method_ckpt_alias(
        output_root=train_cfg.output_root,
        method_dir_name=method_identity["checkpoint_dir_name"],
        src_path=final_model_path,
        filename="final_model.pt",
    )

    plot_series = build_training_plot_series(
        {
            "update_logs": update_logs,
            "train_episode_history": train_episode_history,
        }
    )
    plot_training_history(
        episode_returns=plot_series["episode_returns"],
        final_coverages=plot_series["final_coverages"],
        total_move_distances=plot_series["total_move_distances"],
        mean_overlap_users_step=plot_series["mean_overlap_users_step"],
        title=f"{method_identity['display_name']} Training\n{run_name}",
        save_path=os.path.join(dirs["plot_dir"], TRAIN_PLOT_FILENAME),
        show=False,
    )

    tail = train_episode_history[-10:] if len(train_episode_history) >= 10 else train_episode_history
    last_rollout_summary = update_logs[-1]["rollout_summary"] if update_logs else {}
    train_tail_aggregate = MetricTracker()
    for item in tail:
        train_tail_aggregate.episode_summaries.append(item)
    tail_summary = train_tail_aggregate.aggregate()

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "summary_type": "train_run_summary",
        "run_name": run_name,
        "method": method_identity,
        "paths": {
            **paths_block,
            "best_checkpoint_path": os.path.join(dirs["ckpt_dir"], "best_model.pt"),
            "final_checkpoint_path": final_model_path,
        },
        "ablation_spec": ablation_spec.to_dict(),
        "train_config": asdict(train_cfg),
        "scenario_config": asdict(scenario_cfg),
        "ippo_config": asdict(ippo_cfg),
        "training_backend": {
            "trainer_family": ablation_spec.trainer_family,
            "policy_family": ablation_spec.policy_family,
            "backend_impl": "independent_ppo_loop",
            "placeholder_backend": False,
        },
        "num_total_updates": train_cfg.total_updates,
        "num_total_episodes": global_episode_idx,
        "best_eval_coverage": best_eval_coverage,
        "best_eval_summary": best_eval_summary,
        "last_rollout_summary": last_rollout_summary,
        "tail_train_summary": tail_summary,
        "reward_component_episode_total_means_last10": build_reward_tail_mean(train_episode_history, tail_size=10),
        "paper_metric_row": build_paper_metric_row(method_identity, best_eval_summary or last_rollout_summary),
    }
    save_json(summary_payload, os.path.join(dirs["run_dir"], SUMMARY_FILENAME))

    sync_train_alias_artifacts(
        output_root=train_cfg.output_root,
        method_dir_name=method_identity["output_dir_name"],
        dirs=dirs,
    )

    print("\n===== IPPO Training Finished =====")
    print("run_name:", run_name)
    print("run_dir:", dirs["run_dir"])
    print("method_name:", method_identity["method_name"])
    print("display_name:", method_identity["display_name"])
    print("trainer_family:", method_identity.get("trainer_family"))
    print("policy_family:", method_identity.get("policy_family"))
    print("agent_type:", method_identity.get("agent_type"))
    print("best_eval_coverage:", best_eval_coverage)
    print("final_model_path:", final_model_path)
    print(f"Total updates: {train_cfg.total_updates}")
    print(f"Total episodes: {global_episode_idx}")


def run_registered_maddpg_training(
    method_name: str,
    train_cfg: TrainConfig,
    ablation_spec: AblationSpec,
) -> None:
    scenario_cfg, run_name, dirs, method_identity, paths_block = _build_common_run_context(
        method_name=method_name,
        train_cfg=train_cfg,
        ablation_spec=ablation_spec,
    )

    local_obs_dim = int(scenario_cfg.get_local_obs_dim(method_name))
    maddpg_cfg = MADDPGConfig(
        num_agents=int(scenario_cfg.max_candidate_uavs),
        local_obs_dim=local_obs_dim,
        action_dim=get_training_action_dim(scenario_cfg),
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        num_hidden_layers=2,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.01,
        replay_size=200000,
        batch_size=256,
        update_after=1000,
        update_every=50,
        gradient_steps=50,
        policy_update_freq=2,
        gumbel_tau=1.0,
        explore_epsilon=0.10,
        max_grad_norm=10.0,
        device=train_cfg.device,
    )

    save_manifest(
        run_dir=dirs["run_dir"],
        run_type="train",
        run_name=run_name,
        note=resolve_training_backend_note(method_name, ablation_spec),
        schema_version=SCHEMA_VERSION,
        method=method_identity,
        paths=paths_block,
        ablation_spec=ablation_spec.to_dict(),
        scenario_cfg=scenario_cfg,
        train_cfg=train_cfg,
        maddpg_cfg=asdict(maddpg_cfg),
        training_backend={
            "trainer_family": ablation_spec.trainer_family,
            "policy_family": ablation_spec.policy_family,
            "backend_impl": "discrete_maddpg_loop",
            "placeholder_backend": False,
        },
    )

    agent = MADDPGAgent(maddpg_cfg)
    replay_buffer = MADDPGReplayBuffer(
        capacity=int(maddpg_cfg.replay_size),
        num_agents=int(maddpg_cfg.num_agents),
        obs_dim=int(maddpg_cfg.local_obs_dim),
        action_dim=int(maddpg_cfg.action_dim),
    )

    update_logs: List[Dict[str, Any]] = []
    train_episode_history: List[Dict[str, Any]] = []
    eval_history: List[Dict[str, Any]] = []

    global_episode_idx = 0
    best_eval_coverage = -1.0
    best_eval_summary: Dict[str, Any] = {}

    for update_idx in range(1, train_cfg.total_updates + 1):
        rollout_tracker = MetricTracker()
        online_update_stats_list: List[Dict[str, Any]] = []

        for _ in range(train_cfg.rollout_episodes_per_update):
            env = build_training_env(scenario_cfg)
            collected = run_one_episode_collect_maddpg(
                env=env,
                agent=agent,
                replay_buffer=replay_buffer,
                seed=train_cfg.seed + global_episode_idx,
            )

            ep_summary = rollout_tracker.add_episode(collected["episode_metrics"])
            ep_summary["global_episode_idx"] = global_episode_idx
            ep_summary["update_idx"] = update_idx
            ep_summary["method_name"] = method_identity["method_name"]
            ep_summary["display_name"] = method_identity["display_name"]
            ep_summary["config_name"] = method_identity["config_name"]
            ep_summary["trainer_family"] = method_identity.get("trainer_family")
            ep_summary["policy_family"] = method_identity.get("policy_family")
            ep_summary["agent_type"] = method_identity.get("agent_type")
            train_episode_history.append(ep_summary)

            online_update_stats_list.extend(collected.get("episode_update_stats", []))
            global_episode_idx += 1

        if online_update_stats_list:
            maddpg_stats_raw = {
                "actor_loss": float(np.mean([x.get("actor_loss", 0.0) for x in online_update_stats_list])),
                "critic_loss": float(np.mean([x.get("critic_loss", 0.0) for x in online_update_stats_list])),
                "q_mean": float(np.mean([x.get("q_mean", 0.0) for x in online_update_stats_list])),
                "target_q_mean": float(np.mean([x.get("target_q_mean", 0.0) for x in online_update_stats_list])),
                "actor_q_mean": float(np.mean([x.get("actor_q_mean", 0.0) for x in online_update_stats_list])),
                "buffer_size": float(len(replay_buffer)),
                "num_agents": int(maddpg_cfg.num_agents),
                "per_agent_stats": online_update_stats_list[-1].get("per_agent_stats", []) if online_update_stats_list else [],
                "skipped_update": False,
            }
        else:
            maddpg_stats_raw = {
                "actor_loss": 0.0,
                "critic_loss": 0.0,
                "q_mean": 0.0,
                "target_q_mean": 0.0,
                "actor_q_mean": 0.0,
                "buffer_size": float(len(replay_buffer)),
                "num_agents": int(maddpg_cfg.num_agents),
                "per_agent_stats": [],
                "skipped_update": True,
            }

        train_stats = make_maddpg_train_stats_block(maddpg_stats_raw)
        rollout_summary = rollout_tracker.aggregate()

        log_record = {
            "update_idx": update_idx,
            "global_episode_idx": global_episode_idx,
            "method_name": method_identity["method_name"],
            "display_name": method_identity["display_name"],
            "config_name": method_identity["config_name"],
            "trainer_family": method_identity.get("trainer_family"),
            "policy_family": method_identity.get("policy_family"),
            "agent_type": method_identity.get("agent_type"),
            "rollout_summary": rollout_summary,
            "train_stats": train_stats,
            "backend_impl": "discrete_maddpg_loop",
            "placeholder_backend": False,
            **{k: v for k, v in train_stats.items() if k != "per_agent_stats"},
        }
        update_logs.append(log_record)

        print(
            f"[maddpg][Update {update_idx:04d}] "
            f"coverage={rollout_summary.get('mean_final_coverage_ratio', 0.0):.4f}  "
            f"covered_users={rollout_summary.get('mean_final_covered_users', 0.0):.2f}  "
            f"move={rollout_summary.get('mean_total_move_distance', 0.0):.2f}  "
            f"overlap={rollout_summary.get('mean_mean_overlap_users_step', 0.0):.3f}  "
            f"return={rollout_summary.get('mean_episode_return', 0.0):.3f}  "
            f"actor_loss={train_stats.get('train_actor_loss', 0.0):.4f}  "
            f"critic_loss={train_stats.get('train_critic_loss', 0.0):.4f}"
        )

        if update_idx % train_cfg.eval_interval == 0 or update_idx == train_cfg.total_updates:
            eval_summary = evaluate_policy(
                env_cfg=scenario_cfg,
                agent=agent,
                eval_episodes=train_cfg.eval_episodes,
                base_seed=10000 + update_idx * 100,
            )
            eval_summary["update_idx"] = update_idx
            eval_summary["method_name"] = method_identity["method_name"]
            eval_summary["display_name"] = method_identity["display_name"]
            eval_summary["config_name"] = method_identity["config_name"]
            eval_summary["trainer_family"] = method_identity.get("trainer_family")
            eval_summary["policy_family"] = method_identity.get("policy_family")
            eval_summary["agent_type"] = method_identity.get("agent_type")
            eval_history.append(eval_summary)

            eval_cov = float(eval_summary.get("mean_final_coverage_ratio", 0.0))
            print(
                f"  [Eval-maddpg] coverage={eval_cov:.4f}  "
                f"covered_users={eval_summary.get('mean_final_covered_users', 0.0):.2f}  "
                f"move={eval_summary.get('mean_total_move_distance', 0.0):.2f}  "
                f"success={eval_summary.get('full_coverage_success_rate', 0.0):.3f}"
            )

            if eval_cov >= best_eval_coverage:
                best_eval_coverage = eval_cov
                best_eval_summary = dict(eval_summary)
                best_model_path = os.path.join(dirs["ckpt_dir"], "best_model.pt")
                agent.save(best_model_path)
                ensure_method_ckpt_alias(
                    output_root=train_cfg.output_root,
                    method_dir_name=method_identity["checkpoint_dir_name"],
                    src_path=best_model_path,
                    filename="best_model.pt",
                )

        if update_idx % train_cfg.save_interval == 0:
            ckpt_path = os.path.join(dirs["ckpt_dir"], f"update_{update_idx:05d}.pt")
            agent.save(ckpt_path)

        if should_flush_training_log(update_idx=update_idx, train_cfg=train_cfg):
            training_log_payload = {
                "schema_version": SCHEMA_VERSION,
                "run_type": "train",
                "run_name": run_name,
                "method": method_identity,
                "paths": paths_block,
                "ablation_spec": ablation_spec.to_dict(),
                "scenario_cfg": asdict(scenario_cfg),
                "train_cfg": asdict(train_cfg),
                "maddpg_cfg": asdict(maddpg_cfg),
                "replay_buffer_summary": replay_buffer.summary(),
                "update_logs": update_logs,
                "eval_history": eval_history,
                "train_episode_history": train_episode_history,
                "best_eval_coverage": best_eval_coverage,
                "training_backend": {
                    "trainer_family": ablation_spec.trainer_family,
                    "policy_family": ablation_spec.policy_family,
                    "backend_impl": "discrete_maddpg_loop",
                    "placeholder_backend": False,
                },
            }
            save_json(training_log_payload, os.path.join(dirs["log_dir"], TRAIN_LOG_FILENAME))

    final_model_path = os.path.join(dirs["ckpt_dir"], "final_model.pt")
    agent.save(final_model_path)
    ensure_method_ckpt_alias(
        output_root=train_cfg.output_root,
        method_dir_name=method_identity["checkpoint_dir_name"],
        src_path=final_model_path,
        filename="final_model.pt",
    )

    plot_series = build_training_plot_series(
        {
            "update_logs": update_logs,
            "train_episode_history": train_episode_history,
        }
    )
    plot_training_history(
        episode_returns=plot_series["episode_returns"],
        final_coverages=plot_series["final_coverages"],
        total_move_distances=plot_series["total_move_distances"],
        mean_overlap_users_step=plot_series["mean_overlap_users_step"],
        title=f"{method_identity['display_name']} Training\n{run_name}",
        save_path=os.path.join(dirs["plot_dir"], TRAIN_PLOT_FILENAME),
        show=False,
    )

    tail = train_episode_history[-10:] if len(train_episode_history) >= 10 else train_episode_history
    last_rollout_summary = update_logs[-1]["rollout_summary"] if update_logs else {}
    train_tail_aggregate = MetricTracker()
    for item in tail:
        train_tail_aggregate.episode_summaries.append(item)
    tail_summary = train_tail_aggregate.aggregate()

    summary_payload = {
        "schema_version": SCHEMA_VERSION,
        "summary_type": "train_run_summary",
        "run_name": run_name,
        "method": method_identity,
        "paths": {
            **paths_block,
            "best_checkpoint_path": os.path.join(dirs["ckpt_dir"], "best_model.pt"),
            "final_checkpoint_path": final_model_path,
        },
        "ablation_spec": ablation_spec.to_dict(),
        "train_config": asdict(train_cfg),
        "scenario_config": asdict(scenario_cfg),
        "maddpg_config": asdict(maddpg_cfg),
        "replay_buffer_summary": replay_buffer.summary(),
        "training_backend": {
            "trainer_family": ablation_spec.trainer_family,
            "policy_family": ablation_spec.policy_family,
            "backend_impl": "discrete_maddpg_loop",
            "placeholder_backend": False,
        },
        "num_total_updates": train_cfg.total_updates,
        "num_total_episodes": global_episode_idx,
        "best_eval_coverage": best_eval_coverage,
        "best_eval_summary": best_eval_summary,
        "last_rollout_summary": last_rollout_summary,
        "tail_train_summary": tail_summary,
        "reward_component_episode_total_means_last10": build_reward_tail_mean(train_episode_history, tail_size=10),
        "paper_metric_row": build_paper_metric_row(method_identity, best_eval_summary or last_rollout_summary),
    }
    save_json(summary_payload, os.path.join(dirs["run_dir"], SUMMARY_FILENAME))

    sync_train_alias_artifacts(
        output_root=train_cfg.output_root,
        method_dir_name=method_identity["output_dir_name"],
        dirs=dirs,
    )

    print("\n===== MADDPG Training Finished =====")
    print("run_name:", run_name)
    print("run_dir:", dirs["run_dir"])
    print("method_name:", method_identity["method_name"])
    print("display_name:", method_identity["display_name"])
    print("trainer_family:", method_identity.get("trainer_family"))
    print("policy_family:", method_identity.get("policy_family"))
    print("agent_type:", method_identity.get("agent_type"))
    print("best_eval_coverage:", best_eval_coverage)
    print("final_model_path:", final_model_path)
    print(f"Total updates: {train_cfg.total_updates}")
    print(f"Total episodes: {global_episode_idx}")


def run_ppo_family_training(method_name: str, args: argparse.Namespace) -> None:
    ablation_spec = get_ablation_spec(method_name)
    train_cfg = build_train_config(args, ablation_spec)
    run_registered_shared_training(method_name=method_name, train_cfg=train_cfg, ablation_spec=ablation_spec)


def run_ippo_training(args: argparse.Namespace) -> None:
    method_name = "ippo"
    ablation_spec = get_ablation_spec(method_name)
    train_cfg = build_train_config(args, ablation_spec)
    run_registered_ippo_training(method_name=method_name, train_cfg=train_cfg, ablation_spec=ablation_spec)


def run_maddpg_training(args: argparse.Namespace) -> None:
    method_name = "maddpg"
    ablation_spec = get_ablation_spec(method_name)
    train_cfg = build_train_config(args, ablation_spec)
    run_registered_maddpg_training(method_name=method_name, train_cfg=train_cfg, ablation_spec=ablation_spec)


def main() -> None:
    args = parse_args()
    method_name = resolve_method_name(
        method_name=args.method_name,
        config_name=args.config_name,
        default_method_name="ppo_main",
    )
    ablation_spec = get_ablation_spec(method_name)

    dispatch_map = {
        "ppo_shared": lambda: run_ppo_family_training(method_name, args),
        "ippo": lambda: run_ippo_training(args),
        "maddpg": lambda: run_maddpg_training(args),
    }

    trainer_family = ablation_spec.trainer_family
    if trainer_family not in dispatch_map:
        raise KeyError(f"Unsupported trainer_family: {trainer_family} for method_name={method_name}")

    dispatch_map[trainer_family]()


if __name__ == "__main__":
    main()
