from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from agents.ippo_agent import IPPOConfig, IndependentPPOAgent
from agents.maddpg_agent import MADDPGAgent, MADDPGConfig
from agents.ppo.ppo_agent import PPOConfig, SharedPPOAgent
from baselines.constrained_kmeans_baseline import ConstrainedKMeansBaseline, KMeansDeployResult
from baselines.method_registry import get_method_meta, is_baseline_method
from baselines.random_policy import RandomPolicy
from configs.ablation_config import resolve_method_checkpoint_path
from envs.disaster_deployment_env import DisasterDeploymentEnv
from envs.metrics import EpisodeMetrics


@dataclass
class DeploymentRunResult:
    method_name: str
    success: bool
    final_coverage_ratio: float
    final_covered_users: int
    total_distance: float
    episode_length: int
    active_uav_count: int
    uav_positions: np.ndarray
    active_mask: np.ndarray
    assigned_uav_idx: np.ndarray
    covered_mask: np.ndarray
    trajectory_history: Optional[List[np.ndarray]]
    raw: Dict[str, Any]


@dataclass
class ExecutionContext:
    method_name: str
    method_category: str
    method_type: str
    display_name: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "method_name": self.method_name,
            "method_category": self.method_category,
            "method_type": self.method_type,
            "display_name": self.display_name,
        }


def get_execution_context(method_name: str) -> ExecutionContext:
    meta = get_method_meta(method_name)
    return ExecutionContext(
        method_name=meta.method_name,
        method_category=meta.category,
        method_type=meta.method_type,
        display_name=meta.display_name,
    )


def build_agent_from_checkpoint(ckpt_path: str, device: str) -> Any:
    payload = torch.load(ckpt_path, map_location=device)
    agent_type = payload.get("agent_type", "shared_ppo")

    if agent_type == "ippo":
        cfg_dict = payload["config"]
        ippo_cfg = IPPOConfig(**cfg_dict)
        ippo_cfg.device = device
        agent = IndependentPPOAgent(ippo_cfg)
        agent.load(ckpt_path, strict=False)
        return agent

    if agent_type == "maddpg":
        cfg_dict = payload["config"]
        maddpg_cfg = MADDPGConfig(**cfg_dict)
        maddpg_cfg.device = device
        agent = MADDPGAgent(maddpg_cfg)
        agent.load(ckpt_path, strict=False)
        return agent

    cfg_dict = payload["config"]
    ppo_cfg = PPOConfig(**cfg_dict)
    ppo_cfg.device = device
    agent = SharedPPOAgent(ppo_cfg)
    agent.load(ckpt_path, strict=False)
    return agent


def load_registered_policy_agent(
    method_name: str,
    checkpoint_name: str = "best_model.pt",
    device: str = "cpu",
) -> Any:
    ckpt_path = resolve_method_checkpoint_path(
        method_name=method_name,
        checkpoint_name=checkpoint_name,
    )
    return build_agent_from_checkpoint(ckpt_path=ckpt_path, device=device)


def _pack_result(
    method_name: str,
    env: DisasterDeploymentEnv,
    episode_metrics: EpisodeMetrics,
    traj: Optional[List[np.ndarray]],
    last_info: Dict[str, Any],
    extra_raw: Optional[Dict[str, Any]] = None,
) -> DeploymentRunResult:
    raw = {
        "episode_summary": episode_metrics.summary(),
        "env_info": last_info,
        "num_uavs_used": int(np.sum(env.active_mask)),
    }
    if extra_raw:
        raw.update(extra_raw)

    return DeploymentRunResult(
        method_name=method_name,
        success=bool(np.all(env.covered_mask)),
        final_coverage_ratio=float(env.coverage_ratio),
        final_covered_users=int(env.covered_mask.sum()),
        total_distance=float(np.sum(env.total_distance_per_uav)),
        episode_length=int(last_info.get("step", 0)),
        active_uav_count=int(np.sum(env.active_mask)),
        uav_positions=env.uav_positions.copy(),
        active_mask=env.active_mask.copy(),
        assigned_uav_idx=env.assigned_uav_idx.copy(),
        covered_mask=env.covered_mask.copy(),
        trajectory_history=traj,
        raw=raw,
    )


def run_random_deployment(
    env: DisasterDeploymentEnv,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    method_name: str = "random_masked",
) -> DeploymentRunResult:
    exec_ctx = get_execution_context(method_name)
    obs = env.reset(seed=seed)
    policy = RandomPolicy(
        action_size=env.cfg.action_size,
        num_agents=env.num_agents,
        seed=env.cfg.seed if seed is None else seed,
    )
    traj = [env.uav_positions.copy()]
    episode_metrics = EpisodeMetrics()
    done = False
    steps = 0
    forced_limit = env.cfg.max_steps if max_steps is None else max_steps
    last_info = {"step": 0, "termination_reason": "unknown"}

    while not done and steps < forced_limit:
        actions = policy.act(action_mask=obs["action_mask"])
        obs, reward, done, info = env.step(actions)
        episode_metrics.update(reward, info)
        traj.append(env.uav_positions.copy())
        steps += 1
        last_info = info

    return _pack_result(
        method_name=method_name,
        env=env,
        episode_metrics=episode_metrics,
        traj=traj,
        last_info=last_info,
        extra_raw={
            **exec_ctx.to_dict(),
            "control_mode": "dynamic_multi_step",
        },
    )


def _greedy_local_single_action(env: DisasterDeploymentEnv, agent_id: int) -> int:
    if not bool(env.active_mask[agent_id]):
        return 0

    uncovered_mask = ~env.covered_mask
    if not np.any(uncovered_mask):
        return 0

    current_xy = env.uav_positions[agent_id, :2].astype(np.float32)
    candidate_positions = env.ue_positions[uncovered_mask, :2].astype(np.float32)

    if candidate_positions.shape[0] == 0:
        return 0

    deltas = candidate_positions - current_xy[None, :]
    dists = np.linalg.norm(deltas, axis=1)

    visible_mask = dists <= float(env.cfg.obs_radius)
    if np.any(visible_mask):
        candidate_deltas = deltas[visible_mask]
        candidate_dists = dists[visible_mask]
    else:
        candidate_deltas = deltas
        candidate_dists = dists

    target_delta = candidate_deltas[int(np.argmin(candidate_dists))]
    dx, dy = float(target_delta[0]), float(target_delta[1])

    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0

    if abs(dx) >= abs(dy):
        return 4 if dx > 0 else 3
    return 1 if dy > 0 else 2


def run_greedy_local_deployment(
    env: DisasterDeploymentEnv,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    method_name: str = "greedy_local",
) -> DeploymentRunResult:
    exec_ctx = get_execution_context(method_name)
    env.reset(seed=seed)
    traj = [env.uav_positions.copy()]
    episode_metrics = EpisodeMetrics()
    done = False
    steps = 0
    forced_limit = env.cfg.max_steps if max_steps is None else max_steps
    last_info: Dict[str, Any] = {"step": 0, "termination_reason": "unknown"}

    while not done and steps < forced_limit:
        actions = np.zeros((env.num_agents,), dtype=np.int64)
        for agent_id in range(env.num_agents):
            actions[agent_id] = _greedy_local_single_action(env, agent_id)

        _, reward, done, info = env.step(actions)
        episode_metrics.update(reward, info)
        traj.append(env.uav_positions.copy())
        steps += 1
        last_info = info

    return _pack_result(
        method_name=method_name,
        env=env,
        episode_metrics=episode_metrics,
        traj=traj,
        last_info=last_info,
        extra_raw={
            **exec_ctx.to_dict(),
            "control_mode": "dynamic_multi_step",
        },
    )


def run_kmeans_deployment(
    env: DisasterDeploymentEnv,
    m_min: int = 1,
    m_max: Optional[int] = None,
    n_init: int = 5,
    max_iter: int = 50,
    tol: float = 1e-3,
    seed: Optional[int] = None,
    method_name: str = "constrained_kmeans",
) -> DeploymentRunResult:
    exec_ctx = get_execution_context(method_name)
    env.reset(seed=seed)
    init_positions = env.uav_positions.copy()

    solver = ConstrainedKMeansBaseline(env.cfg)
    result: KMeansDeployResult = solver.solve(
        ue_positions=env.ue_positions,
        m_min=m_min,
        m_max=m_max if m_max is not None else env.cfg.max_candidate_uavs,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_seed=seed,
    )

    final_positions = init_positions.copy()
    final_positions[: result.num_uavs_used] = result.uav_positions

    move_distances = np.zeros((env.num_agents,), dtype=np.float32)
    if result.num_uavs_used > 0:
        move_distances[: result.num_uavs_used] = np.linalg.norm(
            final_positions[: result.num_uavs_used, :2] - init_positions[: result.num_uavs_used, :2],
            axis=1,
        ).astype(np.float32)

    env.active_mask[:] = False
    env.active_mask[: result.num_uavs_used] = True
    env.uav_positions[:] = final_positions
    env.total_distance_per_uav[:] = move_distances
    env._update_coverage()

    traj = [init_positions.copy(), env.uav_positions.copy()]

    episode_metrics = EpisodeMetrics()
    last_info = {
        "step": 1,
        "active_uav_count": int(np.sum(env.active_mask)),
        "coverage_ratio": float(env.coverage_ratio),
        "covered_users": int(env.covered_mask.sum()),
        "move_distance_total_step": float(np.sum(move_distances)),
        "overlap_users": int(np.sum(env.cover_count_per_user > 1)),
        "termination_reason": "static_deploy",
        "total_distance_per_uav": env.total_distance_per_uav.copy(),
    }
    episode_metrics.update(0.0, last_info)

    return _pack_result(
        method_name=method_name,
        env=env,
        episode_metrics=episode_metrics,
        traj=traj,
        last_info=last_info,
        extra_raw={
            **exec_ctx.to_dict(),
            "solver_iterations": int(result.iterations),
            "solver_success": bool(result.success),
            "solver_num_uavs_used": int(result.num_uavs_used),
            "control_mode": "static_one_shot",
        },
    )


def run_ppo_deployment(
    env: DisasterDeploymentEnv,
    agent: Any,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    greedy: bool = False,
    method_name: Optional[str] = None,
) -> DeploymentRunResult:
    method_name = method_name or "ppo_main"
    exec_ctx = get_execution_context(method_name)

    obs_dict = env.reset(seed=seed)
    local_obs = obs_dict["local_obs"]
    action_mask = obs_dict["action_mask"]

    episode_metrics = EpisodeMetrics()
    traj = [env.uav_positions.copy()]

    done = False
    steps = 0
    forced_limit = env.cfg.max_steps if max_steps is None else max_steps
    num_agents = local_obs.shape[0]

    last_info: Dict[str, Any] = {
        "coverage_ratio": float(env.coverage_ratio),
        "covered_users": int(env.covered_mask.sum()),
        "total_distance_per_uav": env.total_distance_per_uav.copy(),
        "remaining_time": env.remaining_time.copy(),
        "step": 0,
        "active_uav_count": int(np.sum(env.active_mask)) if hasattr(env, "active_mask") else int(num_agents),
        "termination_reason": "running",
    }

    while not done and steps < forced_limit:
        if greedy and hasattr(agent, "greedy_act"):
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
        episode_metrics.update(reward, info)

        local_obs = next_obs_dict["local_obs"]
        action_mask = next_obs_dict["action_mask"]

        traj.append(env.uav_positions.copy())
        steps += 1
        last_info = info

    return _pack_result(
        method_name=method_name,
        env=env,
        episode_metrics=episode_metrics,
        traj=traj,
        last_info=last_info,
        extra_raw={
            **exec_ctx.to_dict(),
            "greedy": greedy,
            "control_mode": "dynamic_multi_step",
            "policy_agent_type": type(agent).__name__,
        },
    )


def execute_method_once(
    method_name: str,
    env: DisasterDeploymentEnv,
    max_steps: Optional[int] = None,
    seed: Optional[int] = None,
    device: str = "cpu",
    greedy: bool = True,
    checkpoint_name: str = "best_model.pt",
    agent_cache: Optional[Dict[str, Any]] = None,
) -> DeploymentRunResult:
    if is_baseline_method(method_name):
        if method_name == "random_masked":
            return run_random_deployment(env=env, max_steps=max_steps, seed=seed, method_name=method_name)
        if method_name == "greedy_local":
            return run_greedy_local_deployment(env=env, max_steps=max_steps, seed=seed, method_name=method_name)
        if method_name == "constrained_kmeans":
            return run_kmeans_deployment(env=env, max_steps=max_steps, seed=seed, method_name=method_name)
        raise KeyError(f"Unsupported baseline method: {method_name}")

    if agent_cache is None:
        agent_cache = {}

    if method_name not in agent_cache:
        agent_cache[method_name] = load_registered_policy_agent(
            method_name=method_name,
            checkpoint_name=checkpoint_name,
            device=device,
        )

    agent = agent_cache[method_name]
    return run_ppo_deployment(
        env=env,
        agent=agent,
        max_steps=max_steps,
        seed=seed,
        greedy=greedy,
        method_name=method_name,
    )