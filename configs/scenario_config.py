from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class ScenarioConfig:
    # =========================
    # Geometry
    # =========================
    r_safe: float = 500.0
    r_disaster: float = 2000.0
    bs_pos: Tuple[float, float, float] = (0.0, 0.0, 30.0)

    # =========================
    # Users
    # =========================
    num_users: int = 20
    user_distribution_mode: str = "mixed"   # uniform / clustered / mixed
    num_user_clusters: int = 3
    clustered_user_ratio: float = 0.80
    cluster_radius: float = 220.0
    edge_avoidance_ratio: float = 0.15
    edge_soft_limit_ratio: float = 0.20
    cluster_center_min_radius_ratio: float = 0.18
    cluster_center_max_radius_ratio: float = 0.72
    user_radial_beta_a: float = 2.2
    user_radial_beta_b: float = 3.0

    # =========================
    # Local neighborhood summary for actor observation
    # =========================
    use_global_uncovered_summary: bool = False
    num_direction_sectors: int = 4
    num_radial_bins: int = 3

    # =========================
    # UAV candidates / deployment
    # =========================
    max_candidate_uavs: int = 5
    uav_init_height: float = 100.0
    uav_h_min: float = 20.0
    uav_h_max: float = 300.0
    uav_speed: float = 10.0
    uav_max_time: float = 1200.0
    uav_init_mode: str = "circle"
    custom_uav_init_xy: Tuple[Tuple[float, float], ...] = field(default_factory=tuple)

    # 兼容旧脚本：如果旧代码仍然传 num_uavs，则映射到 max_candidate_uavs
    num_uavs: int = 5

    # =========================
    # Activation design
    # =========================
    allow_activation_action: bool = False
    initially_active_uavs: int = 5
    activation_cost_once: float = 0.0
    active_alive_cost: float = 0.0
    inactive_idle_cost: float = 0.0

    # =========================
    # Time / Episode
    # =========================
    dt: float = 5.0
    max_steps: int = 40
    stagnation_patience: int = 15

    # =========================
    # Action space
    # =========================
    # fixed semantics:
    # - 5 actions when activation is disabled: stay/up/down/left/right
    # - 6 actions when activation is enabled: + activate
    action_size: int = 5

    # =========================
    # QoS / Coverage
    # =========================
    qos_threshold_db: float = 110.0
    use_simplified_qos: bool = True
    simplified_coverage_radius: float = 350.0

    # =========================
    # Observation
    # =========================
    obs_radius: float = 800.0
    max_obs_users: int = 20
    max_obs_uavs: int = 2

    use_neighbor_uav_obs: bool = True
    use_local_neighborhood_summary: bool = True
    use_uncovered_guidance: bool = True

    # 新增：增强观测相关开关
    use_enhanced_obs: bool = False
    enhanced_obs_for_mcg_only: bool = True
    use_user_summary_features: bool = False
    use_neighbor_summary_features: bool = False
    use_overlap_risk_features: bool = False

    # Trusted sensing task. Existing coverage experiments keep this disabled.
    use_trusted_sensing: bool = False
    sensing_radius: float = 350.0
    task_initial_uncertainty: float = 1.0
    task_min_uncertainty: float = 0.05
    task_uncertainty_reduction: float = 0.45
    task_initial_aoi: float = 10.0
    task_max_aoi: float = 40.0
    task_priority_min: float = 0.5
    task_priority_max: float = 1.5
    trusted_sensing_uncertainty_target: float = 0.20

    # Independent resource-cognition environment.
    use_resource_cognition: bool = False
    num_cognition_tasks: int = 20
    cognition_num_bands: int = 1
    cognition_max_task_slots: int = 8
    cognition_observation_noise_std: float = 0.05
    cognition_aoi_increment: float = 1.0
    cognition_task_uncertainty_reduction: float = 0.45
    cognition_sensing_cost: float = 0.05
    cognition_repeat_penalty: float = 1.0

    # =========================
    # Reward（ppo_main 基线原始 reward）
    # =========================
    w_newly_served_users: float = 6.0
    w_delta_coverage_ratio: float = 0.0
    w_step_move_cost: float = 0.20
    w_overlap_penalty: float = 3.0

    w_distance_shaping: float = 0.0
    w_dispersion_penalty: float = 0.0
    min_uav_separation: float = 200.0
    shaping_target_mode: str = "nearest_uncovered"

    w_out_of_bound_penalty: float = 1.0
    w_timeout_penalty: float = 3.0
    w_terminal_success: float = 30.0
    w_final_coverage_bonus: float = 0.0

    w_active_count_penalty: float = 0.0
    w_new_activation_penalty: float = 0.0

    # =========================
    # Reward（mcg_ppo 专用增强塑形）
    # =========================
    method_name: str = "ppo_main"
    use_mcg_reward: bool = False

    reward_weight_coverage: float = 2.0
    reward_weight_marginal_contribution: float = 1.2
    reward_weight_movement_cost: float = 0.35
    reward_weight_overlap_penalty: float = 1.0
    reward_weight_uncovered_guidance: float = 0.6

    reward_weight_uncertainty_gain: float = 8.0
    reward_weight_aoi_gain: float = 4.0
    reward_weight_repeat_sensing_penalty: float = 2.0
    reward_weight_sensing_cost: float = 0.05

    enable_marginal_contribution_reward: bool = True
    enable_overlap_penalty: bool = True

    overlap_distance_threshold: float = 320.0
    guidance_distance_scale: float = 400.0
    mc_reward_clip: float = 20.0
    reward_normalize_for_mcg: bool = True

    # =========================
    # Method / trainer reservation
    # =========================
    rl_method_family: str = "ppo_shared"
    policy_architecture: str = "shared_actor_critic"

    # =========================
    # Random seed
    # =========================
    seed: int = 42

    def __post_init__(self) -> None:
        if self.max_candidate_uavs <= 0 and self.num_uavs > 0:
            self.max_candidate_uavs = self.num_uavs
        if self.num_uavs != self.max_candidate_uavs:
            self.num_uavs = self.max_candidate_uavs

        self.initially_active_uavs = min(self.initially_active_uavs, self.max_candidate_uavs)
        self.max_obs_users = min(self.max_obs_users, self.num_users)
        self.max_obs_uavs = min(self.max_obs_uavs, max(self.max_candidate_uavs - 1, 0))

        self.user_distribution_mode = str(self.user_distribution_mode).lower()
        self.method_name = str(self.method_name).strip() if self.method_name is not None else "ppo_main"
        self.rl_method_family = str(self.rl_method_family).strip() if self.rl_method_family is not None else "ppo_shared"
        self.policy_architecture = (
            str(self.policy_architecture).strip()
            if self.policy_architecture is not None
            else "shared_actor_critic"
        )

        self.num_user_clusters = int(max(1, self.num_user_clusters))
        self.clustered_user_ratio = float(min(max(self.clustered_user_ratio, 0.0), 1.0))
        self.edge_avoidance_ratio = float(min(max(self.edge_avoidance_ratio, 0.0), 1.0))
        self.edge_soft_limit_ratio = float(min(max(self.edge_soft_limit_ratio, 0.01), 0.49))
        self.cluster_center_min_radius_ratio = float(min(max(self.cluster_center_min_radius_ratio, 0.0), 0.95))
        self.cluster_center_max_radius_ratio = float(min(max(self.cluster_center_max_radius_ratio, 0.05), 0.99))
        self.action_size = 6 if bool(self.allow_activation_action) else 5
        self.reward_weight_coverage = float(max(self.reward_weight_coverage, 0.0))
        self.reward_weight_marginal_contribution = float(max(self.reward_weight_marginal_contribution, 0.0))
        self.reward_weight_movement_cost = float(max(self.reward_weight_movement_cost, 0.0))
        self.reward_weight_overlap_penalty = float(max(self.reward_weight_overlap_penalty, 0.0))
        self.reward_weight_uncovered_guidance = float(max(self.reward_weight_uncovered_guidance, 0.0))
        self.sensing_radius = float(max(self.sensing_radius, 1.0))
        self.task_initial_uncertainty = float(min(max(self.task_initial_uncertainty, 0.0), 1.0))
        self.task_min_uncertainty = float(min(max(self.task_min_uncertainty, 0.0), 1.0))
        self.task_uncertainty_reduction = float(min(max(self.task_uncertainty_reduction, 0.0), 1.0))
        self.task_initial_aoi = float(max(self.task_initial_aoi, 0.0))
        self.task_max_aoi = float(max(self.task_max_aoi, 1.0))
        self.task_priority_min = float(max(self.task_priority_min, 0.0))
        self.task_priority_max = float(max(self.task_priority_max, self.task_priority_min))
        self.trusted_sensing_uncertainty_target = float(
            min(max(self.trusted_sensing_uncertainty_target, 0.0), 1.0)
        )
        self.num_cognition_tasks = int(max(self.num_cognition_tasks, 1))
        self.cognition_num_bands = int(max(self.cognition_num_bands, 1))
        self.cognition_max_task_slots = int(max(self.cognition_max_task_slots, 1))
        self.cognition_observation_noise_std = float(max(self.cognition_observation_noise_std, 0.0))
        self.cognition_aoi_increment = float(max(self.cognition_aoi_increment, 0.0))
        self.cognition_task_uncertainty_reduction = float(
            min(max(self.cognition_task_uncertainty_reduction, 0.0), 1.0)
        )
        self.cognition_sensing_cost = float(max(self.cognition_sensing_cost, 0.0))
        self.cognition_repeat_penalty = float(max(self.cognition_repeat_penalty, 0.0))
        self.reward_weight_uncertainty_gain = float(max(self.reward_weight_uncertainty_gain, 0.0))
        self.reward_weight_aoi_gain = float(max(self.reward_weight_aoi_gain, 0.0))
        self.reward_weight_repeat_sensing_penalty = float(max(self.reward_weight_repeat_sensing_penalty, 0.0))
        self.reward_weight_sensing_cost = float(max(self.reward_weight_sensing_cost, 0.0))
        self.overlap_distance_threshold = float(max(self.overlap_distance_threshold, 1.0))
        self.guidance_distance_scale = float(max(self.guidance_distance_scale, 1.0))
        self.mc_reward_clip = float(max(self.mc_reward_clip, 0.0))

        if self.cluster_center_max_radius_ratio <= self.cluster_center_min_radius_ratio:
            self.cluster_center_max_radius_ratio = min(self.cluster_center_min_radius_ratio + 0.1, 0.99)

    def step_size(self) -> float:
        return self.uav_speed * self.dt

    def action_to_delta_xy(self) -> Dict[int, Tuple[float, float]]:
        s = self.step_size()
        return {
            0: (0.0, 0.0),
            1: (0.0, s),
            2: (0.0, -s),
            3: (-s, 0.0),
            4: (s, 0.0),
            5: (0.0, 0.0),
        }

    def get_ppo_main_local_obs_dim(self) -> int:
        self_feat_dim = 11
        user_slot_dim = 5
        neighbor_slot_dim = 5
        local_summary_dim = int(self.num_direction_sectors + self.num_radial_bins + 4)
        guidance_dim = 8
        return int(
            self_feat_dim
            + self.max_obs_users * user_slot_dim
            + self.max_obs_uavs * neighbor_slot_dim
            + local_summary_dim
            + guidance_dim
        )

    def get_mcg_local_obs_dim(self) -> int:
        mcg_extra_obs_dim = 7 + 8 + 6 + 4
        return int(self.get_ppo_main_local_obs_dim() + mcg_extra_obs_dim)

    def get_local_obs_dim(self, method_name: str | None = None) -> int:
        if bool(self.use_enhanced_obs):
            return self.get_mcg_local_obs_dim()
        return self.get_ppo_main_local_obs_dim()

    def get_resource_cognition_action_dim(self) -> int:
        """Movement actions plus one explicit sensing action per local slot."""
        return int(5 + self.cognition_max_task_slots)

    def validate(self) -> None:
        if self.r_safe <= 0 or self.r_disaster <= self.r_safe:
            raise ValueError("Require r_disaster > r_safe > 0.")

        if self.num_users <= 0:
            raise ValueError("num_users must be positive.")

        if self.user_distribution_mode not in {"uniform", "clustered", "mixed"}:
            raise ValueError("user_distribution_mode must be one of {'uniform', 'clustered', 'mixed'}.")

        if self.num_user_clusters <= 0:
            raise ValueError("num_user_clusters must be positive.")

        if not (0.0 <= self.clustered_user_ratio <= 1.0):
            raise ValueError("clustered_user_ratio must be in [0, 1].")

        if self.cluster_radius <= 0:
            raise ValueError("cluster_radius must be positive.")

        if not (0.0 <= self.edge_avoidance_ratio <= 1.0):
            raise ValueError("edge_avoidance_ratio must be in [0, 1].")

        if not (0.0 < self.edge_soft_limit_ratio < 0.5):
            raise ValueError("edge_soft_limit_ratio must be in (0, 0.5).")

        if not (0.0 <= self.cluster_center_min_radius_ratio < 1.0):
            raise ValueError("cluster_center_min_radius_ratio must be in [0, 1).")

        if not (0.0 < self.cluster_center_max_radius_ratio < 1.0):
            raise ValueError("cluster_center_max_radius_ratio must be in (0, 1).")

        if self.cluster_center_max_radius_ratio <= self.cluster_center_min_radius_ratio:
            raise ValueError("cluster_center_max_radius_ratio must be greater than cluster_center_min_radius_ratio.")

        if self.user_radial_beta_a <= 0 or self.user_radial_beta_b <= 0:
            raise ValueError("user_radial_beta_a and user_radial_beta_b must be positive.")

        if self.max_candidate_uavs <= 0:
            raise ValueError("max_candidate_uavs must be positive.")

        if self.initially_active_uavs < 0 or self.initially_active_uavs > self.max_candidate_uavs:
            raise ValueError("initially_active_uavs must be in [0, max_candidate_uavs].")

        if not (self.uav_h_min <= self.uav_init_height <= self.uav_h_max):
            raise ValueError("uav_init_height must be within [uav_h_min, uav_h_max].")

        if self.uav_speed <= 0:
            raise ValueError("uav_speed must be positive.")

        if self.uav_max_time <= 0:
            raise ValueError("uav_max_time must be positive.")

        if self.dt <= 0:
            raise ValueError("dt must be positive.")

        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive.")

        expected_action_size = 6 if bool(self.allow_activation_action) else 5
        if self.action_size != expected_action_size:
            raise ValueError(
                f"action_size must be {expected_action_size} when "
                f"allow_activation_action={self.allow_activation_action}, got {self.action_size}."
            )

        if self.obs_radius <= 0:
            raise ValueError("obs_radius must be positive.")

        if self.sensing_radius <= 0:
            raise ValueError("sensing_radius must be positive.")

        if self.num_cognition_tasks <= 0:
            raise ValueError("num_cognition_tasks must be positive.")

        if self.cognition_num_bands <= 0:
            raise ValueError("cognition_num_bands must be positive.")

        if self.cognition_max_task_slots <= 0:
            raise ValueError("cognition_max_task_slots must be positive.")

        if self.task_min_uncertainty > self.task_initial_uncertainty:
            raise ValueError("task_min_uncertainty must not exceed task_initial_uncertainty.")

        if self.task_max_aoi < self.task_initial_aoi:
            raise ValueError("task_max_aoi must be at least task_initial_aoi.")

        if self.max_obs_users < 0:
            raise ValueError("max_obs_users must be non-negative.")

        if self.max_obs_uavs < 0:
            raise ValueError("max_obs_uavs must be non-negative.")

        if self.simplified_coverage_radius <= 0:
            raise ValueError("simplified_coverage_radius must be positive.")

        if self.rl_method_family == "":
            raise ValueError("rl_method_family must not be empty.")

        if self.policy_architecture == "":
            raise ValueError("policy_architecture must not be empty.")
