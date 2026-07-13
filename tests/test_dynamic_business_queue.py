import numpy as np
import unittest

from envs.task_model import LocalBeliefBatch, TaskTruthBatch
from configs.scenario_config import ScenarioConfig
from envs.resource_cognition_env import ResourceCognitionEnv
from agents.ppo.models import ResourceCognitionObsSliceSpec
from envs.metrics import EpisodeMetrics
from envs.channel import (
    channel_gain_from_path_loss_db,
    shannon_rate_mbps,
)


def make_truth(queue_lengths, queue_capacity=10.0):
    count = len(queue_lengths)
    return TaskTruthBatch(
        positions_xy=np.zeros((count, 2), dtype=np.float32),
        band_ids=np.arange(count, dtype=np.int32),
        true_states=np.zeros(count, dtype=np.float32),
        priorities=np.ones(count, dtype=np.float32),
        demand_levels=np.ones(count, dtype=np.float32),
        arrival_rates=np.zeros(count, dtype=np.float32),
        queue_lengths=np.asarray(queue_lengths, dtype=np.float32),
        queue_capacity=queue_capacity,
    )


def make_beliefs(num_agents=2, num_tasks=1):
    return LocalBeliefBatch(
        num_agents=num_agents,
        task_priorities=np.ones(num_tasks, dtype=np.float32),
    )


def make_env(num_tasks=2, max_steps=5):
    return ResourceCognitionEnv(
        ScenarioConfig(
            use_resource_cognition=True,
            max_candidate_uavs=2,
            num_uavs=2,
            num_cognition_tasks=num_tasks,
            cognition_max_task_slots=num_tasks,
            max_obs_uavs=1,
            max_steps=max_steps,
            uav_init_mode="center",
        )
    )


class DynamicBusinessQueueTests(unittest.TestCase):
    def test_queue_adds_arrivals_without_service(self):
        truth = make_truth([1.0, 0.0])
        stats = truth.advance_business(np.array([2.0, 3.0], dtype=np.float32))
        self.assertTrue(np.allclose(truth.queue_lengths, [3.0, 3.0]))
        self.assertEqual(stats["total_arrivals"], 5.0)
        self.assertEqual(stats["queue_overflow"], 0.0)


    def test_service_is_capped_by_queue_and_capacity(self):
        truth = make_truth([2.0, 10.0])
        truth.advance_business(np.zeros(2, dtype=np.float32))
        served = truth.apply_service(np.array([5.0, 4.0], dtype=np.float32))
        self.assertTrue(np.allclose(truth.queue_lengths, [0.0, 6.0]))
        self.assertEqual(served, 6.0)

    def test_queue_belief_changes_only_after_accepted_message(self):
        beliefs = make_beliefs()
        before = beliefs.queue_estimates[1, 0]
        result = beliefs.fuse_neighbor_message(
            receiver_id=1,
            task_id=0,
            estimate=0.0,
            uncertainty=0.2,
            confidence=0.8,
            message_aoi=0.0,
            queue_estimate=7.0,
            queue_uncertainty=0.2,
            queue_confidence=0.8,
            queue_aoi=0.0,
            arrival_estimate=2.0,
            source_update_step=1,
            current_step=1,
            confidence_threshold=0.05,
            freshness_decay=0.1,
        )
        self.assertEqual(result["queue_accepted"], 1.0)
        self.assertNotEqual(beliefs.queue_estimates[1, 0], before)
        self.assertGreater(beliefs.queue_estimates[1, 0], 0.0)

    def test_scheduling_reduces_queue_and_reports_service_rate(self):
        env = make_env()
        env.reset(seed=7)
        env.task_truth.queue_lengths[:] = [8.0, 0.0]
        env.task_truth.arrival_rates[:] = 0.0
        stats = env._execute_scheduling(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
        )
        self.assertGreater(stats["served_data"], 0.0)
        self.assertLess(env.task_truth.queue_lengths[0], 8.0)
        self.assertGreater(stats["service_rate"], 0.0)
        self.assertLessEqual(stats["service_rate"], 1.0)

    def test_same_band_conflict_reduces_service(self):
        env = make_env()
        env.reset(seed=8)
        env.task_truth.band_ids[:] = 0
        env.task_truth.queue_lengths[:] = 8.0
        env.task_truth.positions_xy[1] = env.task_truth.positions_xy[0]
        stats = env._execute_scheduling(
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        )
        self.assertGreater(stats["conflict_count"], 0)
        self.assertLessEqual(stats["service_by_agent"][0], stats["capacity_by_agent"][0])

    def test_dynamic_queue_observation_has_fixed_dimension_without_truth_leak(self):
        env = make_env(num_tasks=2, max_steps=3)
        first = env.reset(seed=9)["local_obs"].copy()
        expected_dim = 6 + 17 * env.cfg.cognition_max_task_slots + 17 * env.cfg.max_obs_uavs
        self.assertEqual(first.shape[-1], expected_dim)
        env.task_truth.queue_lengths[:] = 10.0
        second = env.get_local_obs(0)
        self.assertEqual(second.shape, first[0].shape)
        self.assertTrue(np.allclose(second, first[0]))

    def test_structured_encoder_uses_dynamic_slot_dimensions(self):
        env = make_env(num_tasks=2)
        spec = ResourceCognitionObsSliceSpec(
            local_obs_dim=env.local_obs_dim,
            num_task_slots=env.cfg.cognition_max_task_slots,
            num_message_slots=env.cfg.max_obs_uavs,
        )
        self.assertEqual(spec.task_slot_dim, 17)
        self.assertEqual(spec.message_slot_dim, 17)
        self.assertEqual(spec.expected_dim, env.local_obs_dim)

    def test_resource_action_dimension_has_three_schedule_levels(self):
        env = make_env(num_tasks=2)
        self.assertEqual(env.action_size, 5 + 2 + 3 * 2)

    def test_schedule_action_decodes_task_and_resource_level(self):
        env = make_env(num_tasks=2)
        env.reset(seed=10)
        env._slot_task_indices = [
            np.array([0, 1], dtype=np.int64),
            np.array([0, 1], dtype=np.int64),
        ]
        start = env._schedule_action_start()
        actions = np.array([start, start + 2 * env.cfg.cognition_max_task_slots + 1])
        agents, tasks, levels = env._decode_schedule_actions(actions)
        self.assertTrue(np.array_equal(agents, [0, 1]))
        self.assertTrue(np.array_equal(tasks, [0, 1]))
        self.assertTrue(np.array_equal(levels, [0, 2]))

    def test_resource_level_changes_raw_physical_capacity(self):
        env = make_env()
        env.reset(seed=13)
        env.task_truth.true_states[:] = 0.0
        env.cfg.cognition_max_service_per_step = 1e6
        assignments = np.array([0, -1], dtype=np.int64)
        conflicts = np.zeros(env.num_agents, dtype=np.float32)
        low = env._physical_service_capacity(
            assignments,
            conflicts,
            use_truth=True,
            resource_levels=np.array([0, 1], dtype=np.int64),
        )
        high = env._physical_service_capacity(
            assignments,
            conflicts,
            use_truth=True,
            resource_levels=np.array([2, 1], dtype=np.int64),
        )
        self.assertGreater(high[4][0], low[4][0])
        self.assertGreater(high[0][0], low[0][0])

    def test_scheduling_reports_capacity_clipping(self):
        env = make_env()
        env.reset(seed=14)
        env.task_truth.queue_lengths[:] = 100.0
        env.cfg.cognition_max_service_per_step = 1.0
        stats = env._execute_scheduling(
            np.array([0], dtype=np.int64),
            np.array([0], dtype=np.int64),
            np.array([2, 1], dtype=np.int64),
        )
        self.assertGreater(stats["mean_raw_service_capacity"], stats["mean_service_capacity"])
        self.assertGreater(stats["capacity_clip_ratio"], 0.0)
        self.assertGreater(stats["capacity_clipped_count"], 0)

    def test_metrics_collect_dynamic_service_fields(self):
        metrics = EpisodeMetrics()
        metrics.update(1.0, {
            "total_arrivals": 3.0,
            "scheduling_served_data": 2.0,
            "total_queue_length": 4.0,
            "service_rate": 0.5,
            "weighted_demand_satisfaction": 0.25,
            "high_priority_service_rate": 0.75,
            "queue_overflow": 1.0,
            "service_energy_consumption": 2.0,
        })
        summary = metrics.summary()
        self.assertEqual(summary["total_arrivals"], 3.0)
        self.assertEqual(summary["total_served_data"], 2.0)
        self.assertEqual(summary["final_total_queue"], 4.0)
        self.assertEqual(summary["weighted_demand_satisfaction"], 0.25)

    def test_path_loss_gain_and_shannon_rate_are_monotonic(self):
        gain = channel_gain_from_path_loss_db(
            np.array([80.0, 100.0], dtype=np.float32)
        )
        rate = shannon_rate_mbps(
            1.0,
            np.array([1.0, 3.0], dtype=np.float32),
        )
        self.assertGreater(gain[0], gain[1])
        self.assertGreater(rate[1], rate[0])

    def test_physical_config_rejects_non_positive_parameters(self):
        with self.assertRaises(ValueError):
            ScenarioConfig(
                use_resource_cognition=True,
                cognition_bandwidth_mhz=0.0,
            ).validate()

    def test_physical_service_capacity_decreases_with_noise(self):
        env = make_env()
        env.reset(seed=11)
        env.task_truth.true_states[:] = 0.0
        assignments = np.array([0, -1], dtype=np.int64)
        conflicts = np.zeros(env.num_agents, dtype=np.float32)
        first = env._physical_service_capacity(assignments, conflicts, use_truth=True)
        env.cfg.cognition_noise_power_w *= 100.0
        second = env._physical_service_capacity(assignments, conflicts, use_truth=True)
        self.assertLessEqual(second[1][0], first[1][0])
        self.assertLessEqual(second[0][0], first[0][0])

    def test_same_band_interference_reduces_physical_capacity(self):
        env = make_env()
        env.reset(seed=12)
        env.task_truth.true_states[:] = 0.0
        env.task_truth.band_ids[:] = 0
        env.task_truth.positions_xy[1] = env.task_truth.positions_xy[0]
        assignments = np.array([0, 1], dtype=np.int64)
        conflicts = np.zeros(env.num_agents, dtype=np.float32)
        with_interference = env._physical_service_capacity(
            assignments,
            conflicts,
            use_truth=True,
        )
        without_interference = env._physical_service_capacity(
            np.array([0, -1], dtype=np.int64),
            conflicts,
            use_truth=True,
        )
        self.assertLess(with_interference[1][0], without_interference[1][0])
        self.assertLess(with_interference[0][0], without_interference[0][0])

    def test_metrics_collect_physical_service_fields(self):
        metrics = EpisodeMetrics()
        metrics.update(1.0, {
            "mean_path_loss_db": 95.0,
            "mean_sinr": 4.0,
            "mean_service_capacity": 1.5,
            "mean_raw_service_capacity": 3.0,
            "capacity_clip_ratio": 0.5,
            "capacity_clipped_count": 1,
            "mean_resource_level": 2.0,
            "service_outage_count": 1,
            "service_outage_rate": 0.5,
            "total_interference_power_w": 2e-13,
        })
        summary = metrics.summary()
        self.assertEqual(summary["mean_path_loss_db"], 95.0)
        self.assertEqual(summary["mean_sinr"], 4.0)
        self.assertEqual(summary["total_service_outages"], 1)
        self.assertEqual(summary["mean_raw_service_capacity"], 3.0)
        self.assertEqual(summary["mean_capacity_clip_ratio"], 0.5)
        self.assertEqual(summary["total_capacity_clipped_count"], 1)
