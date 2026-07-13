import numpy as np
import unittest

from envs.task_model import LocalBeliefBatch, TaskTruthBatch


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
