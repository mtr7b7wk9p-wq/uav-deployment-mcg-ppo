import numpy as np
import unittest

from envs.task_model import TaskTruthBatch


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
