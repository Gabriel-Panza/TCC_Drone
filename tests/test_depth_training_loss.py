import importlib.util
import unittest
from pathlib import Path

import torch


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "finetune_depth_anything_baylands.py"
SPEC = importlib.util.spec_from_file_location("depth_training", MODULE_PATH)
TRAINING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAINING)


class BidirectionalDepthLossTest(unittest.TestCase):
    def test_underestimate_penalty_increases_loss_and_has_finite_gradient(self):
        target = torch.full((1, 4, 4), 10.0)
        prediction = torch.full((1, 4, 4), 5.0, requires_grad=True)
        baseline = TRAINING.masked_loss(
            prediction, target,
            unsafe_underestimate_weight=0.0,
            unsafe_underestimate_tail_weight=0.0,
            edge_underestimate_weight=0.0,
        )
        balanced = TRAINING.masked_loss(
            prediction, target,
            unsafe_underestimate_weight=1.0,
            unsafe_underestimate_tail_weight=1.0,
            edge_underestimate_weight=0.0,
        )
        self.assertGreater(balanced.item(), baseline.item())
        balanced.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_underestimate_penalty_does_not_penalize_overestimate(self):
        target = torch.full((1, 4, 4), 5.0)
        prediction = torch.full((1, 4, 4), 10.0)
        without = TRAINING.masked_loss(
            prediction, target,
            unsafe_underestimate_weight=0.0,
            unsafe_underestimate_tail_weight=0.0,
            edge_underestimate_weight=0.0,
        )
        with_penalty = TRAINING.masked_loss(
            prediction, target,
            unsafe_underestimate_weight=1.0,
            unsafe_underestimate_tail_weight=1.0,
            edge_underestimate_weight=1.0,
        )
        self.assertAlmostEqual(without.item(), with_penalty.item(), places=6)


if __name__ == "__main__":
    unittest.main()
