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

    def test_partial_encoder_unfreezes_only_last_blocks(self):
        class DummyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.pretrained = torch.nn.Module()
                self.pretrained.blocks = torch.nn.ModuleList(
                    [torch.nn.Linear(2, 2) for _ in range(4)]
                )
                self.head = torch.nn.Linear(2, 1)

        model = DummyModel()
        encoder, head = TRAINING.configure_encoder_training(
            model,
            unfreeze_encoder_blocks=2,
        )

        self.assertTrue(
            all(
                not parameter.requires_grad
                for block in model.pretrained.blocks[:2]
                for parameter in block.parameters()
            )
        )
        self.assertTrue(
            all(
                parameter.requires_grad
                for block in model.pretrained.blocks[2:]
                for parameter in block.parameters()
            )
        )
        self.assertEqual(len(encoder), 4)
        self.assertEqual(len(head), 2)

    def test_partial_encoder_rejects_conflicting_modes(self):
        model = torch.nn.Module()
        model.pretrained = torch.nn.Module()
        model.pretrained.blocks = torch.nn.ModuleList([torch.nn.Linear(1, 1)])
        model.head = torch.nn.Linear(1, 1)
        with self.assertRaises(ValueError):
            TRAINING.configure_encoder_training(
                model,
                freeze_encoder=True,
                unfreeze_encoder_blocks=1,
            )

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
