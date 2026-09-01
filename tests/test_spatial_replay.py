"""Regressoes da ordem aproximada do replay de eventos salvos."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from estudos_e_analises.diagnosticar_planejamento_espacial import replay_event_prefix


class ReplayEventPrefixTest(unittest.TestCase):
    def test_preserves_frame_ego_frame_order_without_connecting_cells(self):
        calls = []
        navigator = MagicMock()
        navigator.integrate_depth.side_effect = lambda *args, **kwargs: (
            calls.append("frame") or {
                "points_integrated": 1, "free_only_rays": 2, "total_valid_rays": 3
            }
        )
        navigator.grid.mark_ego_voxel_free.side_effect = (
            lambda position: calls.append(tuple(position))
        )
        frame = {
            "camera_to_ned": np.eye(4),
            "intrinsics": [1, 1, 0, 0],
            "estimated_depth_m": np.ones((1, 1)),
        }
        loaded = MagicMock()
        loaded.__enter__.return_value = frame
        events = [
            {"event": "frame", "file": "first.npz"},
            {"event": "plan", "map_kind": "estimated", "plan": {
                "diagnostics": {"current_position_ned_m": [1, 2, 3]}}},
            {"event": "frame", "file": "second.npz"},
        ]
        with patch(
            "estudos_e_analises.diagnosticar_planejamento_espacial.np.load",
            return_value=loaded,
        ):
            transform, totals = replay_event_prefix(navigator, Path("."), events)
        self.assertEqual(calls, ["frame", (1, 2, 3), "frame"])
        self.assertEqual(totals["historical_ego_updates"], 1)
        self.assertEqual(totals["total_valid_rays"], 6)
        np.testing.assert_array_equal(transform, np.eye(4))
        navigator.plan.assert_not_called()
        navigator.grid.mark_free_sphere.assert_not_called()

    def test_reference_plans_and_missing_positions_do_not_clear_cells(self):
        navigator = MagicMock()
        _, totals = replay_event_prefix(navigator, Path("."), [
            {"event": "plan", "map_kind": "reference", "plan": {
                "diagnostics": {"current_position_ned_m": [1, 2, 3]}}},
            {"event": "plan", "map_kind": "estimated", "plan": {}},
        ])
        navigator.grid.mark_ego_voxel_free.assert_not_called()
        self.assertEqual(totals["historical_ego_updates"], 0)
        self.assertEqual(totals["historical_ego_positions_missing"], 1)


if __name__ == "__main__":
    unittest.main()
