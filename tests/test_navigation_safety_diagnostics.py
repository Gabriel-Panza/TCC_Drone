import unittest

import numpy as np

from spatial_mapping.navigation import SpatialNavigator


class _FakeGrid:
    def __init__(self, free, occupied, ray):
        self.free = set(free)
        self.occupied = set(occupied)
        self.ray = list(ray)

    def free_voxels(self):
        return set(self.free)

    def occupied_voxels(self):
        return set(self.occupied)

    def segment_voxels(self, _start, _end):
        return list(self.ray)

    def world_to_voxel(self, _point):
        return self.ray[0]


class PathSafetyDiagnosticsTests(unittest.TestCase):
    def test_classifies_first_unsafe_voxel(self):
        cases = (
            ("unknown", set(), set(), False),
            ("inflated", set(), {(1, 0, 0)}, True),
            ("occupied", {(1, 0, 0)}, {(1, 0, 0)}, True),
        )

        for expected, occupied, inflated, is_inflated in cases:
            with self.subTest(expected=expected):
                navigator = object.__new__(SpatialNavigator)
                navigator.grid = _FakeGrid(
                    free={(0, 0, 0)},
                    occupied=occupied,
                    ray=[(0, 0, 0), (1, 0, 0)],
                )
                navigator._inflated_obstacles = lambda: set(inflated)

                result = navigator.path_safety_diagnostics(
                    np.asarray([0.1, 0.1, 0.1]),
                    [np.asarray([1.1, 0.1, 0.1])],
                )

                self.assertFalse(result["safe"])
                self.assertEqual(result["first_unsafe_state"], expected)
                self.assertEqual(
                    result["first_unsafe_voxel"],
                    [1, 0, 0],
                )
                self.assertEqual(
                    result["first_unsafe_is_inflated"],
                    is_inflated,
                )

    def test_initial_escape_requires_known_free_exit_and_forbids_reentry(self):
        navigator = object.__new__(SpatialNavigator)
        navigator.grid = _FakeGrid(
            free={(2, 0, 0), (3, 0, 0)}, occupied={(0, 0, 0)},
            ray=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        navigator._inflated_obstacles = lambda: {(0, 0, 0), (1, 0, 0)}
        safe = navigator.initial_escape_path_safety_diagnostics(
            [0., 0., 0.], [[3., 0., 0.]])
        self.assertTrue(safe['safe'])
        self.assertEqual(safe['policy'], 'known_free_initial_escape_no_reentry')
        navigator.grid.ray = [(0,0,0), (1,0,0), (2,0,0), (1,0,0)]
        reentry = navigator.initial_escape_path_safety_diagnostics(
            [0.,0.,0.], [[3.,0.,0.]])
        self.assertFalse(reentry['safe'])
        self.assertEqual(reentry['failure_reason'], 'obstacle_reentry')
        navigator.grid.ray = [(0,0,0), (1,0,0), (4,0,0)]
        unknown = navigator.initial_escape_path_safety_diagnostics(
            [0.,0.,0.], [[4.,0.,0.]])
        self.assertFalse(unknown['safe'])
        self.assertEqual(unknown['failure_reason'], 'unknown')

    def test_dynamic_escape_only_tolerates_inflated_current_voxel(self):
        navigator = object.__new__(SpatialNavigator)
        navigator.grid = _FakeGrid(
            free={(2, 0, 0), (3, 0, 0)}, occupied=set(),
            ray=[(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        navigator._inflated_obstacles = lambda: {(0, 0, 0), (1, 0, 0)}
        result = navigator.path_safety_allowing_dynamic_initial_escape(
            [0., 0., 0.], [[3., 0., 0.]])
        self.assertTrue(result['safe'])
        self.assertTrue(result['dynamic_initial_escape_used'])
        self.assertEqual(
            result['strict_path_safety']['first_unsafe_voxel'], [0, 0, 0])

    def test_dynamic_escape_does_not_tolerate_inflated_voxel_ahead(self):
        navigator = object.__new__(SpatialNavigator)
        navigator.grid = _FakeGrid(
            free={(0, 0, 0), (2, 0, 0)}, occupied=set(),
            ray=[(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        navigator._inflated_obstacles = lambda: {(1, 0, 0)}
        result = navigator.path_safety_allowing_dynamic_initial_escape(
            [0., 0., 0.], [[2., 0., 0.]])
        self.assertFalse(result['safe'])
        self.assertFalse(result['dynamic_initial_escape_used'])
        self.assertEqual(result['first_unsafe_voxel'], [1, 0, 0])

    def test_uses_raw_voxel_fallback_when_postprocessing_is_unsafe(self):
        navigator = object.__new__(SpatialNavigator)
        candidate = [(0.0, 0.0, 0.0), (6.0, 0.0, 0.0)]
        fallback = [(0.0, 0.0, 0.0), (0.75, 0.0, 0.0)]

        def diagnose(_current, waypoints):
            safe = list(waypoints) == fallback
            return {
                "safe": safe,
                "failure_reason": None if safe else "unknown",
            }

        navigator.path_safety_diagnostics = diagnose

        selected, used, rejected, final = (
            navigator._select_safe_waypoint_representation(
                (0.0, 0.0, 0.0),
                candidate,
                fallback,
            )
        )

        self.assertTrue(used)
        self.assertEqual(selected, fallback)
        self.assertEqual(rejected["failure_reason"], "unknown")
        self.assertTrue(final["safe"])


if __name__ == "__main__":
    unittest.main()
