"""Continuidade geometricamente conservadora, sem alterar o controle PX4."""
import unittest
from spatial_mapping.execution import evaluate_path_handoff


class PathHandoffTest(unittest.TestCase):
    def check(self, old, new, **overrides):
        options = dict(existing_safe=True, candidate_safe=True, acceptance_m=.8,
                       min_extension_m=1., require_extension=True)
        options.update(overrides)
        return evaluate_path_handoff((0., 0., 0.), old, new, (10., 0., 0.), **options)

    def test_reverse_is_preserved_even_with_large_endpoint_gain(self):
        d = self.check([(-2., 0., 0.)], [(4., 0., 0.)])
        self.assertEqual(d["action"], "preserve")
        self.assertEqual(d["turn_angle_deg"], 180.)
        self.assertGreater(d["endpoint_progress_gain_m"], 1.)

    def test_unknown_or_inflated_existing_path_is_never_preserved(self):
        self.assertEqual(self.check([(-2., 0., 0.)], [(4., 0., 0.)],
                                   existing_safe=False)["action"], "adopt")

    def test_both_unsafe_are_rejected(self):
        self.assertEqual(self.check([(-2., 0., 0.)], [(4., 0., 0.)],
                                   existing_safe=False, candidate_safe=False)["action"], "reject")

    def test_unsafe_candidate_keeps_only_a_safe_existing_path(self):
        self.assertEqual(self.check([(2., 0., 0.)], [(4., 0., 0.)],
                                   candidate_safe=False)["action"], "preserve")

    def test_aligned_extension_is_not_blocked(self):
        self.assertEqual(self.check([(2., 0., 0.)], [(4., 0., 0.)])["action"], "adopt")

    def test_consumed_path_does_not_keep_commitment(self):
        self.assertEqual(self.check([], [(-3., 0., 0.)])["action"], "adopt")

    def test_near_entry_waypoint_does_not_hide_reverse_direction(self):
        self.assertEqual(self.check([(-2., 0., 0.)],
                                   [(-.2, .1, 0.), (4., 0., 0.)])["action"], "preserve")

    def test_small_forward_gain_keeps_previous_extension_rule(self):
        self.assertEqual(self.check([(2., 0., 0.)], [(2.5, 0., 0.)])["action"], "preserve")

    def test_nonfinite_geometry_fails_closed(self):
        self.assertEqual(self.check([(2., 0., 0.)],
                                   [(float("nan"), 0., 0.)])["action"], "reject")


if __name__ == "__main__":
    unittest.main()
