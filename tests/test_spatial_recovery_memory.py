"""Memoria de trajetoria real: nao confundir caminho aceito com percorrido."""
from dataclasses import replace
import json
import unittest

from spatial_mapping.execution import RecoveryCandidatePolicy, RecoveryProgressGuard


class RecoveryMemoryTest(unittest.TestCase):
    def guard(self):
        return RecoveryProgressGuard(0, (10., 0., 0.), (0., 0., 0.),
                                     (0., 0., 0.), 10.)

    def check(self, guard, points, safe=True):
        return guard.evaluate((-2., 5., 0.), points, path_safe=safe,
                              extension_m=1., blocked_radius_m=.8,
                              arrival_radius_m=1.5)

    def trace(self, guard):
        for p in [(-2., 0., 0.), (-2., 2., 0.), (-2., 5., 0.)]:
            guard.record_position(p, spacing_m=.2)

    def test_reject_retracing_consumed_detour_without_active_path(self):
        guard = self.guard()
        self.trace(guard)
        check = self.check(guard, [(-2., 2., 0.)])
        self.assertFalse(check['allowed'])
        self.assertEqual(check['reason'], 'revisited_recovery_corridor')
        self.assertAlmostEqual(check['endpoint_distance_to_visited_m'], 0.)

    def test_distance_uses_segments_not_only_saved_samples(self):
        guard = self.guard()
        guard.record_position((-2., 0., 0.))
        guard.record_position((-2., 5., 0.))
        self.assertFalse(self.check(guard, [(-2., 2.5, 0.)])['allowed'])

    def test_known_corridor_may_lead_to_new_lateral_exit(self):
        guard = self.guard()
        self.trace(guard)
        check = self.check(guard, [(-2., 2., 0.), (-4., 2., 0.)])
        self.assertTrue(check['allowed'])
        self.assertEqual(check['reason'], 'safe_lateral_detour')

    def test_real_goal_progress_can_revisit_corridor(self):
        guard = self.guard()
        guard.record_position((2., 0., 0.))
        self.assertTrue(self.check(guard, [(2., 0., 0.)])['allowed'])

    def test_new_exit_never_bypasses_safety(self):
        guard = self.guard()
        self.trace(guard)
        self.assertFalse(self.check(guard, [(-4., 2., 0.)], safe=False)['allowed'])

    def test_candidate_evaluation_does_not_record_unflown_path(self):
        guard = self.guard()
        self.trace(guard)
        before = guard.as_dict()
        for _ in range(5):
            self.assertTrue(self.check(guard, [(-4., 2., 0.)])['allowed'])
        self.assertEqual(before, guard.as_dict())

    def test_worker_snapshot_does_not_share_mutable_history(self):
        guard = self.guard()
        self.trace(guard)
        frozen = replace(guard)
        guard.record_position((-4., 2., 0.))
        self.assertTrue(self.check(frozen, [(-4., 2., 0.)])['allowed'])
        self.assertFalse(self.check(guard, [(-4., 2., 0.)])['allowed'])

    def test_policy_json_roundtrip_preserves_history_and_legacy_loads(self):
        guard = self.guard()
        self.trace(guard)
        policy = RecoveryCandidatePolicy(guard, .8, 1.5, 1., 1.5)
        restored = RecoveryCandidatePolicy.from_dict(json.loads(json.dumps(policy.as_dict())))
        self.assertFalse(self.check(restored.guard, [(-2., 2., 0.)])['allowed'])
        restored.guard.record_position((-4., 2., 0.))
        legacy = dict(goal_index=0, goal_ned_m=(10., 0., 0.),
                      stopped_position_ned_m=(0., 0., 0.),
                      stopped_endpoint_ned_m=(0., 0., 0.))
        self.assertEqual(RecoveryProgressGuard(**legacy).visited_positions_ned_m, ())

    def test_history_never_evicts_old_corridor_when_full(self):
        guard = self.guard()
        for i in range(guard.MAX_TRACE_POINTS + 1):
            guard.record_position((-2., float(i), 0.))
        self.assertEqual(len(guard.visited_positions_ned_m), guard.MAX_TRACE_POINTS)
        self.assertEqual(guard.visited_positions_ned_m[0], (-2., 0., 0.))
        self.assertTrue(guard.history_saturated)
        self.assertFalse(self.check(guard, [(-4., 2., 0.)])['allowed'])

    def test_wait_and_active_path_times_do_not_extend_total_deadline(self):
        guard = self.guard()
        guard.observe((-2., 0., 0.), 10., path_missing=True)
        guard.observe((-2., 0., 0.), 12., path_missing=False)
        guard.observe((-2., 5., 0.), 18., path_missing=True)
        guard.observe((-2., 5., 0.), 28., path_missing=True)
        self.assertEqual(guard.waiting_elapsed_s, 12.)
        self.assertEqual(guard.active_path_elapsed_s, 6.)
        self.assertTrue(guard.observation_expired(28., 18.))
        self.assertEqual(guard.observation_started_s, 10.)


if __name__ == '__main__':
    unittest.main()
