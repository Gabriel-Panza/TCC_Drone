"""Uma janela para a saida ja executada; nunca um reset por novo A*."""
from dataclasses import replace
import json
import unittest

from spatial_mapping.execution import RecoveryExitWindow, RecoveryProgressGuard


class ExitWindowTest(unittest.TestCase):
    path = ((5., 0., 0.), (10., 0., 0.))

    def moving(self):
        w = RecoveryExitWindow().observe((0., 0., 0.), self.path, 0, 26., 1.)
        return w.observe((1.2, 0., 0.), self.path, 0, 27.5, 1.)

    def decide(self, w, now=28., **kw):
        args = dict(base_deadline_s=28., path_safe=True, endpoint_progress_m=5.,
                    arrival=False, min_progress_m=1., corridor_m=.8)
        args.update(kw)
        return w.decide(now, **args)

    def test_grant_once_for_measured_progress(self):
        w, check = self.decide(self.moving())
        self.assertTrue(check['allowed'])
        self.assertTrue(check['granted_now'])
        self.assertEqual(w.grace_deadline_s, 34.)
        w = w.observe((2.3, 0., 0.), self.path, 0, 29., 1.)
        w, check = self.decide(w, 29.)
        self.assertTrue(check['allowed'])
        self.assertFalse(check['granted_now'])
        self.assertEqual(w.grace_deadline_s, 34.)

    def test_prediction_alone_is_not_measured_progress(self):
        w = RecoveryExitWindow().observe((0., 0., 0.), self.path, 0, 27., 1.)
        self.assertFalse(self.decide(w)[1]['allowed'])

    def test_lateral_detour_without_endpoint_progress_gets_no_grace(self):
        self.assertFalse(self.decide(self.moving(), endpoint_progress_m=.9)[1]['allowed'])

    def test_unsafe_path_never_gets_grace(self):
        self.assertFalse(self.decide(self.moving(), path_safe=False)[1]['allowed'])

    def test_stall_ends_window_even_with_active_path(self):
        w, _ = self.decide(self.moving())
        w = w.observe((1.2, 0., 0.), self.path, 0, 30.5, 1.)
        check = self.decide(w, 30.5)[1]
        self.assertFalse(check['allowed'])
        self.assertEqual(check['reason'], 'exit_stalled')

    def test_oscillation_does_not_reset_progress_milestones(self):
        w, _ = self.decide(self.moving())
        for now, x in ((28.5, .5), (29., 1.2), (30., .5), (30.5, 1.2)):
            w = w.observe((x, 0., 0.), self.path, 0, now, 1.)
        self.assertEqual(w.last_progress_s, 27.5)
        self.assertFalse(self.decide(w, 30.5)[1]['allowed'])

    def test_regression_and_off_corridor_end_window(self):
        w, _ = self.decide(self.moving())
        for point, reason in (((0., 0., 0.), 'exit_regressed'),
                              ((2., 1., 0.), 'exit_off_corridor')):
            sample = w.observe(point, self.path, 0, 28.5, 1.)
            self.assertEqual(self.decide(sample, 28.5)[1]['reason'], reason)

    def test_waypoint_consumption_is_not_motion(self):
        w = RecoveryExitWindow().observe((0., 0., 0.), self.path, 0, 26., 1.)
        w = w.observe((0., 0., 0.), self.path, 1, 27., 1.)
        self.assertEqual(w.progress_m, 0.)
        self.assertFalse(self.decide(w)[1]['allowed'])

    def test_new_plan_cannot_renew_grace(self):
        w, _ = self.decide(self.moving())
        new_path = ((6., 0., 0.), (11., 0., 0.))
        w = w.observe((1.2, 0., 0.), new_path, 0, 28.1, 1.)
        w = w.observe((3., 0., 0.), new_path, 0, 29., 1.)
        check = self.decide(w, 29.)[1]
        self.assertFalse(check['allowed'])
        self.assertEqual(check['reason'], 'exit_path_changed')
        self.assertEqual(w.grace_deadline_s, 34.)

    def test_waiting_never_uses_grace(self):
        w, _ = self.decide(self.moving())
        w = w.observe((1.2, 0., 0.), self.path, len(self.path), 28.1, 1.)
        self.assertFalse(self.decide(w, 28.1)[1]['allowed'])

    def test_hard_deadline_survives_continuous_progress_and_late_check(self):
        w, _ = self.decide(self.moving())
        w = w.observe((7., 0., 0.), self.path, 1, 33.9, 1.)
        self.assertEqual(self.decide(w, 34.)[1]['reason'], 'exit_grace_expired')
        self.assertFalse(self.decide(self.moving(), 40.)[1]['allowed'])

    def test_corner_progress_measured_along_frozen_polyline(self):
        path = ((0., 3., 0.), (4., 3., 0.))
        w = RecoveryExitWindow().observe((0., 0., 0.), path, 0, 26., 1.)
        w = w.observe((2., 3., 0.), path, 1, 27.5, 1.)
        self.assertAlmostEqual(w.progress_m, 5.)

    def test_window_is_frozen_and_serializable_in_guard(self):
        g = RecoveryProgressGuard(0, (10., 0., 0.), (0., 0., 0.), (0., 0., 0.))
        g.exit_window = self.moving()
        frozen = replace(g)
        g.exit_window, _ = self.decide(g.exit_window)
        self.assertIsNone(frozen.exit_window.grace_deadline_s)
        restored = RecoveryProgressGuard(**json.loads(json.dumps(g.as_dict())))
        self.assertEqual(restored.exit_window, g.exit_window)


if __name__ == '__main__':
    unittest.main()
