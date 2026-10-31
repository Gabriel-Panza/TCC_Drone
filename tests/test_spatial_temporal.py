import unittest
import numpy as np
from spatial_mapping.temporal import LatestObservation, PoseHistory
from spatial_mapping.geometry import body_position_from_camera_transform, camera_to_ned_transform


class TemporalAlignmentTest(unittest.TestCase):
    def test_body_envelope_recovers_camera_lever_arm_under_rotation(self):
        mount = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        offset = [.15, 0, -.22]
        body = [3., 4., -1.5]
        for q in [[1, 0, 0, 0], [.9238795, .3826834, 0, 0], [.70710678, 0, 0, .70710678]]:
            transform = camera_to_ned_transform(body, q, offset, mount)
            np.testing.assert_allclose(body_position_from_camera_transform(transform, offset, mount), body)

    def test_pose_is_interpolated_at_capture(self):
        h = PoseHistory(max_gap_s=2)
        h.add(1, [0, 0, 0], [1, 0, 0, 0])
        h.add(2, [2, 0, 0], [0, 0, 0, 1])
        p, q = h.at(1.5)
        np.testing.assert_allclose(p, [1, 0, 0])
        np.testing.assert_allclose(q, [2**-.5, 0, 0, 2**-.5])

    def test_no_extrapolation_or_large_gap(self):
        h = PoseHistory()
        h.add(1, [0, 0, 0], [1, 0, 0, 0])
        h.add(2, [0, 0, 0], [1, 0, 0, 0])
        for stamp in [.9, 1.5, 2.1]:
            self.assertIsNone(h.at(stamp))

    def test_reset_and_antipodal_quaternions(self):
        h = PoseHistory(max_gap_s=2)
        h.add(10, [100, 0, 0], [1, 0, 0, 0])
        h.add(1, [0, 0, 0], [1, 0, 0, 0])
        h.add(2, [2, 0, 0], [-1, 0, 0, 0])
        self.assertIsNone(h.at(10))
        np.testing.assert_allclose(h.at(1.5)[1], [1, 0, 0, 0])

    def test_mailbox_replaces_without_replaying(self):
        m = LatestObservation()
        self.assertTrue(m.submit(1, 'old'))
        self.assertTrue(m.submit(2, 'new'))
        self.assertEqual(m.take(), (2, 'new'))
        self.assertIsNone(m.take())
        self.assertFalse(m.submit(2, 'duplicate'))
        self.assertFalse(m.submit(1, 'late'))
        self.assertEqual(m.replaced, 1)
