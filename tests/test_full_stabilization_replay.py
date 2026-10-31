import unittest
import numpy as np
from scripts.reproduce_full_stabilization import make_controller
class FullStabilizationTest(unittest.TestCase):
 def controller(self):
  s=make_controller(np.eye(3),dict(pan_gyro_weight=.35,pan_yaw_gain=1.,pan_max_delta_rad=np.deg2rad(12),stabilization_zoom=1.25))
  s.current_roll=0.;s.current_pitch=0.;s.current_yaw_attitude=.1
  s.current_gyro_rad_s=np.array([0.,0.,2.])
  s.prev_stabilization_stamp_s=1.;s.prev_stabilization_yaw=0.
  return s
 def test_mixture_uses_real_gyro_not_pose_derived_surrogate(self):
  s=self.controller();pan,source=s.calcular_delta_pan_imu(1.04)
  self.assertEqual(source,"attitude+gyro")
  self.assertAlmostEqual(pan,-(.65*.1+.35*2*.04))
 def test_sparse_rgb_disables_gyro_as_in_original_controller(self):
  s=self.controller();pan,source=s.calcular_delta_pan_imu(1.66)
  self.assertEqual(source,"attitude");self.assertAlmostEqual(pan,-.1)
 def test_angle_wrap_and_clipping(self):
  s=self.controller();s.current_yaw_attitude=-np.pi+.01;s.prev_stabilization_yaw=np.pi-.01
  _,source=s.calcular_delta_pan_imu(1.66)
  self.assertEqual(source,"attitude");self.assertAlmostEqual(s.last_pan_delta_rad,-.02)
  s=self.controller();s.current_yaw_attitude=1.
  pan,_=s.calcular_delta_pan_imu(1.66)
  self.assertAlmostEqual(pan,-np.deg2rad(12))
 def test_homography_contains_pan_rotation(self):
  s=self.controller();h,pan,source=s.montar_homografia_compensacao_imu(1.04)
  k=np.array([[1.25,0,320],[0,1.25,240],[0,0,1]])
  ry=np.array([[np.cos(pan),0,np.sin(pan)],[0,1,0],[-np.sin(pan),0,np.cos(pan)]])
  np.testing.assert_allclose(h,k@ry)
