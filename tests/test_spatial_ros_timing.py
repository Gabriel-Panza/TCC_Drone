"""ROS smoke test with synthetic packets; execution/arm explicitly disabled."""
import time
import unittest

import numpy as np

try:
    import rclpy
    from rclpy.time import Time
    from drone_controller import DroneOffboardNode
    from px4_msgs.msg import VehicleLocalPosition, VehicleAttitude, VehicleOdometry
    ROS_AVAILABLE = True
except ImportError:
    ROS_AVAILABLE = False


@unittest.skipUnless(ROS_AVAILABLE, 'requires sourced ROS and PX4 overlay')
class RosTemporalSmokeTest(unittest.TestCase):
    def test_v21_uses_source_pose_and_accepts_missing_reference(self):
        rclpy.init(args=['--ros-args', '--params-file', 'config/spatial_monocular_v21.yaml',
                         '-p', 'spatial_save_dataset:=false'])
        node = DroneOffboardNode()
        try:
            source_rgb = object()
            node.spatial_v21_rgb.append((10.0, source_rgb))
            node.spatial_v21_capture_poses.add(10.0, [0.0, 0.0, -1.5], [1.0, 0.0, 0.0, 0.0])
            node.spatial_v21_capture_poses.add(10.5, [2.0, 0.0, -1.5], [1.0, 0.0, 0.0, 0.0])
            node.latest_depth_estimated = np.ones((2, 2), dtype=np.float32)
            node.latest_depth_estimated_stamp_s = 10.0

            observation, age = node.obter_observacao_v21(10.5, None)
            np.testing.assert_array_equal(observation['position'], [0.0, 0.0, -1.5])
            self.assertEqual(age, 0.5)
            self.assertIs(observation['source_rgb'], source_rgb)
            self.assertIsNone(observation['reference_transform'])

            observation, _ = node.obter_observacao_v21(10.5, 10.5)
            np.testing.assert_array_equal(observation['reference_position'], [2.0, 0.0, -1.5])
            observation, age = node.obter_observacao_v21(11.2, None)
            self.assertAlmostEqual(age, 1.2)
            np.testing.assert_array_equal(observation['position'], [0.0, 0.0, -1.5])
            observation, age = node.obter_observacao_v21(11.4, None)
            self.assertIsNone(observation)
            self.assertAlmostEqual(age, 1.4)
            self.assertEqual(node.spatial_v21_reference_max_age_s, 0.60)
            node.spatial_last_integrated_depth_stamp_s = 10.0
            self.assertEqual(node.obter_observacao_v21(10.5, None), (None, None))
        finally:
            node.destroy_node()
            rclpy.shutdown()

    def test_delayed_unique_depth_uses_capture_pose_without_arming(self):
        rclpy.init(args=['--ros-args', '--params-file', 'config/spatial_monocular_temporal.yaml',
                         '-p', 'spatial_save_dataset:=false',
                         '-p', 'spatial_protect_pending_obstacles:=true'])
        node = DroneOffboardNode()
        try:
            self.assertFalse(node.spatial_execute_path)
            self.assertFalse(node.voo_iniciado)
            self.assertTrue(node.spatial_estimated_navigator.config.protect_pending_obstacles)
            self.assertTrue(node.spatial_estimated_navigator.grid.config.latch_pending_obstacles)
            position = VehicleLocalPosition()
            position.x, position.y, position.z = 0., 0., -1.5
            node.pos_callback(position)
            attitude = VehicleAttitude()
            attitude.q = [1., 0., 0., 0.]
            node.attitude_callback(attitude)
            runtime = node.spatial_observation_runtime
            node.get_clock()._set_ros_time_is_active(True)
            for stamp, x in [(10., 0.), (10.45, .09), (10.55, .11), (11., .2)]:
                node.get_clock().set_ros_time_override(Time(seconds=stamp))
                odometry = VehicleOdometry()
                odometry.pose_frame = VehicleOdometry.POSE_FRAME_NED
                odometry.position = [x, 0., -1.5]
                odometry.q = [1., 0., 0., 0.]
                runtime.receive_pose(odometry)
            self.assertAlmostEqual(node.get_clock().now().nanoseconds * 1e-9, 11.)
            msg = node.bridge.cv2_to_imgmsg(np.zeros((960,1280,3), dtype=np.uint8), encoding='bgr8')
            msg.header.stamp.sec, msg.header.stamp.nanosec = 10, 500_000_000
            node.image_callback(msg)
            reference = np.full((320,240), 5., dtype=np.float32)
            runtime.receive_reference(10.5, reference)
            runtime.submit(10.5, reference)
            deadline = time.perf_counter() + 8
            while node.spatial_frames_integrated < 1 and time.perf_counter() < deadline:
                time.sleep(.01)
            self.assertEqual(node.spatial_frames_integrated, 1, (runtime.fault, dict(runtime.counts)))
            self.assertIsNone(runtime.fault)
            self.assertEqual(node.spatial_last_observation_stamp_s, 10.5)
            runtime.submit(10.5, reference)
            time.sleep(.15)
            self.assertEqual(node.spatial_frames_integrated, 1)
            self.assertEqual(runtime.mailbox.duplicates, 1)
            self.assertFalse(node.voo_iniciado)
        finally:
            node.destroy_node()
            rclpy.shutdown()
