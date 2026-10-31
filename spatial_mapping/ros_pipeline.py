"""Opt-in ROS scheduling for the next monocular experiment (not a qualification)."""
import copy
import threading
import time
from collections import Counter, deque

import cv2
import numpy as np
from px4_msgs.msg import TrajectorySetpoint, VehicleOdometry
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String
import json

from .geometry import camera_to_ned_transform
from .temporal import LatestObservation, PoseHistory


class SpatialObservationRuntime:
    """Keep sensor reception, mapping and the command heartbeat independent.

    Pose samples are coherent VehicleOdometry samples indexed by ROS reception
    time in /clock, NOT alleged hardware-exact PX4 sample times. Transport error
    remains explicit and must be measured before a flight qualification.
    """
    def __init__(self, controller):
        self.node = controller
        self.mapping_size = (
            int(controller.declare_parameter('spatial_mapping_width', 320).value),
            int(controller.declare_parameter('spatial_mapping_height', 240).value))
        if min(self.mapping_size) < 1:
            raise ValueError('mapping dimensions must be positive')
        if not controller.get_parameter('use_sim_time').value:
            raise ValueError('spatial_async_mapping requires use_sim_time and a live /clock')
        self.history = PoseHistory()
        self.mailbox = LatestObservation()
        self.rgb = deque(maxlen=48)
        self.reference = deque(maxlen=32)
        self.input_timings = {}
        self.published_source_timings = set()
        self.command_source_stamp = None
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.command = None
        self.last_source_stamp = None
        self.last_mapping_wall = None
        self.fault = None
        self.counts = Counter()
        self.timings = deque(maxlen=20000)
        self.trajectory = deque(maxlen=40000)
        self.last_heartbeat = None
        self.group = MutuallyExclusiveCallbackGroup()
        self.odometry = controller.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.receive_pose,
            qos_profile_sensor_data, callback_group=controller.spatial_sensor_group)
        self.timing_subscription = controller.create_subscription(
            String, controller.monocular_depth_topic + '/timing', self.receive_timing,
            qos_profile_sensor_data, callback_group=controller.spatial_sensor_group)
        self.timer = controller.create_timer(.04, self.heartbeat, callback_group=self.group)
        self.worker = threading.Thread(target=self.loop, name='spatial-observation-worker', daemon=True)
        self.worker.start()

    def receive_pose(self, msg):
        # PX4 boot/sample timestamps are a different domain; never equate them
        # silently with image timestamps. ROS receipt alignment is recorded.
        if int(msg.pose_frame) != int(VehicleOdometry.POSE_FRAME_NED):
            self.counts['unsupported_odometry_frame'] += 1
            return
        stamp = self.node.get_clock().now().nanoseconds * 1e-9
        accepted = self.history.add(stamp, msg.position, msg.q)
        if accepted and self.node.spatial_takeoff_complete and not self.node.missao_concluida:
            self.trajectory.append(dict(stamp_s=stamp,
                position=[float(v) for v in msg.position],
                velocity=[float(v) for v in msg.velocity]))

    def receive_timing(self, msg):
        try:
            timing = json.loads(msg.data)
            stamp = float(timing['source_stamp_s'])
            received, started, published = [float(timing[k]) for k in (
                'rgb_received_monotonic_s', 'inference_started_monotonic_s', 'depth_published_monotonic_s')]
            if not (np.isfinite([stamp, received, started, published]).all()
                    and 0 <= received <= started <= published):
                raise ValueError('invalid timing order')
            with self.lock:
                self.input_timings[stamp] = timing
                while len(self.input_timings) > 128:
                    self.input_timings.pop(next(iter(self.input_timings)))
        except (ValueError, KeyError, TypeError):
            self.counts['invalid_timing_message'] += 1

    def receive_rgb(self, msg):
        stamp = self.node.image_timestamp_s(msg)
        with self.lock:
            if self.rgb and stamp <= self.rgb[-1][0]:
                self.counts['nonmonotonic_rgb'] += 1
                return
            self.rgb.append((stamp, msg))
            reference = next((d for s, d in reversed(self.reference) if abs(s-stamp) < 1e-6), None)
        if self.node.spatial_depth_source == 'ground_truth_debug' and reference is not None:
            self.submit(stamp, reference)

    def receive_reference(self, stamp, depth):
        with self.lock:
            self.reference.append((stamp, depth))
            has_rgb = any(abs(s-stamp) < 1e-6 for s, _ in self.rgb)
        if self.node.spatial_depth_source == 'ground_truth_debug' and has_rgb:
            self.submit(stamp, depth)

    def submit(self, stamp, depth):
        if self.mailbox.submit(stamp, (depth, time.perf_counter())):
            self.wake.set()

    def cache_command(self, msg):
        plan = getattr(self.node, 'spatial_current_plan', None)
        source = None if plan is None else plan.diagnostics.get('observation_source_stamp_s')
        with self.lock:
            self.command = copy.deepcopy(msg), time.perf_counter()
            self.command_source_stamp = source

    def preflight_ready(self):
        now = self.node.get_clock().now().nanoseconds * 1e-9
        stamp = self.mailbox.last_submitted
        return (not self.fault and stamp is not None
                and 0 <= now-stamp <= self.node.monocular_depth_max_age_s
                and self.history.at(stamp) is not None)

    def loop(self):
        while not self.stop.is_set():
            self.wake.wait(.1)
            self.wake.clear()
            item = self.mailbox.take()
            if item is None:
                continue
            stamp, (depth, received_wall) = item
            try:
                now = self.node.get_clock().now().nanoseconds * 1e-9
                if not 0 <= now - stamp <= self.node.monocular_depth_max_age_s:
                    self.counts['stale_or_future_depth'] += 1
                    continue
                pose = self.history.at(stamp)
                with self.lock:
                    rgb = next((m for s, m in reversed(self.rgb) if abs(s - stamp) < 1e-6), None)
                    refs = list(self.reference)
                reference = min(refs, key=lambda s: abs(s[0] - stamp)) if refs else None
                if (pose is None or rgb is None or reference is None
                        or abs(reference[0] - stamp) > self.node.depth_gt_max_age_s):
                    self.counts['missing_pose_rgb_or_reference'] += 1
                    continue
                ref_pose = self.history.at(reference[0])
                if ref_pose is None:
                    self.counts['missing_reference_pose'] += 1
                    continue
                bgr = self.node.bridge.imgmsg_to_cv2(rgb, desired_encoding='bgr8')
                depth = cv2.resize(depth, self.mapping_size, interpolation=cv2.INTER_NEAREST)
                reference_depth = cv2.resize(reference[1], self.mapping_size, interpolation=cv2.INTER_NEAREST)
                reference_transform = camera_to_ned_transform(
                    *ref_pose, self.node.camera_translation_body_m,
                    self.node.camera_rotation_body_from_optical)
                sample = dict(depth=depth, position=pose[0], attitude=pose[1],
                    reference_transform=reference_transform,
                    reference_position=ref_pose[0], source_stamp_s=stamp,
                    age_s=now - stamp, reference_stamp_s=reference[0],
                    received_wall=received_wall,
                    pose_timestamp_policy='ros_receipt_interpolated_not_px4_sample_time')
                before = self.node.spatial_frames_integrated
                self.node.processar_mapeamento_espacial(
                    bgr, reference_depth, stamp, rgb.width, rgb.height, observation=sample)
                if self.node.spatial_frames_integrated > before:
                    self.last_source_stamp = stamp
                    self.last_mapping_wall = time.perf_counter()
                    self.counts['unique_integrations'] += 1
                    self.timings.append(dict(stage='integration_complete', source_stamp_s=stamp,
                        receive_to_complete_ms=(self.last_mapping_wall - received_wall) * 1000))
            except Exception as error:
                self.fault = str(error)
                self.counts['worker_errors'] += 1
                self.node.get_logger().error(f'Spatial worker stopped advancing: {error}')

    def heartbeat(self):
        """Bounded work, no map/recorder locks; stale navigation means braking."""
        node = self.node
        if not node.spatial_execute_path or self.stop.is_set():
            return
        wall = time.perf_counter()
        now = node.get_clock().now().nanoseconds * 1e-9
        with self.lock:
            command = self.command
            source = getattr(self, 'command_source_stamp', None)
            timing = getattr(self, 'input_timings', {}).get(source)
        node.publish_offboard_control_mode()
        if command is None or node.current_z is None:
            return
        msg, issued = command
        msg = copy.deepcopy(msg)
        stale_map = node.spatial_takeoff_complete and (
            self.last_source_stamp is None or now < self.last_source_stamp
            or now - self.last_source_stamp > node.monocular_depth_max_age_s
            or self.last_mapping_wall is None or wall - self.last_mapping_wall > 1.5)
        braking = bool(self.fault or wall - issued > .25 or stale_map)
        if braking:
            altitude = msg.position[2] if np.isfinite(msg.position[2]) else node.current_z
            msg.position = [float('nan'), float('nan'), float(altitude)]
            msg.velocity = [0., 0., float('nan')]
            self.counts['watchdog_brake_commands'] += 1
        msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
        node.trajectory_setpoint_publisher.publish(msg)
        gap = None if self.last_heartbeat is None else (wall - self.last_heartbeat) * 1000
        first = not braking and timing is not None and source not in self.published_source_timings
        if first:
            self.published_source_timings.add(source)
        self.timings.append(dict(stage='setpoint_published', monotonic_s=wall,
            heartbeat_gap_ms=gap, source_stamp_s=self.last_source_stamp,
            command_observation_stamp_s=source,
            rgb_receive_to_first_plan_command_ms=(None if not first else
                (wall-timing['rgb_received_monotonic_s'])*1000),
            map_age_s=None if self.last_source_stamp is None else now-self.last_source_stamp))
        self.last_heartbeat = wall

    def close(self):
        self.timer.cancel()
        self.stop.set()
        self.wake.set()
        self.worker.join(timeout=10)
        if self.worker.is_alive():
            raise RuntimeError('mapping worker did not stop; maps must not be closed concurrently')
        return dict(counts=dict(self.counts), replaced=self.mailbox.replaced,
                    duplicate_depth=self.mailbox.duplicates, fault=self.fault,
                    pose_timestamp_policy='ros_receipt_interpolated_not_px4_sample_time',
                    monotonic_clock=time.get_clock_info('perf_counter').implementation,
                    latency_scope='RGB receipt to ROS command publication; no PX4 receive acknowledgement',
                    timings=list(self.timings), trajectory=list(self.trajectory))
