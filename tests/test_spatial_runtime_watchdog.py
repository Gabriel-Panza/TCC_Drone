"""Exercise the real heartbeat method without starting ROS or a simulator."""
import ast
import copy
from pathlib import Path
import threading
import time
import unittest
from collections import deque, Counter
from types import SimpleNamespace

import numpy as np


class HeartbeatWatchdogTest(unittest.TestCase):
    def setUp(self):
        source = Path('spatial_mapping/ros_pipeline.py').read_text()
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        namespace = dict(copy=copy, time=time, np=np)
        exec(compile(ast.Module(body=[cls], type_ignores=[]), 'runtime', 'exec'), namespace)
        self.runtime = namespace['SpatialObservationRuntime'].__new__(namespace['SpatialObservationRuntime'])
        runtime = self.runtime
        self.published = []
        runtime.node = SimpleNamespace(
            spatial_execute_path=True, current_z=-1.5, spatial_takeoff_complete=True,
            monocular_depth_max_age_s=.6,
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=10_000_000_000)),
            publish_offboard_control_mode=lambda: None,
            trajectory_setpoint_publisher=SimpleNamespace(publish=lambda msg: self.published.append(msg)))
        runtime.stop = threading.Event()
        runtime.lock = threading.Lock()
        runtime.command = (SimpleNamespace(position=[2., 3., -1.5], velocity=[float('nan')]*3), time.perf_counter())
        runtime.last_source_stamp = 9.9
        runtime.last_mapping_wall = time.perf_counter()
        runtime.last_heartbeat = None
        runtime.published_source_timings = set()
        runtime.counts, runtime.timings = Counter(), deque()
        runtime.fault = None

    def test_fresh_command_is_republished_without_map_lock(self):
        self.runtime.heartbeat()
        self.assertEqual(self.published[-1].position, [2., 3., -1.5])

    def test_stalled_navigation_requests_zero_horizontal_velocity(self):
        msg, _ = self.runtime.command
        self.runtime.command = msg, time.perf_counter() - 1
        self.runtime.heartbeat()
        self.assertTrue(np.isnan(self.published[-1].position[0]))
        self.assertEqual(self.published[-1].velocity[:2], [0., 0.])

    def test_stale_future_map_and_worker_fault_brake(self):
        for stamp, fault in [(8., None), (11., None), (9.9, 'failed')]:
            self.runtime.last_source_stamp, self.runtime.fault = stamp, fault
            self.runtime.heartbeat()
            self.assertEqual(self.published[-1].velocity[:2], [0., 0.])

    def test_disabled_execution_never_publishes(self):
        self.runtime.node.spatial_execute_path = False
        self.runtime.heartbeat()
        self.assertEqual(self.published, [])
