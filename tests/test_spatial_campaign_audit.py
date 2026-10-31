import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from spatial_mapping.navigation import SpatialNavigationConfig
from scripts.run_monocular_temporal_campaign import audit_actual_trajectory


class ActualTrajectoryAuditTest(unittest.TestCase):
    def run_audit(self, occupied=False, gap=40., speed=0., unknown=False):
        with TemporaryDirectory() as folder:
            run = Path(folder)
            config = SpatialNavigationConfig(voxel_resolution_m=1., drone_clearance_radius_m=0., drone_vertical_clearance_m=.01)
            (run/'manifest.json').write_text(json.dumps(dict(metadata=dict(reference_spatial_config=asdict(config)))))
            runtime = dict(fault=None, trajectory=[
                dict(position=[.5,.5,.5], velocity=[speed,0,0]),
                dict(position=[2.5,.5,.5], velocity=[speed,0,0])],
                timings=[dict(stage='setpoint_published', heartbeat_gap_ms=gap)])
            (run/'temporal_runtime.json').write_text(json.dumps(runtime))
            np.savez(run/'reference_map.npz', voxels=np.array([[0,0,0],[1,0,0],[2,0,0]]),
                     log_odds=np.array([-1., 1. if occupied else (0. if unknown else -1.), -1.]),
                     occupied_threshold=.6)
            return audit_actual_trajectory(run)

    def test_safe_observed_trajectory_passes(self):
        self.assertTrue(self.run_audit()['passed'])

    def test_actual_collision_fails_even_without_a_plan_collision_record(self):
        self.assertFalse(self.run_audit(occupied=True)['checks']['collision_free'])

    def test_unknown_coverage_stalled_heartbeat_and_overspeed_fail(self):
        self.assertFalse(self.run_audit(unknown=True)['passed'])
        self.assertFalse(self.run_audit(gap=300.)['passed'])
        self.assertFalse(self.run_audit(speed=1.)['passed'])
