"""A GT diagnostic cannot authorize monocular flight, even with perfect checks."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from estudos_e_analises import avaliar_mapeamento_monocular_offline as evaluator


class GroundTruthDiagnosticTest(unittest.TestCase):
    def test_gt_cannot_qualify_even_when_all_spatial_checks_pass(self):
        with TemporaryDirectory() as folder:
            run = Path(folder)
            (run/'manifest.json').write_text('{}')
            np.savez(run/'frame.npz', rgb_bgr=np.zeros((2,2,3), dtype=np.uint8),
                     reference_depth_m=np.full((2,2), 5.), camera_to_ned=np.eye(4),
                     intrinsics=np.array([1.,1.,0.,0.]))
            events = [dict(event='frame', file='frame.npz') for _ in range(100)]
            events.append(dict(event='plan', map_kind='estimated', plan=dict(
                diagnostics=dict(current_position_ned_m=[0.,0.,0.]), requested_goal_ned_m=[2.,0.,0.])))
            (run/'events.jsonl').write_text('\n'.join(map(json.dumps, events)))
            navigator = Mock()
            navigator.grid.interval_hypotheses = SimpleNamespace(possible=set(),cleared_by_multiview=0)
            navigator.grid.segment_voxels.return_value = [(0,0,0), (1,0,0)]
            navigator.grid.free_voxels.return_value = {(0,0,0), (1,0,0)}
            navigator.grid.occupied_voxels.return_value = set()
            navigator.grid.pending_occupied_voxels.return_value = set()
            navigator._inflated_obstacles.return_value = set()
            navigator.path_is_safe.return_value = True
            navigator.plan.return_value = SimpleNamespace(success=True,
                waypoints_ned_m=[[2.,0.,0.]], path_length_m=2., diagnostics={}, reason='success')
            output = run/'result.json'
            with patch('sys.argv', ['replay', str(run), '--diagnostic-reference-depth', '--output', str(output)]), \
                 patch.object(evaluator, 'SpatialNavigator', return_value=navigator), \
                 patch.object(evaluator, 'MetricDepthOnnx') as model, \
                 patch.object(evaluator, 'obstacle_proximity', return_value={}), \
                 patch.object(evaluator, 'occupancy_metrics', return_value={'false_free_rate': 0.}), \
                 redirect_stdout(io.StringIO()):
                evaluator.main()
            data = json.loads(output.read_text())
            self.assertTrue(all(data['qualification']['checks'].values()))
            self.assertFalse(data['qualification']['passed'])
            self.assertTrue(data['diagnostic_only'])
            self.assertIsNone(data['model_sha256'])
            model.assert_not_called()

    def test_monocular_mode_still_requires_model(self):
        with patch('sys.argv', ['replay', '/tmp/unused']), redirect_stdout(io.StringIO()), \
             patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as error:
            evaluator.main()
        self.assertEqual(error.exception.code, 2)

    def test_gt_rejects_monocular_prediction_cache(self):
        with patch('sys.argv', ['replay', '/tmp/unused', '--diagnostic-reference-depth',
                               '--prediction-cache-dir', '/tmp/unused-cache']), \
             patch('sys.stderr', io.StringIO()), self.assertRaises(SystemExit) as error:
            evaluator.main()
        self.assertEqual(error.exception.code, 2)
