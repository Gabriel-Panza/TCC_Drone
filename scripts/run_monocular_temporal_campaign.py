#!/usr/bin/env python3
"""Qualify the frozen temporal experiment before staged 1/3/10 SITL runs."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MODEL_REPORT = ROOT / 'models/monocular_validation_v19_final_independent.json'
FROZEN_FILES = [
    'config/monocular_temporal_campaign.json', 'config/spatial_monocular_temporal.yaml',
    'config/monocular_depth_temporal.yaml', 'drone_controller.py', 'main.py',
    'monocular_depth_node.py', 'spatial_mapping/temporal.py',
    'spatial_mapping/ros_pipeline.py', 'spatial_mapping/navigation.py',
    'spatial_mapping/occupancy.py', 'spatial_mapping/depth_model.py',
    'spatial_mapping/execution.py', 'spatial_mapping/geometry.py',
    'scripts/run_spatial_battery.sh', 'scripts/run_monocular_temporal_campaign.py',
    'estudos_e_analises/avaliar_mapeamento_monocular_offline.py',
    'estudos_e_analises/avaliar_profundidade_monocular.py',
]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(bundle):
    return {p: digest(ROOT / p) for p in FROZEN_FILES + [bundle['model']]}


def verify(report, model, controller, estimator):
    result = json.loads(report.read_text())
    bundle = json.loads((ROOT / 'config/monocular_temporal_campaign.json').read_text())
    if (result.get('qualification', {}).get('passed') is not True
            or result['fingerprint'] != fingerprint(bundle)
            or Path(model).resolve() != (ROOT / bundle['model']).resolve()
            or Path(controller).resolve() != (ROOT / bundle['controller_config']).resolve()
            or Path(estimator).resolve() != (ROOT / bundle['estimator_config']).resolve()):
        raise ValueError('bundle unqualified, modified or different from runtime configuration')
    for key, expected in bundle['px4_speed'].items():
        if float(os.environ.get('SPATIAL_'+key, 'nan')) != expected:
            raise ValueError('PX4 speed limits differ from frozen experiment')


def run_logged(command, log):
    with log.open('w') as stream:
        subprocess.run(list(map(str, command)), cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-sitl', action='store_true')
    parser.add_argument('--verify', type=Path)
    parser.add_argument('--model')
    parser.add_argument('--controller')
    parser.add_argument('--estimator')
    args = parser.parse_args()
    if args.verify:
        verify(args.verify, args.model, args.controller, args.estimator)
        return 0
    bundle = json.loads((ROOT / 'config/monocular_temporal_campaign.json').read_text())
    output = ROOT / 'logs/monocular_temporal_campaign' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    runtime = ROOT / 'models/runtime_env/bin/python'
    model = ROOT / bundle['model']
    if digest(model) != bundle['model_sha256']:
        raise ValueError('model SHA differs from frozen experiment')
    frozen = fingerprint(bundle)
    params = yaml.safe_load((ROOT / bundle['controller_config']).read_text())['drone_offboard_node']['ros__parameters']
    estimator = yaml.safe_load((ROOT / bundle['estimator_config']).read_text())['monocular_metric_depth_node']['ros__parameters']
    historical_pixel = json.loads(MODEL_REPORT.read_text())
    pixel = output / 'pixel.json'
    print(f'Campaign: {output}', flush=True)
    run_logged([runtime, ROOT/'estudos_e_analises/avaliar_profundidade_monocular.py',
        '--model', model, '--calibration-run', historical_pixel['calibration_run'],
        '--validation-run', historical_pixel['validation_run'], '--frames', '24',
        '--output', pixel], output/'pixel.log')
    pixel_result = json.loads(pixel.read_text())
    rows = []
    for dataset in bundle['datasets']:
        run = ROOT / dataset
        report = output / (run.name + '.json')
        command = [runtime, ROOT/'estudos_e_analises/avaliar_mapeamento_monocular_offline.py',
            run, '--model', model, '--output', report,
            '--prediction-cache-dir', ROOT/'logs/monocular_temporal_campaign/prediction_cache'/bundle['model_sha256']/run.name,
            '--replay-stride', params['spatial_depth_stride'],
            '--edge-stride', params['spatial_depth_edge_stride'],
            '--edge-relative-threshold', params['spatial_depth_edge_relative_threshold'],
            '--depth-output-scale', estimator['post_depth_scale'],
            '--conservative-depth-shift-m', estimator['conservative_depth_shift_m'],
            '--free-space-margin-m', params['spatial_depth_free_space_margin_m'],
            '--free-space-margin-ratio', params['spatial_depth_free_space_margin_ratio'],
            '--free-space-margin-max-m', params['spatial_depth_free_space_margin_max_m'],
            '--occupied-uncertainty-m', params['spatial_depth_occupied_uncertainty_m'],
            '--obstacle-vertical-band-m', params['spatial_obstacle_vertical_band_m'],
            '--vertical-clearance-m', params['spatial_vertical_clearance_m'],
            '--reference-obstacle-vertical-band-m', params['spatial_reference_obstacle_vertical_band_m'],
            '--reference-vertical-clearance-m', params['spatial_reference_vertical_clearance_m'],
            '--occupied-observations-required', params['spatial_occupied_observations_required'],
            '--free-observations-required', params['spatial_free_observations_required'],
            '--minimum-executable-path-m', params['spatial_min_executable_path_m'],
            '--lock-path-altitude-to-goal', '--body-altitude-envelope']
        run_logged(command, output/(run.name+'.log'))
        result = json.loads(report.read_text())
        planning = result['planning']
        row = dict(dataset=dataset, report=str(report), frames=result['frames_integrated'],
            success_rate=planning['success_rate'],
            collisions=planning['successes']-planning['collision_free_successes'],
            false_free_rate=result['map']['false_free_rate'],
            checks=result['qualification']['checks'], passed=result['qualification']['passed'])
        rows.append(row)
        print(json.dumps(row), flush=True)
    passed = pixel_result['qualification']['passed'] and all(row['passed'] for row in rows)
    if fingerprint(bundle) != frozen:
        raise ValueError('experiment changed during qualification')
    result = dict(schema='monocular_temporal_campaign_v1', model_sha256=bundle['model_sha256'],
        fingerprint=frozen, pixel_report=str(pixel), spatial_results=rows,
        qualification=dict(passed=passed),
        limitation='Replay does not validate asynchronous timing, transport, braking or flight tracking.',
        sitl='not_started' if passed else 'blocked_by_offline_gate')
    overall = output / 'campaign.json'
    overall.write_text(json.dumps(result, indent=2)+'\n')
    print(f'Qualification passed={passed}; report={overall}', flush=True)
    if not passed:
        return 2
    if args.run_sitl:
        env = dict(os.environ, MONOCULAR_MODEL_PATH=str(model),
            MONOCULAR_PARAMS_FILE=str(ROOT/bundle['estimator_config']),
            SPATIAL_CONFIG=str(ROOT/bundle['controller_config']),
            MONOCULAR_VALIDATION_REPORT=str(pixel), MONOCULAR_MAPPING_REPORT=str(overall),
            MONOCULAR_BUNDLE_REPORT=str(overall), BATTERY_CONTINUE_SAFE_ABORTS='0',
            SPATIAL_MPC_XY_VEL_MAX=str(bundle['px4_speed']['MPC_XY_VEL_MAX']),
            SPATIAL_MPC_XY_CRUISE=str(bundle['px4_speed']['MPC_XY_CRUISE']))
        result['stages'] = []
        for count in bundle['stages']:
            before = set((ROOT/'logs/spatial_battery').glob('*/summary.tsv'))
            # Never bypass the ordinary runner gate and never retry a failed stage.
            completed = subprocess.run(['bash', str(ROOT/'scripts/run_spatial_battery.sh'),
                'monocular_topic', str(count)], cwd=ROOT, env=env, check=False)
            new_summaries = set((ROOT/'logs/spatial_battery').glob('*/summary.tsv')) - before
            audits = []
            if completed.returncode == 0 and len(new_summaries) == 1:
                with next(iter(new_summaries)).open() as stream:
                    runs = list(csv.DictReader(stream, delimiter='\t'))
                for row in runs:
                    if row['mission_complete'].lower() != 'true':
                        audits.append(dict(passed=False, reason='incomplete mission'))
                    else:
                        audits.append(audit_actual_trajectory(Path(row['dataset'])))
            stage_passed = completed.returncode == 0 and len(audits) == count and all(a['passed'] for a in audits)
            result['stages'].append(dict(requested_runs=count, exit_code=completed.returncode,
                                        passed=stage_passed, audits=audits))
            result['sitl'] = 'guarded_stage_completed' if stage_passed else 'stage_failed'
            overall.write_text(json.dumps(result, indent=2)+'\n')
            if not stage_passed:
                return completed.returncode or 2
    return 0


def audit_actual_trajectory(run):
    """Check the airborne body path against final inflated GT occupancy, not just A* plans."""
    import numpy as np
    from spatial_mapping.navigation import SpatialNavigationConfig, SpatialNavigator
    data = json.loads((run/'temporal_runtime.json').read_text())
    manifest = json.loads((run/'manifest.json').read_text())
    nav = SpatialNavigator(SpatialNavigationConfig(**manifest['metadata']['reference_spatial_config']))
    with np.load(run/'reference_map.npz', allow_pickle=False) as saved:
        nav.grid._log_odds = dict(zip(map(tuple, saved['voxels']), map(float, saved['log_odds'])))
        nav.grid._confirmed_occupied = {v for v, value in nav.grid._log_odds.items()
                                        if value >= float(saved['occupied_threshold'])}
    points = [p['position'] for p in data['trajectory']]
    blocked = nav._inflated_obstacles()
    traversals = [set(nav.grid.segment_voxels(a, b)) for a, b in zip(points, points[1:])]
    collisions = sum(bool(segment & blocked) for segment in traversals)
    known_free = nav.grid.free_voxels()
    unobserved = set().union(*traversals) - known_free if traversals else set()
    velocities = np.asarray([p['velocity'] for p in data['trajectory']], dtype=float).reshape(-1, 3)
    max_speed = float(np.max(np.linalg.norm(velocities[:, :2], axis=1))) if len(velocities) else float('inf')
    gaps = [t['heartbeat_gap_ms'] for t in data['timings']
            if t['stage'] == 'setpoint_published' and t['heartbeat_gap_ms'] is not None]
    checks = dict(trajectory_present=len(points) >= 2,
                  collision_free=collisions == 0,
                  reference_centerline_observed=not unobserved,
                  velocity_samples_valid=bool(len(velocities)) and bool(np.isfinite(velocities).all()),
                  horizontal_speed=max_speed <= .825,
                  worker_healthy=data['fault'] is None,
                  heartbeat_present=bool(gaps),
                  heartbeat_max_gap=bool(gaps) and max(gaps) <= 200.)
    return dict(passed=all(checks.values()), checks=checks,
                collision_segments=collisions, max_heartbeat_gap_ms=max(gaps) if gaps else None,
                unobserved_centerline_voxels=len(unobserved), max_horizontal_speed_m_s=max_speed,
                scope='guarded SITL; final discrete inflated GT map; transport timing not hardware-exact')


if __name__ == '__main__':
    raise SystemExit(main())
