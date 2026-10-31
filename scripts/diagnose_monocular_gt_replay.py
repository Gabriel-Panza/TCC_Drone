#!/usr/bin/env python3
"""Causal GT controls for the temporal campaign; never starts or qualifies flight."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = ROOT / 'estudos_e_analises/avaliar_mapeamento_monocular_offline.py'


def command_for(dataset, output, params, estimator):
    command = [sys.executable, str(EVALUATOR), str(ROOT / dataset),
               '--diagnostic-reference-depth', '--output', str(output),
               '--lock-path-altitude-to-goal', '--body-altitude-envelope']
    mapping = {
        'replay-stride': 'spatial_depth_stride',
        'edge-stride': 'spatial_depth_edge_stride',
        'edge-relative-threshold': 'spatial_depth_edge_relative_threshold',
        'free-space-margin-m': 'spatial_depth_free_space_margin_m',
        'free-space-margin-ratio': 'spatial_depth_free_space_margin_ratio',
        'free-space-margin-max-m': 'spatial_depth_free_space_margin_max_m',
        'occupied-uncertainty-m': 'spatial_depth_occupied_uncertainty_m',
        'obstacle-vertical-band-m': 'spatial_obstacle_vertical_band_m',
        'vertical-clearance-m': 'spatial_vertical_clearance_m',
        'reference-obstacle-vertical-band-m': 'spatial_reference_obstacle_vertical_band_m',
        'reference-vertical-clearance-m': 'spatial_reference_vertical_clearance_m',
        'occupied-observations-required': 'spatial_occupied_observations_required',
        'free-observations-required': 'spatial_free_observations_required',
        'minimum-executable-path-m': 'spatial_min_executable_path_m',
    }
    for flag, key in mapping.items():
        command.extend(['--' + flag, str(params[key])])
    command.extend(['--depth-output-scale', str(estimator['post_depth_scale']),
                    '--conservative-depth-shift-m', str(estimator['conservative_depth_shift_m'])])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--all-datasets', action='store_true',
                        help='Also run the strict GT substitution on the other four datasets.')
    args = parser.parse_args()
    bundle = json.loads((ROOT / 'config/monocular_temporal_campaign.json').read_text())
    params = yaml.safe_load((ROOT / bundle['controller_config']).read_text())['drone_offboard_node']['ros__parameters']
    estimator = yaml.safe_load((ROOT / bundle['estimator_config']).read_text())['monocular_metric_depth_node']['ros__parameters']
    output = ROOT / 'logs/monocular_gt_diagnostic' / datetime.now().strftime('%Y%m%d_%H%M%S')
    output.mkdir(parents=True, exist_ok=False)
    snapshot_paths = [bundle['controller_config'], bundle['estimator_config'],
                      'config/monocular_temporal_campaign.json',
                      'estudos_e_analises/avaliar_mapeamento_monocular_offline.py',
                      'spatial_mapping/navigation.py', 'spatial_mapping/occupancy.py',
                      'spatial_mapping/geometry.py', 'scripts/diagnose_monocular_gt_replay.py']
    result = dict(diagnostic_only=True, qualifies_monocular_flight=False,
                  fingerprint={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in snapshot_paths},
                  cases=[], note='Recorded trajectories and plan requests are fixed; no flight or independent test.')
    dataset = bundle['datasets'][0]
    raw = dict(estimator, post_depth_scale=1., conservative_depth_shift_m=0.)
    matched = dict(params)
    matched.update(spatial_depth_free_space_margin_m=0., spatial_depth_free_space_margin_ratio=0.,
                   spatial_depth_occupied_uncertainty_m=0., spatial_occupied_observations_required=1,
                   spatial_obstacle_vertical_band_m=params['spatial_reference_obstacle_vertical_band_m'],
                   spatial_vertical_clearance_m=params['spatial_reference_vertical_clearance_m'])
    cases = [('gt_same_pipeline', dataset, params, estimator),
             ('gt_raw_same_map', dataset, params, raw),
             ('gt_raw_matched_reference', dataset, matched, raw)]
    if args.all_datasets:
        cases += [('gt_same_pipeline', d, params, estimator) for d in bundle['datasets'][1:]]
    print('Diagnostic: ' + str(output), flush=True)
    for label, dataset, config, depth_config in cases:
        stem = Path(dataset).name + '_' + label
        report = output / (stem + '.json')
        command = command_for(dataset, report, config, depth_config)
        with (output / (stem + '.log')).open('w') as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        data = json.loads(report.read_text())
        planning = data['planning']
        row = dict(case=label, dataset=dataset, report=str(report), command=command,
                   frames=data['frames_integrated'], success_rate=planning['success_rate'],
                   collisions=planning['successes']-planning['collision_free_successes'],
                   false_free_rate=data['map']['false_free_rate'],
                   spatial_checks=data['qualification']['checks'])
        assert data['diagnostic_only'] and data['qualification']['passed'] is False
        result['cases'].append(row)
        (output / 'diagnostic.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'command'}), flush=True)
    print('No flight authorized; summary: ' + str(output / 'diagnostic.json'), flush=True)


if __name__ == '__main__':
    main()
