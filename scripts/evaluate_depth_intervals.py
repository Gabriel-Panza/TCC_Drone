#!/usr/bin/env python3
"""Fit once on a separate sequence, freeze, then compare point/interval replay."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spatial_mapping.depth_model import MetricDepthOnnx
from spatial_mapping.depth_intervals import DepthIntervalProfile
from scripts.diagnose_monocular_gt_replay import command_for


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def calibrate(run, cache, model_path, model_sha):
    events = [json.loads(s) for s in (run/'events.jsonl').read_text().splitlines() if s.strip()]
    frames = [e['file'] for e in events if e.get('event')=='frame' and e.get('file')]
    selected = [frames[i] for i in np.unique(np.linspace(0,len(frames)-1,min(32,len(frames)),dtype=int))]
    model = None
    samples, frame_rows = [], []
    for name in selected:
        with np.load(run/name) as frame:
            rgb = frame['rgb_bgr']
            gt = frame['reference_depth_m'].astype(float)
        prediction_file = cache/(Path(name).stem+'.npy')
        if prediction_file.exists():
            pred = np.load(prediction_file).astype(float)
        else:
            if model is None:
                model = MetricDepthOnnx(model_path,input_width=686,input_height=518,
                    backend='onnxruntime',max_depth_m=80.,input_aspect_tolerance=.03)
            pred = model.predict(rgb)
            cache.mkdir(parents=True,exist_ok=True)
            np.save(prediction_file,pred.astype(np.float32))
        if pred.shape != gt.shape:
            pred = cv2.resize(pred,(gt.shape[1],gt.shape[0]),interpolation=cv2.INTER_LINEAR)
        # Fixed sampling and quantiles declared before reading evaluation runs.
        pred, gt = pred[::4,::4], gt[::4,::4]
        valid = np.isfinite(pred)&np.isfinite(gt)&(pred>=.5)&(pred<=30)&(gt>=.5)&(gt<=30)
        p, g = pred[valid], gt[valid]
        samples.append(np.stack([p,g],axis=1))
        frame_rows.append(dict(file=str(run/name),sha256=digest(run/name),pixels=len(p),
            mae_m=float(np.mean(np.abs(g-p))),
            overestimate_gt_by_1m_rate=float(np.mean(p-g>1))))
    pairs = np.concatenate(samples)
    edges = np.array([.5,5.,10.,20.,30.])
    bounds, stats = [], []
    for i in range(4):
        selected = (pairs[:,0]>=edges[i])&((pairs[:,0]<edges[i+1]) if i<3 else (pairs[:,0]<=edges[i+1]))
        values = pairs[selected,1]-pairs[selected,0]
        fallback = len(values)<100
        fitted = pairs[:,1]-pairs[:,0] if fallback else values
        lo, hi = np.quantile(fitted,[.025,.975])
        bounds.append([min(0.,float(lo)),max(0.,float(hi))])
        stats.append(dict(predicted_range_m=edges[i:i+2].tolist(),pixels=len(values),
                          pooled_fallback=fallback,raw_residual_quantiles_m=[float(lo),float(hi)]))
    profile = dict(schema='depth_interval_v1',model_sha256=model_sha,
        calibration_run=str(run),sampled_frames=len(frame_rows),pixel_stride=4,
        quantiles=[.025,.975],predicted_bin_edges_m=edges.tolist(),residual_bounds_m=bounds,
        bins=stats,note='Descriptive empirical pixel intervals; not a probabilistic safety guarantee.')
    fitted = DepthIntervalProfile(profile)
    lower, upper = fitted.bounds(pairs[:,0])
    profile['calibration_coverage'] = float(np.mean((pairs[:,1]>=lower)&(pairs[:,1]<=upper)))
    return profile, sorted(frame_rows,key=lambda r:r['mae_m'],reverse=True)


def main():
    bundle = json.loads((ROOT/'config/monocular_temporal_campaign.json').read_text())
    old_pixel = json.loads((ROOT/'models/monocular_validation_v19_final_independent.json').read_text())
    calibration = Path(old_pixel['calibration_run'])
    evaluations = [ROOT/p for p in bundle['datasets'][3:]]
    if calibration.resolve() in [p.resolve() for p in evaluations]:
        raise ValueError('calibration/evaluation overlap')
    model = ROOT/bundle['model']
    if digest(model)!=bundle['model_sha256']:
        raise ValueError('unexpected model')
    out = ROOT/'logs/monocular_depth_intervals'/datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True,exist_ok=False)
    cache = ROOT/'logs/monocular_temporal_campaign/prediction_cache'/bundle['model_sha256']
    split = dict(calibration=str(calibration),evaluation=list(map(str,evaluations)),
                 sampled_calibration_frames=32,quantiles=[.025,.975],bins_m=[.5,5,10,20,30],
                 historical_development_data=True,flight_authorized=False)
    (out/'split.json').write_text(json.dumps(split,indent=2)+'\n')
    print('Experiment: '+str(out),flush=True)
    profile, hard_frames = calibrate(calibration,cache/calibration.name,model,bundle['model_sha256'])
    profile_file = out/'frozen_profile.json'
    profile_file.write_text(json.dumps(profile,indent=2)+'\n')
    (out/'calibration_hard_frames.json').write_text(json.dumps(hard_frames,indent=2)+'\n')
    frozen = digest(profile_file)
    files = ['spatial_mapping/depth_intervals.py','spatial_mapping/navigation.py',
             'spatial_mapping/occupancy.py','scripts/evaluate_depth_intervals.py',
             'estudos_e_analises/avaliar_mapeamento_monocular_offline.py']
    fingerprint = {p:digest(ROOT/p) for p in files}
    result = dict(split=split,profile_sha256=frozen,fingerprint=fingerprint,cases=[])
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Frozen profile: '+json.dumps(profile['residual_bounds_m']),flush=True)
    params = yaml.safe_load((ROOT/bundle['controller_config']).read_text())['drone_offboard_node']['ros__parameters']
    params.update(spatial_obstacle_vertical_band_m=.2,spatial_vertical_clearance_m=.2,
                  spatial_depth_free_space_margin_m=0.,spatial_depth_free_space_margin_ratio=0.,
                  spatial_depth_occupied_uncertainty_m=0.,spatial_occupied_observations_required=2)
    for run in evaluations:
        for kind in ['point','interval']:
            report = out/(run.name+'_'+kind+'.json')
            command = command_for(str(run),report,params,dict(post_depth_scale=1.,conservative_depth_shift_m=0.))
            command.remove('--diagnostic-reference-depth')
            command += ['--model',str(model),'--prediction-cache-dir',str(cache/run.name),
                        '--protect-pending-obstacles','--allow-known-free-initial-escape']
            if kind=='interval':
                command += ['--depth-interval-profile',str(profile_file)]
            with (out/(run.name+'_'+kind+'.log')).open('w') as stream:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            data = json.loads(report.read_text())
            row = dict(dataset=str(run),kind=kind,command=command,frames=data['frames_integrated'],
                success_rate=data['planning']['success_rate'],false_free_rate=data['map']['false_free_rate'],
                collisions=data['planning']['successes']-data['planning']['collision_free_successes'],
                checks=data['qualification']['checks'],passed=data['qualification']['passed'])
            result['cases'].append(row)
            (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
    if digest(profile_file)!=frozen or {p:digest(ROOT/p) for p in files}!=fingerprint:
        raise ValueError('frozen experiment changed')
    print('Complete; no flight authorized.',flush=True)


if __name__=='__main__':
    main()
