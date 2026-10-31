#!/usr/bin/env python3
"""Fresh native captures and paired 320/640 spatial replay; never starts flight."""
import argparse
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial_mapping.camera_sampling import matched_nearest_pair
from scripts.diagnose_monocular_gt_replay import command_for
from scripts.evaluate_depth_intervals import calibrate

SOURCES=[
    "scripts/run_monocular_resolution_640.py","scripts/collect_camera_consistency.py",
    "spatial_mapping/camera_sampling.py","spatial_mapping/depth_model.py",
    "spatial_mapping/geometry.py","spatial_mapping/navigation.py","spatial_mapping/occupancy.py",
    "spatial_mapping/depth_intervals.py","spatial_mapping/metrics.py",
    "estudos_e_analises/avaliar_mapeamento_monocular_offline.py",
    "scripts/evaluate_depth_intervals.py","scripts/diagnose_monocular_gt_replay.py",
    "config/monocular_temporal_campaign.json","config/spatial_monocular_temporal.yaml",
]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    path.write_text(json.dumps(data,indent=2)+"\n")

def sampling_intrinsics(k,native_width,output_width,legacy=False):
    step=native_width/output_width
    if step!=int(step):raise ValueError("Only integer decimation supported by this replay")
    offset=0. if legacy else .5
    return np.array([k[0,0]/step,k[1,1]/step,(k[0,2]-offset)/step,(k[1,2]-offset)/step])

def prepare_capture(capture,legacy,out,variants_to_prepare=("legacy320","aligned640")):
    if not variants_to_prepare or not set(variants_to_prepare)<= {"legacy320","aligned640"}:
        raise ValueError("Unknown data preparation variant")
    summary=json.loads((capture/"summary.json").read_text())
    if not summary.get("completed") or not summary.get("render_resources_complete"):
        raise ValueError("Rejected incomplete/resource-invalid capture")
    pairs=json.loads((capture/"pairs.json").read_text())
    if any(not r.get("ros_"+key+"_payload_identical",False) for r in pairs for key in ["rgb","depth"]):
        raise ValueError("ROS payload differs or was not checked")
    protocol=json.loads((capture/"protocol.json").read_text())
    if Path(protocol["trajectory_run"]).resolve()!=legacy.resolve():
        raise ValueError("Capture belongs to another trajectory")
    if len(pairs)!=len(protocol["trajectory_frames"]):raise ValueError("Need exactly one selected pair per trajectory frame")
    rows={r["trajectory_frame_file"]:r for r in pairs}
    info=json.loads((capture/"camera_info.json").read_text())
    k=np.asarray(info["depth"]["k"]).reshape(3,3)
    if max(abs(np.array(info["rgb"]["k"])-np.array(info["depth"]["k"])))>1e-3:
        raise ValueError("RGB and GT intrinsics differ")
    variants={label:out/label for label in variants_to_prepare}
    manifests={}
    for label,folder in variants.items():
        (folder/"frames").mkdir(parents=True,exist_ok=False)
        manifest=json.loads((legacy/"manifest.json").read_text())
        manifest["frames_saved"]=len(pairs);manifest["complete"]=True
        manifest["metadata"].update(native_capture=str(capture),diagnostic_pose_reenactment=True,
            original_trajectory=str(legacy),data_variant=label,common_reference="matched native GT 640x480",
            no_flight=True,new_training=False)
        save(folder/"manifest.json",manifest)
        manifests[label]=manifest
    events=[json.loads(line) for line in (legacy/"events.jsonl").read_text().splitlines() if line.strip()]
    streams={label:[] for label in variants};hashes={}
    to_ned=np.eye(4);to_ned[:3,:3]=[[0,1,0],[1,0,0],[0,0,-1]]
    for event in events:
        if event.get("event")=="frame" and event.get("file"):
            if event["file"] not in rows:raise ValueError("Missing captured frame")
            row=rows[event["file"]]
            if row["rgb_stamp_ns"]!=row["depth_stamp_ns"]:raise ValueError("Mismatched native stamps")
            with np.load(capture/row["file"]) as f:
                rgb=f["rgb_rgb"];gt=f["reference_depth_m"];pose=to_ned@f["camera_to_world"]
            if (hashlib.sha256(rgb.tobytes()).hexdigest()!=row["rgb_sha256"] or
                    hashlib.sha256(gt.tobytes()).hexdigest()!=row["depth_sha256"]):
                raise ValueError("Native captured arrays differ from recorded hashes")
            hashes[row["file"]]=digest(capture/row["file"])
            common=matched_nearest_pair(rgb,gt,k,(640,480))
            ref_k=sampling_intrinsics(k,gt.shape[1],640)
            for label,folder in variants.items():
                if label=="legacy320":
                    bgr=cv2.cvtColor(cv2.resize(rgb,(320,240),interpolation=cv2.INTER_AREA),cv2.COLOR_RGB2BGR)
                    intrinsics=sampling_intrinsics(k,gt.shape[1],320,legacy=True)
                else:
                    bgr=cv2.cvtColor(common["rgb_rgb"],cv2.COLOR_RGB2BGR)
                    intrinsics=ref_k
                np.savez_compressed(folder/event["file"],rgb_bgr=bgr,
                    reference_depth_m=common["reference_depth_m"],reference_intrinsics=ref_k,
                    estimated_depth_m=np.empty((0,0),dtype=np.float32),
                    camera_to_ned=pose,reference_camera_to_ned=pose,intrinsics=intrinsics,
                    rgb_stamp_ns=row["rgb_stamp_ns"],reference_stamp_ns=row["depth_stamp_ns"])
                current=copy.deepcopy(event)
                current.update(timestamp_s=row["stamp_ns"]*1e-9,original_timestamp_s=event["timestamp_s"],
                    capture_file=row["file"],source="fresh_stationary_pose_reenactment")
                streams[label].append(current)
        else:
            for label in streams:streams[label].append(copy.deepcopy(event))
    for label,folder in variants.items():
        (folder/"events.jsonl").write_text("".join(json.dumps(e)+"\n" for e in streams[label]))
        save(folder/"native_input_sha256.json",hashes)
    return variants

def capture_sequence(run,label,out):
    log=out/(label+"_capture.log")
    command=["/usr/bin/python3",str(ROOT/"scripts/collect_camera_consistency.py"),
        "--scene","baylands","--ros-bridge","--trajectory-run",str(run),"--pairs-per-pose","1"]
    print("Capturing "+label+": "+str(run),flush=True)
    with log.open("w") as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    locations=[line.split("Collection: ",1)[1] for line in log.read_text().splitlines() if line.startswith("Collection: ")]
    if len(locations)!=1:raise ValueError("Capture output missing")
    capture=Path(locations[0])
    print("Captured "+label+": "+str(capture),flush=True)
    return capture,command

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-captures",type=Path,
        help="Resume only exact completed captures listed in a preregistered interrupted experiment.")
    args=parser.parse_args()
    bundle=json.loads((ROOT/"config/monocular_temporal_campaign.json").read_text())
    model=ROOT/bundle["model"]
    if digest(model)!=bundle["model_sha256"]:raise ValueError("Model SHA mismatch")
    out=ROOT/"logs/monocular_resolution_640"/datetime.now().strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True,exist_ok=False)
    calibration=ROOT/"datasets/spatial_mapping/run_20260827_165430_607620"
    evaluations=[ROOT/p for p in bundle["datasets"][3:]]
    datasets=[("calibration",calibration)]+[("evaluation_"+str(i+1),p) for i,p in enumerate(evaluations)]
    frozen={name:digest(ROOT/name) for name in SOURCES}
    protocol=dict(model=str(model),model_sha256=digest(model),datasets={label:str(run) for label,run in datasets},
        trajectory_metadata_sha256={label:{name:digest(run/name) for name in ["manifest.json","events.jsonl"]}
                                    for label,run in datasets},
        cases=["legacy320_point","aligned640_point","aligned640_equal_ray_density_point",
               "legacy320_interval","aligned640_interval","gt640_control"],
        calibration_frames=32,interval_quantiles=[.025,.975],interval_bins_m=[.5,5,10,20,30],
        minimum_frames=100,minimum_success_rate=.8,maximum_false_free_rate=.1,maximum_colliding_plans=0,
        common_reference_resolution=[640,480],common_reference_stride=5,common_reference_edge_stride=2,
        acquisition="Stationary at each historical camera pose; fresh RGB/GT, not continuous motion or flight.",
        evaluation_status="Development data, historical trajectories, same scene; not an independent held-out test.",
        flight_authorized=False,source_sha256=frozen,git_head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip())
    save(out/"preregistered.json",protocol)
    source_dir=out/"source"
    for name in SOURCES:
        target=source_dir/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,target)
    result=dict(protocol=protocol,captures={},profiles={},cases=[],flight="not_started")
    save(out/"summary.json",result)
    print("Experiment: "+str(out),flush=True)
    reuse=json.loads(args.reuse_captures.read_text()) if args.reuse_captures else None
    if reuse and reuse["protocol"]["datasets"]!=protocol["datasets"]:raise ValueError("Resume dataset mismatch")
    prepared={}
    for label,run in datasets:
        if reuse and label in reuse.get("captures",{}):
            capture=Path(reuse["captures"][label]["folder"]);command=None
        else:capture,command=capture_sequence(run,label,out)
        result["captures"][label]=dict(folder=str(capture),command=command)
        save(out/"summary.json",result)
        prepared[label]=prepare_capture(capture,run,out/label)
    # No evaluation replay or inference is opened before both profiles are frozen.
    for variant in ["legacy320","aligned640"]:
        cache=out/"predictions"/variant/"calibration"
        profile,_=calibrate(prepared["calibration"][variant],cache,model,bundle["model_sha256"])
        target=out/(variant+"_frozen_profile.json");save(target,profile)
        result["profiles"][variant]=dict(path=str(target),sha256=digest(target))
    save(out/"summary.json",result)
    params=yaml.safe_load((ROOT/bundle["controller_config"]).read_text())["drone_offboard_node"]["ros__parameters"]
    params.update(spatial_occupied_support_radius_voxels=0)
    for label,original in datasets[1:]:
        for kind in protocol["cases"]:
            variant="legacy320" if kind.startswith("legacy320") else "aligned640"
            p=dict(params)
            if "interval" in kind:
                p.update(spatial_obstacle_vertical_band_m=.2,spatial_vertical_clearance_m=.2,
                    spatial_depth_free_space_margin_m=0.,spatial_depth_free_space_margin_ratio=0.,
                    spatial_depth_occupied_uncertainty_m=0.,spatial_occupied_observations_required=2)
            if kind=="gt640_control":
                p.update(spatial_obstacle_vertical_band_m=.2,spatial_vertical_clearance_m=.2,
                    spatial_depth_free_space_margin_m=0.,spatial_depth_free_space_margin_ratio=0.,
                    spatial_depth_occupied_uncertainty_m=0.,spatial_occupied_observations_required=1)
            if "equal_ray_density" in kind:
                p.update(spatial_depth_stride=10,spatial_depth_edge_stride=4)
            report=out/(label+"_"+kind+".json")
            command=command_for(str(prepared[label][variant]),report,p,
                dict(post_depth_scale=1.,conservative_depth_shift_m=0.))
            command+=["--reference-replay-stride","5","--reference-edge-stride","2",
                "--protect-pending-obstacles","--allow-known-free-initial-escape"]
            if kind!="gt640_control":
                command.remove("--diagnostic-reference-depth")
                command+=["--model",str(model),"--prediction-cache-dir",str(out/"predictions"/variant/label)]
            if "interval" in kind:command+=["--depth-interval-profile",result["profiles"][variant]["path"]]
            print("Replay "+label+" "+kind,flush=True)
            with (out/(label+"_"+kind+".log")).open("w") as stream:
                subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
            data=json.loads(report.read_text());planning=data["planning"]
            row=dict(dataset=label,kind=kind,frames=data["frames_integrated"],success_rate=planning["success_rate"],
                false_free_rate=data["map"]["false_free_rate"],
                collisions=planning["successes"]-planning["collision_free_successes"],
                checks=data["qualification"]["checks"],spatial_thresholds_met=all(data["qualification"]["checks"].values()),
                report=str(report),command=command,reference_map=data["reference_map"])
            result["cases"].append(row);save(out/"summary.json",result)
            print(json.dumps({k:v for k,v in row.items() if k not in ["command","reference_map"]}),flush=True)
        maps=[c["reference_map"] for c in result["cases"] if c["dataset"]==label]
        if not all(m==maps[0] for m in maps):raise ValueError("Reference map changed between cases")
    if {name:digest(ROOT/name) for name in SOURCES}!=frozen:raise ValueError("Source changed during experiment")
    if any(digest(v["path"])!=v["sha256"] for v in result["profiles"].values()):raise ValueError("Profile changed")
    result["completed"]=True
    result["conclusion"]="Spatial development experiment only; cannot qualify asynchronous timing, tracking or flight."
    save(out/"summary.json",result)
    print("Completed without flight: "+str(out/"summary.json"),flush=True)

if __name__=="__main__":main()
