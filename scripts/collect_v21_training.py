#!/usr/bin/env python3
"""Regenerate only the original training/validation poses, retaining native data."""
import argparse
from datetime import datetime
import hashlib,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_monocular_resolution_640 import prepare_capture,digest,save

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume",type=Path)
    args=parser.parse_args()
    previous=json.loads((ROOT/"models/depth_anything_v2_metric_baylands_vits_v19_final.training.json").read_text())
    bundle=json.loads((ROOT/"config/monocular_temporal_campaign.json").read_text())
    train=[Path(p).resolve() for p in previous["train_runs"]]
    validation=[Path(p).resolve() for p in previous["validation_runs"]]
    reserved={Path(ROOT/p).resolve() for p in bundle["datasets"]}
    reserved.add(ROOT/"datasets/spatial_mapping/run_20260827_165430_607620")
    if set(train)&(reserved|set(validation)):raise ValueError("Training/reserved run overlap")
    jobs=[("train",p) for p in train]+[("validation",p) for p in validation]
    checkpoint=ROOT/"models/depth_anything_v2_metric_baylands_vits_v19_final.pth"
    weights=["edge_multiplier","unsafe_tail_weight","unsafe_max_depth_m",
        "unsafe_overestimate_weight","unsafe_overestimate_tail_weight","unsafe_overestimate_tail_fraction",
        "edge_overestimate_weight","unsafe_underestimate_weight","unsafe_underestimate_tail_weight",
        "unsafe_underestimate_tail_fraction","edge_underestimate_weight","edge_gradient_weight"]
    sources=["scripts/collect_v21_training.py","scripts/collect_camera_consistency.py",
        "scripts/run_monocular_resolution_640.py","spatial_mapping/camera_sampling.py",
        "spatial_mapping/depth_preprocessing.py","spatial_mapping/depth_model.py",
        "scripts/finetune_depth_anything_baylands.py"]
    spec=dict(schema="v21_native_training_data_v1",checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
        train_runs=list(map(str,train)),validation_runs=list(map(str,validation)),excluded_from_training=sorted(map(str,reserved)),
        loss_weights={k:previous[k] for k in weights},
        hyperparameters=dict(epochs=3,batch_size=2,head_lr=2e-6,encoder_lr=1e-7,
            unfreeze_encoder_blocks=2,seed=42,encoder="vits",aligned_640=True),
        focus_runs=previous["focus_runs"],focus_repeat=previous["focus_repeat"],
        focus_frames=previous["focus_frames"],focus_frame_repeat=previous["focus_frame_repeat"],
        calibration_source=str(ROOT/"logs/monocular_resolution_640/20261029_200646/calibration/aligned640"),
        evaluation_sources=[str(ROOT/"logs/monocular_resolution_640/20261029_200646"/label/"aligned640")
                            for label in ["evaluation_1","evaluation_2"]],
        selection="Epochs ranked only on fresh original validation run: failed spatial checks excluding frame count, collisions, normalized gate excess, negative success, bidirectional edge error, MAE.",
        selection_limitation="Original validation has 77 frames; selection is descriptive, not flight qualification or independent testing.",
        post_training="Freeze selected checkpoint, fit intervals only on calibration, then evaluate both fixed 122/117-frame sets. No evaluation-driven training retries.",
        spatial_limits=dict(minimum_frames=100,minimum_success_rate=.8,maximum_false_free_rate=.1,maximum_colliding_plans=0),
        acquisition=dict(native_size=[1280,960],working_size=[640,480],sampling="matched_nearest",
            pose_rpc="native_with_bounded_cli_fallback",settle_seconds=.2,ros_bridge=True,
            actual_pose_check=True,exact_rgb_gt_stamps=True,no_flight=True),
        source_sha256={name:digest(ROOT/name) for name in sources},
        original_manifest_sha256={str(p):{name:digest(p/name) for name in ["manifest.json","events.jsonl"]} for _,p in jobs})
    if args.resume:
        out=args.resume.resolve();saved=json.loads((out/"data_split.json").read_text())
        if saved!=spec:raise ValueError("Data protocol or source changed")
        result=json.loads((out/"data_collection.json").read_text())
    else:
        out=ROOT/"logs/monocular_v21_native"/datetime.now().strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True,exist_ok=False);save(out/"data_split.json",spec)
        for name in sources:
            dest=out/"source"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        result=dict(completed=False,jobs=[],flight_started=False);save(out/"data_collection.json",result)
    print("V21 experiment: "+str(out),flush=True)
    for role,run in jobs:
        if any(j["source"]==str(run) and j["completed"] for j in result["jobs"]):continue
        log=out/(role+"_"+run.name+"_capture.log")
        command=["/usr/bin/python3",str(ROOT/"scripts/collect_camera_consistency.py"),"--scene","baylands",
            "--ros-bridge","--trajectory-run",str(run),"--pairs-per-pose","1","--settle-seconds",".2","--pose-rpc","native"]
        print("Capture "+role+" "+run.name,flush=True)
        with log.open("w") as stream:
            subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
        captures=[line.split("Collection: ",1)[1] for line in log.read_text().splitlines() if line.startswith("Collection: ")]
        if len(captures)!=1:raise ValueError("Missing capture location")
        capture=Path(captures[0])
        prepared=prepare_capture(capture,run,out/"data"/role/run.name,variants_to_prepare=("aligned640",))
        folder=prepared["aligned640"]
        manifest=json.loads((folder/"manifest.json").read_text())
        manifest["metadata"].update(dataset_role=role,v21_fresh_native=True)
        save(folder/"manifest.json",manifest)
        row=dict(role=role,source=str(run),capture=str(capture),prepared=str(folder),
            frames=manifest["frames_saved"],completed=True,command=command)
        result["jobs"].append(row);save(out/"data_collection.json",result)
        print(json.dumps(row),flush=True)
    if {name:digest(ROOT/name) for name in sources}!=spec["source_sha256"]:raise ValueError("Data generation code changed")
    result["completed"]=True
    result["train_frames"]=sum(j["frames"] for j in result["jobs"] if j["role"]=="train")
    result["validation_frames"]=sum(j["frames"] for j in result["jobs"] if j["role"]=="validation")
    save(out/"data_collection.json",result)
    print("V21 fresh native data complete: "+str(out),flush=True)

if __name__=="__main__":main()
