#!/usr/bin/env python3
"""Select v21 only on spatial validation, freeze it, then apply unchanged gates."""
import argparse,hashlib,json,math,shutil,subprocess,sys,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import yaml
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.diagnose_monocular_gt_replay import command_for
from scripts.evaluate_depth_intervals import calibrate
from scripts.run_monocular_resolution_640 import digest,save

def spatial_rank(report,metrics):
    planning=report["planning"];checks=report["qualification"]["checks"]
    success=planning["success_rate"];false_free=report["map"]["false_free_rate"]
    collisions=planning["successes"]-planning["collision_free_successes"]
    if not math.isfinite(false_free):false_free=float("inf")
    failed=sum(not checks[name] for name in ["plan_success_rate","false_free_rate","collision_free_plans"])
    excess=max(0.,(.8-success)/.8)+max(0.,(false_free-.1)/.1)
    edges=metrics["dangerous_far_edge_rate_1m"]+metrics["dangerous_near_edge_rate_1m"]
    return (failed,collisions,excess,-success,edges,metrics["mae_m"])

def logged(command,path):
    with path.open("w") as stream:
        subprocess.run(list(map(str,command)),cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    parser.add_argument("--register-only",action="store_true")
    parser.add_argument("--wait-for-training",action="store_true")
    args=parser.parse_args();out=args.experiment.resolve()
    spec=json.loads((out/"data_split.json").read_text())
    data=json.loads((out/"data_collection.json").read_text())
    if not data.get("completed"):raise ValueError("Data incomplete")
    validation=[Path(j["prepared"]) for j in data["jobs"] if j["role"]=="validation"]
    if len(validation)!=1:raise ValueError("Expected original separate validation sequence")
    sources=["scripts/evaluate_v21_native.py","scripts/export_depth_anything_v2_metric_onnx.py",
        "scripts/finetune_depth_anything_baylands.py","scripts/evaluate_depth_intervals.py",
        "scripts/diagnose_monocular_gt_replay.py","spatial_mapping/depth_model.py",
        "spatial_mapping/depth_preprocessing.py","spatial_mapping/geometry.py",
        "spatial_mapping/navigation.py","spatial_mapping/occupancy.py","spatial_mapping/depth_intervals.py",
        "estudos_e_analises/avaliar_mapeamento_monocular_offline.py",
        "estudos_e_analises/avaliar_profundidade_monocular.py","config/spatial_monocular_temporal.yaml"]
    protocol=dict(data_split_sha256=digest(out/"data_split.json"),data_collection_sha256=digest(out/"data_collection.json"),
        epochs=3,checkpoint_selection="failed spatial checks excluding frame count, collisions, normalized joint gate excess, negative success, bidirectional edge error, MAE",
        validation=str(validation[0]),validation_frames=77,validation_qualification=False,
        point_profile="Same aligned-640 pending-protected raw point profile as resolution experiment",
        interval_profile="Same pending-protected interval profile, calibrated after checkpoint selection",
        profile_selection="Same spatial rank on validation; point wins exact ties. Freeze before evaluation.",
        calibration=spec["calibration_source"],evaluation=spec["evaluation_sources"],
        pixel_gate="Unmodified limits of avaliar_profundidade_monocular.py; 24 uniform frames, raw depth",
        spatial_limits=dict(minimum_frames=100,minimum_success_rate=.8,maximum_false_free_rate=.1,maximum_colliding_plans=0),
        selection_uses_evaluation=False,source_sha256={name:digest(ROOT/name) for name in sources},
        flight_authorized=False)
    target=out/"candidate_protocol.json"
    if args.register_only:
        if target.exists():raise ValueError("Candidate protocol already frozen")
        save(target,protocol)
        for name in sources:
            dest=out/"candidate_source"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
        print("V21 selection and gate protocol frozen.",flush=True);return
    if json.loads(target.read_text())!=protocol:raise ValueError("Candidate protocol changed")
    deadline=time.monotonic()+10800
    while not (out/"training_complete.json").exists():
        if not args.wait_for_training:raise ValueError("Training incomplete")
        if time.monotonic()>deadline:raise TimeoutError("Training did not finish")
        time.sleep(3)
    complete=json.loads((out/"training_complete.json").read_text())
    if not complete.get("completed"):raise ValueError("Training incomplete")
    history=json.loads((out/"training/candidate.history.json").read_text())
    params=yaml.safe_load((ROOT/"config/spatial_monocular_temporal.yaml").read_text())["drone_offboard_node"]["ros__parameters"]
    model_dir=out/"epochs";model_dir.mkdir(exist_ok=True)
    candidates=[]
    for i in range(1,4):
        checkpoint=out/("training/candidate_epoch"+str(i).zfill(2)+".pth")
        model=model_dir/("epoch"+str(i).zfill(2)+".onnx")
        print("Export checkpoint epoch "+str(i),flush=True)
        logged([ROOT/"models/export_env/bin/python",ROOT/"scripts/export_depth_anything_v2_metric_onnx.py",
            "--source-dir",ROOT.parent/"Depth-Anything-V2","--checkpoint",checkpoint,"--output",model,
            "--encoder","vits","--input-width","686","--input-height","518"],model.with_suffix(".export.log"))
        candidates.append(dict(epoch=i,checkpoint=str(checkpoint),checkpoint_sha256=digest(checkpoint),
            model=str(model),model_sha256=digest(model),training_validation_metrics=history[i-1]))
    def replay(run,model,report,cache,kind="point",profile=None):
        p=dict(params)
        if kind=="interval":
            p.update(spatial_obstacle_vertical_band_m=.2,spatial_vertical_clearance_m=.2,
                spatial_depth_free_space_margin_m=0.,spatial_depth_free_space_margin_ratio=0.,
                spatial_depth_occupied_uncertainty_m=0.,spatial_occupied_observations_required=2)
        command=command_for(str(run),report,p,dict(post_depth_scale=1.,conservative_depth_shift_m=0.))
        command.remove("--diagnostic-reference-depth")
        command+=["--model",str(model),"--prediction-cache-dir",str(cache),
            "--reference-replay-stride","5","--reference-edge-stride","2",
            "--protect-pending-obstacles","--allow-known-free-initial-escape"]
        if kind=="interval":command+=["--depth-interval-profile",str(profile)]
        logged(command,report.with_suffix(".log"))
        return json.loads(report.read_text()),command
    def validate(candidate):
        report=out/("validation_epoch"+str(candidate["epoch"]).zfill(2)+".json")
        cache=out/"predictions"/candidate["model_sha256"]/"validation"
        print("Spatial validation epoch "+str(candidate["epoch"]),flush=True)
        result,command=replay(validation[0],candidate["model"],report,cache)
        candidate.update(spatial_report=str(report),spatial_command=command,
            rank=spatial_rank(result,candidate["training_validation_metrics"]))
        print("Validation epoch "+str(candidate["epoch"])+" rank="+str(candidate["rank"]),flush=True)
        return candidate
    with ThreadPoolExecutor(max_workers=3) as pool:
        candidates=list(pool.map(validate,candidates))
    candidates.sort(key=lambda c:(tuple(c["rank"]),c["epoch"]))
    chosen=candidates[0]
    save(out/"checkpoint_selection.json",dict(candidates=candidates,selected_epoch=chosen["epoch"],
        selection_set="validation only",evaluation_opened=False))
    final_checkpoint=ROOT/"models/depth_anything_v2_metric_baylands_vits_v21_native640.pth"
    final_model=ROOT/"models/depth_anything_v2_metric_baylands_vits_v21_native640_686x518_fp32.onnx"
    if final_model.exists() or final_checkpoint.exists():raise ValueError("Never overwrite a candidate")
    shutil.copyfile(chosen["checkpoint"],final_checkpoint);shutil.copyfile(chosen["model"],final_model)
    metadata=json.loads((out/"training/candidate.training.json").read_text())
    metadata.update(selected_epoch_spatial_validation=chosen["epoch"],
        selection_report=str(out/"checkpoint_selection.json"),data_protocol=str(out/"data_split.json"),
        candidate_protocol=str(target),model_sha256=digest(final_model))
    save(final_checkpoint.with_suffix(".training.json"),metadata)
    print("Frozen selected epoch="+str(chosen["epoch"])+" model="+str(final_model),flush=True)
    calibration=Path(spec["calibration_source"])
    profile,hard=calibrate(calibration,out/"predictions"/chosen["model_sha256"]/"calibration",
                           final_model,chosen["model_sha256"])
    profile_path=out/"frozen_interval_profile.json";save(profile_path,profile)
    save(out/"calibration_frames.json",hard)
    profile_sha=digest(profile_path)
    interval_val=out/"validation_selected_interval.json"
    interval_data,interval_command=replay(validation[0],final_model,interval_val,
        out/"predictions"/chosen["model_sha256"]/"validation","interval",profile_path)
    interval_rank=spatial_rank(interval_data,chosen["training_validation_metrics"])
    kind="interval" if interval_rank<tuple(chosen["rank"]) else "point"
    frozen=dict(selected_epoch=chosen["epoch"],model=str(final_model),model_sha256=digest(final_model),
        kind=kind,interval_profile=str(profile_path),interval_profile_sha256=profile_sha,
        point_validation_rank=chosen["rank"],interval_validation_rank=interval_rank,
        interval_validation_command=interval_command,selection_set="validation only",evaluation_opened=False)
    save(out/"frozen_candidate.json",frozen)
    print("Frozen mapping profile="+kind,flush=True)
    pixel=out/"pixel_gate.json"
    logged([ROOT/"models/runtime_env/bin/python",ROOT/"estudos_e_analises/avaliar_profundidade_monocular.py",
        "--model",final_model,"--calibration-run",calibration,"--validation-run",validation[0],
        "--frames","24","--output",pixel],out/"pixel_gate.log")
    pixel_data=json.loads(pixel.read_text())
    result=dict(selected=frozen,pixel_qualification=pixel_data["qualification"],spatial_cases=[],
        qualification=dict(passed=False),flight="not_started",
        limitation="Development re-rendered trajectories; no moving-flight or external-scenario validation.")
    save(out/"candidate_summary.json",result)
    print("Pixel gate="+str(pixel_data["qualification"]),flush=True)
    def evaluate_case(item):
        index,run=item;report=out/("evaluation_"+str(index)+".json")
        data,command=replay(Path(run),final_model,report,out/"predictions"/chosen["model_sha256"]/("evaluation_"+str(index)),kind,profile_path)
        planning=data["planning"]
        return dict(dataset=run,frames=data["frames_integrated"],kind=kind,
            success_rate=planning["success_rate"],false_free_rate=data["map"]["false_free_rate"],
            collisions=planning["successes"]-planning["collision_free_successes"],
            qualification=data["qualification"],report=str(report),command=command)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(evaluate_case,item) for item in enumerate(spec["evaluation_sources"],1)]
        for future in as_completed(futures):
            row=future.result();result["spatial_cases"].append(row);save(out/"candidate_summary.json",result)
            print(json.dumps({k:v for k,v in row.items() if k!="command"}),flush=True)
    if digest(final_model)!=frozen["model_sha256"] or digest(profile_path)!=profile_sha:raise ValueError("Frozen candidate changed")
    if any(digest(ROOT/name)!=expected for name,expected in protocol["source_sha256"].items()):raise ValueError("Frozen sources changed")
    result["spatial_cases"].sort(key=lambda r:r["dataset"])
    result["qualification"]["passed"]=bool(pixel_data["qualification"]["passed"] and
        len(result["spatial_cases"])==2 and all(r["qualification"]["passed"] for r in result["spatial_cases"]))
    result["completed"]=True
    result["flight"]="eligible_for_runtime_integration_audit" if result["qualification"]["passed"] else "blocked_by_offline_gate"
    save(out/"candidate_summary.json",result)
    print("V21 attempt complete: qualification="+str(result["qualification"]["passed"]),flush=True)

if __name__=="__main__":main()
