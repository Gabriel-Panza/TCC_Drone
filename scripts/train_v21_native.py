#!/usr/bin/env python3
"""Run the single preregistered v21 training attempt on fresh aligned data."""
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json,subprocess,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,data):path.write_text(json.dumps(data,indent=2)+"\n")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment",type=Path,required=True)
    args=parser.parse_args()
    out=args.experiment.resolve()
    spec=json.loads((out/"data_split.json").read_text())
    data=json.loads((out/"data_collection.json").read_text())
    if not data.get("completed"):raise ValueError("Native data collection incomplete")
    for name,expected in spec["source_sha256"].items():
        if digest(ROOT/name)!=expected:raise ValueError("Frozen data/training code changed: "+name)
    if digest(spec["checkpoint"])!=spec["checkpoint_sha256"]:raise ValueError("Checkpoint changed")
    train={j["source"]:Path(j["prepared"]) for j in data["jobs"] if j["role"]=="train"}
    validation=[Path(j["prepared"]) for j in data["jobs"] if j["role"]=="validation"]
    if set(train)!=set(spec["train_runs"]):raise ValueError("Training role mismatch")
    source_validation={j["source"] for j in data["jobs"] if j["role"]=="validation"}
    if set(train)&source_validation:raise ValueError("Data leakage")
    folders=list(train.values())
    focus=[train[str(Path(p).resolve().parent.parent)]/"frames"/Path(p).name for p in spec["focus_frames"]]
    if any(not p.is_file() for p in focus):raise ValueError("Missing newly captured focus frame")
    target=out/"training/candidate.pth"
    if target.exists():raise ValueError("Training output already exists; never overwrite an attempt")
    target.parent.mkdir(exist_ok=True)
    command=[str(ROOT/"models/train_env/bin/python"),str(ROOT/"scripts/finetune_depth_anything_baylands.py"),
        "--source-dir",str(ROOT.parent/"Depth-Anything-V2"),"--checkpoint",spec["checkpoint"],"--output",str(target),
        "--aligned-640","--save-each-epoch"]
    h=spec["hyperparameters"]
    for key in ["epochs","batch_size","head_lr","encoder_lr","unfreeze_encoder_blocks","seed","encoder"]:
        command+=["--"+key.replace("_","-"),str(h[key])]
    for key,value in spec["loss_weights"].items():command+=["--"+key.replace("_","-"),str(value)]
    for run in folders:command+=["--train-run",str(run)]
    for run in validation:command+=["--validation-run",str(run)]
    command+=["--focus-repeat",str(spec["focus_repeat"]),"--focus-frame-repeat",str(spec["focus_frame_repeat"])]
    for original in spec["focus_runs"]:command+=["--focus-run",str(train[str(Path(original).resolve())])]
    for frame in focus:command+=["--focus-frame",str(frame)]
    planned=data["train_frames"]+sum(next(j["frames"] for j in data["jobs"] if j["source"]==p)*spec["focus_repeat"] for p in spec["focus_runs"])+len(focus)*spec["focus_frame_repeat"]
    sources=["scripts/train_v21_native.py","scripts/finetune_depth_anything_baylands.py",
        "spatial_mapping/depth_preprocessing.py","spatial_mapping/depth_model.py"]
    protocol=dict(started_at=datetime.now(ZoneInfo("America/Sao_Paulo")).isoformat(),training_command=command,
        planned_effective_frames=planned,train_unique_frames=data["train_frames"],validation_frames=data["validation_frames"],
        checkpoint_selection=spec["selection"],selection_limitation=spec["selection_limitation"],
        data_split_sha256=digest(out/"data_split.json"),data_collection_sha256=digest(out/"data_collection.json"),
        source_sha256={name:digest(ROOT/name) for name in sources},flight_authorized=False)
    if (out/"training_protocol.json").exists():raise ValueError("Training protocol already frozen")
    save(out/"training_protocol.json",protocol)
    for name in sources:
        dest=out/"training_source"/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/name,dest)
    print("Training v21; planned effective frames="+str(planned)+"; log="+str(out/"training.log"),flush=True)
    with (out/"training.log").open("w") as stream:
        subprocess.run(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,check=True)
    if any(digest(ROOT/name)!=expected for name,expected in protocol["source_sha256"].items()):raise ValueError("Training sources changed")
    history=json.loads(target.with_suffix(".history.json").read_text())
    if len(history)!=h["epochs"]:raise ValueError("Training epochs incomplete")
    save(out/"training_complete.json",dict(completed=True,epochs=len(history),training_best=str(target),
        epoch_checkpoints=[str(target.with_name("candidate_epoch"+str(i).zfill(2)+".pth")) for i in range(1,h["epochs"]+1)],
        history_sha256=digest(target.with_suffix(".history.json")),flight_authorized=False))
    print("Training completed; checkpoint must still be selected by spatial validation.",flush=True)

if __name__=="__main__":main()
