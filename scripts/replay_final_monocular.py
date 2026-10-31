"""Portable replay of the frozen final configuration. No training or flight."""
from pathlib import Path
import argparse,json,tarfile,sys,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--export-full-bundle",type=Path);p.add_argument("--bundle",type=Path)
 p.add_argument("--output",type=Path,default=ROOT/"logs/reproducao_final")
 a=p.parse_args()
 config=json.loads((ROOT/"config/final_monocular_replay.json").read_text())
 if a.export_full_bundle:
  dataset=ROOT/config["source_run"];model=ROOT/config["source_model"];cache=ROOT/config["source_predictions"]
  spec={"temporal_profile":str(ROOT/config["source_profile"])}
  files=[(q,Path("run")/q.relative_to(dataset)) for q in dataset.rglob("*") if q.is_file()]
  files += [(q,Path("predictions")/q.name) for q in cache.glob("*.npy")]
  files += [(model,Path("model.onnx")),(Path(spec["temporal_profile"]),Path("profile.json"))]
  a.export_full_bundle.parent.mkdir(parents=True,exist_ok=True)
  with tarfile.open(a.export_full_bundle,"w:gz") as t:
   for src,dest in files:t.add(src,arcname=str(dest),recursive=False)
  print("Bundle exported:",a.export_full_bundle);return
 if not a.bundle:p.error("Use --bundle FILE (full inputs) or --export-full-bundle FILE on the original workstation.")
 a.output.mkdir(parents=True,exist_ok=False)
 data=a.output/"inputs";data.mkdir()
 with tarfile.open(a.bundle) as t:
  for m in t.getmembers():
   dest=(data/m.name).resolve()
   if not dest.is_relative_to(data.resolve()) or not m.isfile():raise ValueError("Unsafe archive entry")
   with t.extractfile(m) as src:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(src.read())
 expected=config["model_sha256"]
 assert hashlib.sha256((data/"model.onnx").read_bytes()).hexdigest()==expected
 c=[sys.executable,str(ROOT/"estudos_e_analises/avaliar_mapeamento_monocular_offline.py"),
    str(data/"run"),"--output",str(a.output/"validation.json")]
 args=list(config["arguments"])
 for flag in ["--output","--model","--prediction-cache-dir"]:args.extend([flag,""])
 for flag,new in [("--output",a.output/"validation.json"),("--model",data/"model.onnx"),("--prediction-cache-dir",data/"predictions")]:
  args[args.index(flag)+1]=str(new)
 c=[sys.executable,c[1],str(data/"run"),*args]
 spec=dict(command=c,overrides=config["overrides"],
           audit=str(a.output/"mapping_audit.json"),temporal_profile=str(data/"profile.json"))
 path=a.output/"spec.json";path.write_text(json.dumps(spec,indent=2))
 subprocess.run([sys.executable,str(ROOT/"scripts/replay_curved_mapping.py"),"--worker",str(path)],cwd=ROOT,check=True,
                stdout=(a.output/"replay.log").open("w"),stderr=subprocess.STDOUT)
 r=json.loads((a.output/"validation.json").read_text())
 assert r["planning"]["successes"]==12 and r["planning"]["attempts"]==17
 assert r["planning"]["collision_free_successes"]==12 and r["frames_integrated"]==77
 assert abs(r["map"]["false_free_rate"]-4/401)<1e-12
 print("Portable replay reproduced: 12/17, false-free 4/401, 0 discrete collisions.")
if __name__=="__main__":main()
