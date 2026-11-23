"""Offline bounded curved initial escape. No unknown-space or obstacle release."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
from spatial_mapping.v21_entry import curved_entries, try_curved_entry

def worker(spec_path):
    import spatial_mapping.navigation as nav
    import scripts.replay_mapping_backend as old
    from scripts.replay_temporal_mapping import worker as temporal
    original=nav.SpatialNavigator.plan
    replay=old.worker;trace=[]
    def install(path):
        straight=nav.SpatialNavigator.plan
        def plan(self,current,goal,**kwargs):
            result=straight(self,current,goal,**kwargs)
            if getattr(self,"audit_role",-1)!=0 or result.success or not kwargs.get("allow_initial_escape"):return result
            return try_curved_entry(self, current, goal, result, original, trace)
        nav.SpatialNavigator.plan=plan
        return replay(path)
    old.worker=install
    try:temporal(spec_path)
    finally:save(Path(spec_path).with_name("curved_trace.json"),trace)

if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1]!="--worker":
        raise SystemExit("Use scripts/replay_final_monocular.py --bundle FILE --output DIR")
    worker(sys.argv[2])
