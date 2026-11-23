"""Experimental start-entry search; map and safety validators are unchanged."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def save(p,data):Path(p).write_text(json.dumps(data,indent=2)+"\n")
def worker(spec_path):
    from spatial_mapping.v21_entry import try_straight_entry
    import spatial_mapping.navigation as nav
    from scripts.replay_mapping_backend import worker as replay_worker
    original_plan=nav.SpatialNavigator.plan;trace=[]
    def plan(self,current,goal,**kwargs):
        result=original_plan(self,current,goal,**kwargs)
        if getattr(self,"audit_role",-1)!=0 or result.success or not kwargs.get("allow_initial_escape",False):
            return result
        return try_straight_entry(self, current, goal, result, original_plan, trace, **kwargs)
    nav.SpatialNavigator.plan=plan
    try:replay_worker(spec_path)
    finally:save(Path(spec_path).with_name(Path(spec_path).stem+"_entry_trace.json"),trace)

if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1]!="--worker":
        raise SystemExit("Use scripts/replay_final_monocular.py --bundle FILE --output DIR")
    worker(sys.argv[2])
