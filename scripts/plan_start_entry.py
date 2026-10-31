"""Experimental start-entry search; map and safety validators are unchanged."""
import copy,json,sys
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def save(p,data):Path(p).write_text(json.dumps(data,indent=2)+"\n")
def worker(spec_path):
    import numpy as np
    import spatial_mapping.navigation as nav
    from scripts.replay_mapping_backend import worker as replay_worker
    original_plan=nav.SpatialNavigator.plan;trace=[]
    def plan(self,current,goal,**kwargs):
        result=original_plan(self,current,goal,**kwargs)
        if getattr(self,"audit_role",-1)!=0 or result.success or not kwargs.get("allow_initial_escape",False):
            return result
        voxel=self.grid.world_to_voxel(current)
        blocked=self._inflated_obstacles()
        row=dict(frame=self.frames_integrated,original_reason=result.reason,candidates=0,
            prefix_rejections={},attempts=[],recovered=False)
        trace.append(row)
        if voxel not in blocked:return result
        goal_layer=self.grid.world_to_voxel(goal)[2]
        delta=np.asarray(goal)-np.asarray(current);norm=float(np.linalg.norm(delta[:2]))
        candidates=[]
        for v in self.grid.free_voxels()-blocked:
            if self.config.lock_path_altitude_to_goal and v[2]!=goal_layer:continue
            distance=float(np.linalg.norm(np.asarray(v)-np.asarray(voxel)))
            if distance>6:continue
            target=self.grid.voxel_to_world(v)
            if self.config.lock_path_altitude_to_goal:target[2]=goal[2]
            progress=float(np.dot((target-np.asarray(current))[:2],delta[:2])/norm) if norm else 0.
            candidates.append((distance,-progress,tuple(map(int,v)),target))
        candidates.sort(key=lambda x:x[:3]);row["candidates"]=len(candidates)
        rejected=Counter();attempts=0
        for distance,_,entry,target in candidates:
            prefix=self.initial_escape_path_safety_diagnostics(current,[target])
            if not prefix["safe"]:
                rejected[prefix["failure_reason"]]+=1;continue
            clone=copy.deepcopy(self)
            clone._nearest_voxel=lambda requested,available,max_offset=2,e=entry:e if e in available else None
            proposed=original_plan(clone,current,goal,**kwargs)
            attempts+=1
            row["attempts"].append(dict(entry=list(entry),offset_voxels=distance,
                success=proposed.success,path_length_m=proposed.path_length_m,reason=proposed.reason))
            if proposed.success and proposed.path_length_m>=1.5:
                row["recovered"]=True;row["prefix_rejections"]=dict(rejected)
                return proposed
            if attempts>=8:break
        row["prefix_rejections"]=dict(rejected)
        return result
    nav.SpatialNavigator.plan=plan
    try:replay_worker(spec_path)
    finally:save(Path(spec_path).with_name(Path(spec_path).stem+"_entry_trace.json"),trace)

if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1]!="--worker":
        raise SystemExit("Use scripts/replay_final_monocular.py --bundle FILE --output DIR")
    worker(sys.argv[2])
