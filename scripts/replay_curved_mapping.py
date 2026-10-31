"""Offline bounded curved initial escape. No unknown-space or obstacle release."""
import copy,heapq,json,sys
from pathlib import Path
from itertools import product
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
def curved_entries(nav,current,goal,radius_voxels=6,max_length_m=6.):
    grid=nav.grid;start=grid.world_to_voxel(current)
    free=grid.free_voxels();blocked=nav._inflated_obstacles()
    if start not in free or start not in blocked:return [],0
    layer=grid.world_to_voxel(goal)[2]
    nodes={v for v in free if v[2]==layer and np.linalg.norm(np.array(v)-start)<=radius_voxels}
    nodes.add(start)
    def point(v):
        if v==start:return np.asarray(current,dtype=float)
        p=grid.voxel_to_world(v);p[2]=goal[2];return p
    distance={start:0.};parent={};queue=[(0.,start)];exits=[];visited=0
    offsets=[v for v in product([-1,0,1],repeat=3) if v!=(0,0,0)]
    while queue:
        cost,v=heapq.heappop(queue)
        if cost!=distance[v]:continue
        visited+=1
        if v not in blocked:
            chain=[v]
            while chain[-1]!=start:chain.append(parent[chain[-1]])
            chain.reverse()
            exits.append((cost,[point(n) for n in chain[1:]]));continue
        for d in offsets:
            w=tuple(v[i]+d[i] for i in range(3))
            if w not in nodes:continue
            a,b=point(v),point(w)
            new=cost+float(np.linalg.norm(b-a))
            if new>max_length_m or new>=distance.get(w,float("inf")):continue
            cells=list(grid.segment_voxels(a,b))
            if any(c not in free for c in cells):continue
            left=False;valid=True
            for c in cells:
                if c not in blocked:left=True
                elif left:valid=False;break
            # A segment cannot exit and return into the clearance shell.
            if not valid or (left and w in blocked):continue
            distance[w]=new;parent[w]=v;heapq.heappush(queue,(new,w))
    return exits,visited

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
            entries,visited=curved_entries(self,current,goal)
            row=dict(frame=self.frames_integrated,visited=visited,exits=len(entries),attempts=[],recovered=False)
            trace.append(row)
            for length,prefix in entries[:8]:
                if not self.initial_escape_path_safety_diagnostics(current,prefix)["safe"]:continue
                clone=copy.deepcopy(self)
                suffix=original(clone,prefix[-1],goal,allow_initial_escape=False)
                if not suffix.success:
                    row["attempts"].append(dict(prefix_length_m=length,success=False,reason=suffix.reason));continue
                waypoints=prefix+[np.asarray(p) for p in suffix.waypoints_ned_m]
                check=self.initial_escape_path_safety_diagnostics(current,waypoints)
                path_length=self._waypoint_length([np.asarray(current),*waypoints])
                direction=np.asarray(goal)-np.asarray(current)
                progress=float(np.dot(waypoints[-1]-current,direction)/max(np.linalg.norm(direction),1e-9))
                safe=check["safe"] and path_length>=1.5 and progress>=self.config.min_subgoal_progress_m
                row["attempts"].append(dict(prefix_length_m=length,success=bool(safe),progress_m=progress,reason=check["failure_reason"]))
                if not safe:continue
                suffix.waypoints_ned_m=[tuple(map(float,p)) for p in waypoints]
                suffix.path_length_m=path_length
                suffix.raw_path_length_m+=length
                cells=[]
                for a,b in zip([np.asarray(current),*waypoints[:-1]],waypoints):
                    for cell in self.grid.segment_voxels(a,b):
                        if not cells or cells[-1]!=cell:cells.append(cell)
                suffix.path_voxels=cells
                suffix.reason="curved_initial_escape"
                suffix.diagnostics=dict(result.diagnostics,curved_initial_escape=True,
                    curved_prefix_length_m=length,final_path_safety=check)
                row["recovered"]=True
                return suffix
            return result
        nav.SpatialNavigator.plan=plan
        return replay(path)
    old.worker=install
    try:temporal(spec_path)
    finally:save(Path(spec_path).with_name("curved_trace.json"),trace)

if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1]!="--worker":
        raise SystemExit("Use scripts/replay_final_monocular.py --bundle FILE --output DIR")
    worker(sys.argv[2])
