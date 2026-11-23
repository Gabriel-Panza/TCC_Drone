"""Shared v21 start-entry search used by the replay and guarded SITL."""

import copy
import heapq
from collections import Counter
from itertools import product

import numpy as np


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


def try_straight_entry(self, current, goal, result, original_plan, trace, **kwargs):
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


def try_curved_entry(self, current, goal, result, original, trace):
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
