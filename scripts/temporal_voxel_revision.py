"""Experimental offline revision of latched obstacles from explicit multiview free evidence."""
import numpy as np

class TemporalVoxelRevision:
    """Three translated observations after the last hit; retirement means unknown."""
    def __init__(self, required_views=3, baseline_m=.75):
        if required_views<3 or int(required_views)!=required_views or not np.isfinite(baseline_m) or baseline_m<=0:
            raise ValueError("need >=3 views and a positive finite baseline")
        self.required_views=int(required_views);self.baseline=float(baseline_m)
        self.birth={};self.views={};self.retired=[]
    def observe(self, positives, current_hits, certified_free, origin, frame):
        p=np.asarray(origin,dtype=float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError("invalid origin")
        if not current_hits<=positives:raise ValueError("current hits must remain positive")
        for v in set(self.birth)-positives:
            self.birth.pop(v,None);self.views.pop(v,None)
        for v in positives:
            if v not in self.birth or v in current_hits:
                self.birth[v]=p.copy();self.views[v]=[]
        ready=set()
        for v in certified_free&positives-current_hits:
            if np.linalg.norm(p-self.birth[v])<self.baseline-1e-9:continue
            views=self.views[v]
            if any(np.linalg.norm(p-q)<self.baseline-1e-9 for q in views):continue
            views.append(p.copy())
            if len(views)>=self.required_views:
                ready.add(v)
                self.retired.append(dict(voxel=list(map(int,v)),frame=int(frame),
                    birth_origin=self.birth[v].tolist(),free_origins=[q.tolist() for q in views]))
                self.birth.pop(v);self.views.pop(v)
        return ready
    @staticmethod
    def retire_to_unknown(grid, voxels):
        """No free evidence is synthesized. Later raw rays must establish free."""
        for v in voxels:
            grid._confirmed_occupied.discard(v)
            grid._occupied_evidence_frames.pop(v,None)
            grid._pending_free_observations.pop(v,None)
            grid._free_viewpoint_sectors.pop(v,None)
            grid._log_odds[v]=0.
