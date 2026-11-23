"""Conservative revision of latched obstacles from explicit multiview free evidence."""
import numpy as np

from .projected_free_evidence import projected_free_voxels

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


def conservative_lower_bound(depth, k, config, profile):
    d=np.asarray(depth,dtype=float)
    valid=np.isfinite(d)&(d>=config.min_depth_m)&(d<=config.max_depth_m)
    lower=np.full(d.shape,np.nan)
    yy,xx=np.indices(d.shape)
    radial=np.sqrt(1+((xx-k.cx)/k.fx)**2+((yy-k.cy)/k.fy)**2)
    margin=config.depth_free_space_margin_m+config.depth_free_space_margin_ratio*d*radial
    margin=np.minimum(margin,config.depth_free_space_margin_max_m)
    calibrated=profile.bounds(d[valid])[0]
    lower[valid]=np.minimum(calibrated,np.maximum(0,d[valid]-margin[valid]/radial[valid]))
    return lower


def revise_after_rays(self, engine, origin, points, mask, *, max_range_m,
                      occupied_uncertainty_m, intrinsics, pose, lower,
                      occupied_vertical_bounds=None):
    hits=set()
    for p in points[np.asarray(mask,dtype=bool)]:
        distance=float(np.linalg.norm(p-origin))
        if distance==0 or not np.isfinite(distance) or distance>max_range_m:continue
        hits.update(self._occupied_ray_voxels(
            origin, p, distance, occupied_uncertainty_m, occupied_vertical_bounds,
        ))
    positives=self.occupied_voxels()|self.pending_occupied_voxels()
    assert hits<=positives
    # Full projected voxel footprint must clear the far face, including every pixel.
    certified=projected_free_voxels(positives-hits,self.config.resolution_m,
        intrinsics,pose,lower)
    retired=engine.observe(positives,hits,certified,origin,self._integration_frame)
    confirmed=self.occupied_voxels()
    engine.retire_to_unknown(self,retired)
    assert not retired&self.free_voxels()
    return dict(frame=self._integration_frame,positive_voxels=len(positives),
        current_positive=len(hits),certified_free=len(certified),retired=len(retired),
        retired_confirmed=sum(v in confirmed for v in retired))
