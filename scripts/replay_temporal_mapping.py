"""Integração temporal da configuração monocular final no replay offline."""
import sys,json
from pathlib import Path
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
def worker(spec_path):
    import spatial_mapping.navigation as nav
    from spatial_mapping.occupancy import OccupancyGrid3D
    from spatial_mapping.depth_intervals import DepthIntervalProfile
    from spatial_mapping.projected_free_evidence import projected_free_voxels
    from scripts.temporal_voxel_revision import TemporalVoxelRevision
    from scripts.plan_start_entry import worker as replay
    spec=json.loads(Path(spec_path).read_text())
    profile=DepthIntervalProfile(json.loads(Path(spec["temporal_profile"]).read_text()))
    original_integrate=nav.SpatialNavigator.integrate_depth
    original_rays=OccupancyGrid3D.integrate_rays
    context={};engine=TemporalVoxelRevision(3,.75);records=[]
    def integrate(self,depth,k,pose,*args,**kwargs):
        if getattr(self,"audit_role",-1)!=0:
            return original_integrate(self,depth,k,pose,*args,**kwargs)
        d=np.asarray(depth,dtype=float)
        valid=np.isfinite(d)&(d>=self.config.min_depth_m)&(d<=self.config.max_depth_m)
        lower=np.full(d.shape,np.nan)
        yy,xx=np.indices(d.shape)
        radial=np.sqrt(1+((xx-k.cx)/k.fx)**2+((yy-k.cy)/k.fy)**2)
        margin=self.config.depth_free_space_margin_m+self.config.depth_free_space_margin_ratio*d*radial
        margin=np.minimum(margin,self.config.depth_free_space_margin_max_m)
        calibrated=profile.bounds(d[valid])[0]
        lower[valid]=np.minimum(calibrated,np.maximum(0,d[valid]-margin[valid]/radial[valid]))
        context.update(grid=self.grid,k=k,pose=np.asarray(pose),lower=lower)
        try:return original_integrate(self,depth,k,pose,*args,**kwargs)
        finally:context.clear()
    def rays(self,origin,points,mask,**kwargs):
        result=original_rays(self,origin,points,mask,**kwargs)
        if context.get("grid") is not self:return result
        hits=set()
        for p in points[np.asarray(mask,dtype=bool)]:
            distance=float(np.linalg.norm(p-origin))
            if distance==0 or not np.isfinite(distance) or distance>kwargs["max_range_m"]:continue
            start=origin+(p-origin)*(max(0.,distance-kwargs["occupied_uncertainty_m"])/distance)
            hits.update(self._ray_voxels(start,p))
        positives=self.occupied_voxels()|self.pending_occupied_voxels()
        assert hits<=positives
        # Full projected voxel footprint must clear the far face, including every pixel.
        certified=projected_free_voxels(positives-hits,self.config.resolution_m,
            context["k"],context["pose"],context["lower"])
        retired=engine.observe(positives,hits,certified,origin,self._integration_frame)
        confirmed=self.occupied_voxels()
        engine.retire_to_unknown(self,retired)
        assert not retired&self.free_voxels()
        records.append(dict(frame=self._integration_frame,positive_voxels=len(positives),
            current_positive=len(hits),certified_free=len(certified),retired=len(retired),
            retired_confirmed=sum(v in confirmed for v in retired)))
        return result
    nav.SpatialNavigator.integrate_depth=integrate
    OccupancyGrid3D.integrate_rays=rays
    try:replay(spec_path)
    finally:save(Path(spec_path).with_name("temporal_revision_audit.json"),dict(
        rule="3 translated full-footprint conservative free observations after last positive",
        retired_to_unknown=True,frames=records,retirements=engine.retired,uses_gt=False))
if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1]!="--worker":
        raise SystemExit("Use scripts/replay_final_monocular.py --bundle FILE --output DIR")
    worker(sys.argv[2])
