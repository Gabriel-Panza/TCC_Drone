"""Integração temporal da configuração monocular final no replay offline."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")
def worker(spec_path):
    import spatial_mapping.navigation as nav
    from spatial_mapping.occupancy import OccupancyGrid3D
    from spatial_mapping.depth_intervals import DepthIntervalProfile
    from spatial_mapping.temporal_revision import (
        TemporalVoxelRevision, conservative_lower_bound, revise_after_rays,
    )
    from scripts.plan_start_entry import worker as replay
    spec=json.loads(Path(spec_path).read_text())
    profile=DepthIntervalProfile(json.loads(Path(spec["temporal_profile"]).read_text()))
    original_integrate=nav.SpatialNavigator.integrate_depth
    original_rays=OccupancyGrid3D.integrate_rays
    context={};engine=TemporalVoxelRevision(3,.75);records=[]
    def integrate(self,depth,k,pose,*args,**kwargs):
        if getattr(self,"audit_role",-1)!=0:
            return original_integrate(self,depth,k,pose,*args,**kwargs)
        lower = conservative_lower_bound(depth, k, self.config, profile)
        context.update(grid=self.grid, k=k, pose=np.asarray(pose), lower=lower)
        try:return original_integrate(self,depth,k,pose,*args,**kwargs)
        finally:context.clear()
    def rays(self,origin,points,mask,**kwargs):
        result=original_rays(self,origin,points,mask,**kwargs)
        if context.get("grid") is not self:return result
        records.append(revise_after_rays(
            self, engine, origin, points, mask, intrinsics=context["k"],
            pose=context["pose"], lower=context["lower"],
            max_range_m=kwargs["max_range_m"],
            occupied_uncertainty_m=kwargs["occupied_uncertainty_m"],
        ))
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
