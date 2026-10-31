"""Backend de reprodução do mapa e das métricas na configuração final."""
import json,sys,runpy
from dataclasses import replace,asdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def save(p,x):Path(p).write_text(json.dumps(x,indent=2)+"\n")

def worker(spec_path):
    spec=json.loads(Path(spec_path).read_text())
    import spatial_mapping.navigation as nav
    import spatial_mapping.metrics as metrics
    original=nav.SpatialNavigator;metric=metrics.occupancy_metrics
    instances=[];failures=[]
    class Audited(original):
        def __init__(self,config):
            self.audit_role=len(instances)
            if self.audit_role==0:config=replace(config,**spec["overrides"])
            super().__init__(config);self.ever_positive=set()
            instances.append(self)
        def integrate_depth(self,*args,**kwargs):
            result=super().integrate_depth(*args,**kwargs)
            if self.audit_role<2:self.ever_positive.update(self.grid.occupied_voxels()|self.grid.pending_occupied_voxels())
            return result
        def plan(self,current,*args,**kwargs):
            result=super().plan(current,*args,**kwargs)
            if self.audit_role==0 and not result.success:
                voxel=self.grid.world_to_voxel(current)
                all_blocked=self._inflated_obstacles()
                confirmed=self.grid.inflated_occupied_voxels(self.config.drone_clearance_radius_m,
                    self.config.drone_vertical_clearance_m,include_pending=False)
                neighbours=[(voxel[0]+x,voxel[1]+y,voxel[2]+z) for x in [-1,0,1] for y in [-1,0,1] for z in [-1,0,1] if x or y or z]
                failures.append(dict(frame=self.frames_integrated,reason=result.reason,
                    current_blocked=voxel in all_blocked,current_blocked_by_confirmed=voxel in confirmed,
                    neighbours_blocked_only_by_pending=sum(v in all_blocked and v not in confirmed for v in neighbours),
                    neighbours_known_free=sum(v in self.grid.free_voxels() and v not in all_blocked for v in neighbours)))
            return result
    def audited_metrics(estimated,reference):
        result=metric(estimated,reference)
        ref=reference.occupied_voxels();false_free=estimated.free_voxels()&ref
        confirmed=estimated.occupied_voxels();pending=estimated.pending_occupied_voxels()
        audit=dict(false_free=len(false_free),false_free_never_received_positive=len(false_free-instances[0].ever_positive),
            reference_occupied=len(ref),estimated_confirmed=len(confirmed),estimated_pending=len(pending),
            confirmed_not_in_reference=len(confirmed-ref),pending_not_in_reference=len(pending-ref),
            failed_plans=failures,estimated_config=asdict(instances[0].config),reference_config=asdict(instances[1].config),
            example_false_free=[list(map(int,v)) for v in sorted(false_free)[:20]])
        save(spec["audit"],audit);return result
    nav.SpatialNavigator=Audited;metrics.occupancy_metrics=audited_metrics
    command=spec["command"];sys.argv=[command[1]]+command[2:]
    runpy.run_path(command[1],run_name="__main__")

if __name__=="__main__":
    if len(sys.argv)!=3 or sys.argv[1]!="--worker":
        raise SystemExit("Use scripts/replay_final_monocular.py --bundle FILE --output DIR")
    worker(sys.argv[2])
