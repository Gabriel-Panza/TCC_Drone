"""Static-scene interval hypotheses, separate from positive occupancy evidence."""
import numpy as np


class IntervalHypotheses:
    def __init__(self, baseline_m=.75, required_free_views=2):
        if (not np.isfinite(baseline_m) or baseline_m <= 0 or required_free_views < 2
                or int(required_free_views)!=required_free_views):
            raise ValueError('interval disproof needs a positive baseline and at least two views')
        self.baseline_m = float(baseline_m)
        self.required_free_views = int(required_free_views)
        self.possible = set()
        self.disproved = set()
        self.birth = {}
        self.free_views = {}
        self.narrow_views = {}
        self.cleared_by_multiview = 0

    def record_narrow(self, voxel, origin, frame, window):
        views = [v for v in self.narrow_views.get(voxel,[]) if v[3] >= frame-window+1]
        if len(views)<2 and all(np.linalg.norm(np.asarray(origin)-v[:3]) >= self.baseline_m-1e-9 for v in views):
            views.append((*map(float,origin),int(frame)))
        self.narrow_views[voxel] = views

    def add_distinct(self, views, origin):
        origin = tuple(float(v) for v in origin)
        if all(np.linalg.norm(np.asarray(origin)-p) >= self.baseline_m-1e-9 for p in views):
            views.append(origin)
            return True
        return False

    def observe(self, candidates, free, origin, atomic):
        """Broad possibilities are weaker than repeated spatially separated free evidence.

        A free ray must end BEFORE its calibrated lower depth bound. This is
        empirical evidence, not a statistical independence or safety guarantee.
        Time alone never deletes a hypothesis. Atomic evidence always wins.
        """
        origin = np.asarray(origin,dtype=float)
        if origin.shape!=(3,) or not np.isfinite(origin).all():
            raise ValueError('invalid observation origin')
        for voxel in atomic:
            self.possible.discard(voxel)
            self.disproved.discard(voxel)
            self.birth.pop(voxel,None)
            self.free_views.pop(voxel,None)
        for voxel in candidates-atomic-self.disproved:
            self.possible.add(voxel)
            self.birth.setdefault(voxel,tuple(float(v) for v in origin))
        removed = 0
        for voxel in free & (self.possible - atomic):
            if np.linalg.norm(np.asarray(origin)-self.birth[voxel]) < self.baseline_m-1e-9:
                continue
            views = self.free_views.setdefault(voxel,[])
            self.add_distinct(views,origin)
            if len(views) >= self.required_free_views:
                self.possible.remove(voxel)
                self.disproved.add(voxel)
                self.birth.pop(voxel,None)
                self.free_views.pop(voxel,None)
                removed += 1
        self.cleared_by_multiview += removed
        return removed

    def ego_free(self, voxel):
        self.possible.discard(voxel)
        self.disproved.add(voxel)
        self.birth.pop(voxel,None)
        self.free_views.pop(voxel,None)
        self.narrow_views.pop(voxel,None)

    def snapshot_arrays(self):
        return dict(
            interval_possible=np.asarray(list(self.possible),dtype=np.int64).reshape(-1,3),
            interval_disproved=np.asarray(list(self.disproved),dtype=np.int64).reshape(-1,3),
            interval_birth=np.asarray([(*v,*p) for v,p in self.birth.items()],dtype=float).reshape(-1,6),
            interval_free_views=np.asarray([(*v,*p) for v,ps in self.free_views.items() for p in ps],dtype=float).reshape(-1,6),
            interval_narrow_views=np.asarray([(*v,*p) for v,ps in self.narrow_views.items() for p in ps],dtype=float).reshape(-1,7),
            interval_cleared_by_multiview=np.asarray(self.cleared_by_multiview,dtype=np.int64))

    def load_arrays(self, data, required=False):
        names = ['interval_possible','interval_disproved','interval_birth','interval_free_views',
                 'interval_narrow_views','interval_cleared_by_multiview']
        if not all(k in data for k in names):
            if required or any(k in data for k in names):
                raise ValueError('incomplete interval hypothesis snapshot')
            return
        values = {}
        for name,width in zip(names[:5],[3,3,6,6,7]):
            a = np.asarray(data[name],dtype=float)
            if (a.ndim!=2 or a.shape[1]!=width or not np.isfinite(a).all()
                    or not (a[:,:3]==np.floor(a[:,:3])).all()):
                raise ValueError('invalid interval snapshot array: '+name)
            values[name] = a
        self.possible = {tuple(map(int,v)) for v in values['interval_possible']}
        self.disproved = {tuple(map(int,v)) for v in values['interval_disproved']}
        self.birth = {tuple(map(int,v[:3])):tuple(v[3:]) for v in values['interval_birth']}
        self.free_views,self.narrow_views = {},{}
        for name,target in [('interval_free_views',self.free_views),('interval_narrow_views',self.narrow_views)]:
            for row in values[name]:
                target.setdefault(tuple(map(int,row[:3])),[]).append(tuple(row[3:]))
        if (self.possible & self.disproved or set(self.birth)!=self.possible
                or not set(self.free_views)<=self.possible
                or any(len(v)>=self.required_free_views for v in self.free_views.values())):
            raise ValueError('inconsistent interval hypothesis snapshot')
        for voxel,views in self.free_views.items():
            if (any(np.linalg.norm(np.asarray(v)-self.birth[voxel])<self.baseline_m-1e-9 for v in views)
                    or any(np.linalg.norm(np.asarray(v)-w)<self.baseline_m-1e-9
                           for i,v in enumerate(views) for w in views[:i])):
                raise ValueError('non-separated free views in snapshot')
        for views in self.narrow_views.values():
            if (len(views)>2 or any(v[3]<0 or v[3]!=int(v[3]) for v in views)
                    or any(np.linalg.norm(np.asarray(v[:3])-w[:3])<self.baseline_m-1e-9
                           for i,v in enumerate(views) for w in views[:i])):
                raise ValueError('invalid narrow views in snapshot')
        self.cleared_by_multiview = int(np.asarray(data[names[-1]]).item())
