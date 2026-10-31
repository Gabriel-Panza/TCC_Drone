"""Frozen empirical depth intervals in optical-axis metres, not radial distance."""
import numpy as np


class DepthIntervalProfile:
    def __init__(self, profile):
        if profile.get('schema') != 'depth_interval_v1':
            raise ValueError('unsupported depth interval profile')
        self.edges = np.asarray(profile['predicted_bin_edges_m'], dtype=float)
        self.residuals = np.asarray(profile['residual_bounds_m'], dtype=float)
        if (self.edges.ndim != 1 or len(self.edges) < 2
                or not np.isfinite(self.edges).all() or self.edges[0] <= 0
                or not (np.diff(self.edges) > 0).all()
                or self.residuals.shape != (len(self.edges)-1, 2)
                or not np.isfinite(self.residuals).all()
                or not (self.residuals[:,0] <= 0).all()
                or not (self.residuals[:,1] >= 0).all()):
            raise ValueError('invalid depth interval bounds')

    def bounds(self, optical_depth):
        depth = np.asarray(optical_depth, dtype=float)
        if not np.isfinite(depth).all() or (depth < self.edges[0]).any() or (depth > self.edges[-1]).any():
            raise ValueError('depth outside calibrated domain')
        index = np.clip(np.searchsorted(self.edges, depth, side='right')-1, 0, len(self.residuals)-1)
        lower = np.maximum(self.edges[0], depth+self.residuals[index,0])
        upper = np.minimum(self.edges[-1], depth+self.residuals[index,1])
        return lower, upper


def clip_vertical_segments(near, far, reference_z, band):
    """Clip possible obstacle locations to the physical body-height envelope."""
    near, far = np.asarray(near, dtype=float), np.asarray(far, dtype=float)
    dz = far[:,2]-near[:,2]
    moving = np.abs(dz) > 1e-12
    low, high = np.zeros(len(near)), np.ones(len(near))
    a = np.divide(reference_z-band-near[:,2], dz, out=np.zeros(len(near)), where=moving)
    b = np.divide(reference_z+band-near[:,2], dz, out=np.zeros(len(near)), where=moving)
    low[moving] = np.maximum(0., np.minimum(a,b)[moving])
    high[moving] = np.minimum(1., np.maximum(a,b)[moving])
    selected = ((moving & (low <= high))
                | (~moving & (np.abs(near[:,2]-reference_z) <= band)))
    start = near + low[:,None]*(far-near)
    end = near + high[:,None]*(far-near)
    return selected, np.stack([start,end], axis=1)
