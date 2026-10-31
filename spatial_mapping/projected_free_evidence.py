"""Conservative image-footprint evidence for disproving a voxel hypothesis."""
from itertools import product
import numpy as np


def projected_free_voxels(voxels, resolution_m, intrinsics, camera_to_ned, lower_depth):
    """Require every pixel in the complete projected AABB to clear its far face.

    Depth is optical-axis depth. Unknown pixels, partial FOV coverage and
    voxels crossing the camera plane never yield negative evidence. Calibrated
    lower bounds are empirical: this geometric test is not a safety guarantee.
    """
    if not voxels:
        return set()
    ordered = list(voxels)
    coordinates = np.asarray(ordered,dtype=float)
    transform = np.asarray(camera_to_ned,dtype=float)
    lower = np.asarray(lower_depth,dtype=float)
    if (coordinates.shape!=(len(ordered),3) or not np.isfinite(coordinates).all()
            or not (coordinates==np.floor(coordinates)).all()
            or transform.shape!=(4,4) or not np.isfinite(transform).all()
            or lower.ndim!=2 or min(lower.shape)<1
            or not np.isfinite(resolution_m) or resolution_m<=0):
        raise ValueError('invalid projected free-space geometry')
    corners = (coordinates[:,None,:]+np.asarray(list(product([0.,1.],repeat=3))))*resolution_m
    camera = (corners-transform[:3,3])@transform[:3,:3]
    z = camera[:,:,2]
    front = np.all(z>1e-9,axis=1)
    denominator = np.where(z>1e-9,z,1.)
    u = intrinsics.fx*camera[:,:,0]/denominator+intrinsics.cx
    v = intrinsics.fy*camera[:,:,1]/denominator+intrinsics.cy
    left,right = np.floor(u.min(axis=1)).astype(int),np.ceil(u.max(axis=1)).astype(int)
    top,bottom = np.floor(v.min(axis=1)).astype(int),np.ceil(v.max(axis=1)).astype(int)
    inside = front&(left>=0)&(top>=0)&(right<lower.shape[1])&(bottom<lower.shape[0])
    far = z.max(axis=1)
    result = set()
    for i in np.flatnonzero(inside):
        patch = lower[top[i]:bottom[i]+1,left[i]:right[i]+1]
        if np.isfinite(patch).all() and np.min(patch)>far[i]+1e-6:
            result.add(ordered[i])
    return result
