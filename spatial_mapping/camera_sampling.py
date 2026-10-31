"""Explicit Gazebo pixel centres and matched RGB/depth sampling for new data."""
import numpy as np


def optical_rays(width,height,k,pixel_offset=.5,columns=None,rows=None):
    matrix=np.asarray(k,dtype=float).reshape(3,3)
    if not np.isfinite(matrix).all() or matrix[0,0]<=0 or matrix[1,1]<=0:
        raise ValueError('invalid camera matrix')
    x=np.arange(width) if columns is None else np.asarray(columns)
    y=np.arange(height) if rows is None else np.asarray(rows)
    xx,yy=np.meshgrid(x,y)
    return np.stack([(xx+pixel_offset-matrix[0,2])/matrix[0,0],
                     (yy+pixel_offset-matrix[1,2])/matrix[1,1],np.ones_like(xx,dtype=float)],axis=-1)


def matched_nearest_pair(rgb,depth,k,size,pixel_offset=.5):
    """Sample both arrays at the SAME native pixels and preserve their exact rays.

    For noninteger size ratios the sampled rays need not form a uniform pinhole
    grid. Return source coordinates/rays instead of inventing a scaled matrix.
    Native inputs and historical datasets are never modified.
    """
    if rgb.shape[:2]!=depth.shape or depth.ndim!=2:
        raise ValueError('RGB/depth shapes differ')
    width,height=map(int,size)
    if min(width,height)<1:raise ValueError('invalid output size')
    columns=np.arange(width,dtype=np.int64)*depth.shape[1]//width
    rows=np.arange(height,dtype=np.int64)*depth.shape[0]//height
    return dict(rgb_rgb=rgb[rows[:,None],columns[None,:]].copy(),
                reference_depth_m=depth[rows[:,None],columns[None,:]].copy(),
                source_columns=columns,source_rows=rows,
                rays_optical=optical_rays(depth.shape[1],depth.shape[0],k,pixel_offset=pixel_offset,columns=columns,rows=rows))


def box_depth(origin,directions,minimum,maximum):
    """Ray/AABB intersection parameter; optical rays have Z=1 (not unit norm)."""
    origin=np.asarray(origin,dtype=float)
    directions=np.asarray(directions,dtype=float)
    near=np.full(directions.shape[:-1],-np.inf)
    far=np.full(directions.shape[:-1],np.inf)
    for axis in range(3):
        d=directions[...,axis];moving=np.abs(d)>1e-12
        a=np.divide(minimum[axis]-origin[axis],d,out=np.zeros_like(d),where=moving)
        b=np.divide(maximum[axis]-origin[axis],d,out=np.zeros_like(d),where=moving)
        inside=minimum[axis]<=origin[axis]<=maximum[axis]
        lo=np.where(moving,np.minimum(a,b),-np.inf if inside else np.inf)
        hi=np.where(moving,np.maximum(a,b),np.inf if inside else -np.inf)
        near=np.maximum(near,lo);far=np.minimum(far,hi)
    return np.where((far>=np.maximum(near,0))&(far>0),np.where(near>0,near,far),np.inf)


def reprojection_metrics(source_depth,source_rays,source_pose,target_depth,target_k,target_pose,
                         target_pixel_offset=.5,stride=4,tolerance_m=.05):
    """Compare GT views, reporting occlusion and sampling separately from disagreement."""
    depth=source_depth[::stride,::stride].astype(float)
    rays=source_rays[::stride,::stride]
    valid=np.isfinite(depth)&(depth>=.5)&(depth<=30.)
    pc=rays[valid]*depth[valid,None]
    world=pc@source_pose[:3,:3].T+source_pose[:3,3]
    camera=(world-target_pose[:3,3])@target_pose[:3,:3]
    front=camera[:,2]>.1
    camera=camera[front];z=camera[:,2]
    k=np.asarray(target_k).reshape(3,3)
    u=np.rint(k[0,0]*camera[:,0]/z+k[0,2]-target_pixel_offset).astype(int)
    v=np.rint(k[1,1]*camera[:,1]/z+k[1,2]-target_pixel_offset).astype(int)
    inside=(u>=1)&(v>=1)&(u<target_depth.shape[1]-1)&(v<target_depth.shape[0]-1)
    u,v,z=u[inside],v[inside],z[inside]
    observed=target_depth[v,u].astype(float)
    usable=np.isfinite(observed)&(observed>=.1)
    delta=observed-z
    agree=usable&(np.abs(delta)<=tolerance_m)
    occluded=usable&(delta < -tolerance_m)
    missing=usable&(delta > tolerance_m)
    nearest=np.full(len(z),np.inf)
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            neighbour=target_depth[v+dy,u+dx].astype(float)
            err=np.where(np.isfinite(neighbour)&(neighbour>=.1),np.abs(neighbour-z),np.inf)
            nearest=np.minimum(nearest,err)
    denominator=int(usable.sum())
    ratio=lambda a:float(a.sum()/denominator) if denominator else None
    return dict(source_points=int(valid.sum()),overlap_points=len(z),usable_target_points=denominator,
        agreement_fraction=ratio(agree),occluded_fraction=ratio(occluded),
        disappeared_fraction=ratio(missing),disappeared_unmatched_3x3_fraction=ratio(missing&(nearest>tolerance_m)),
        tolerance_m=tolerance_m,source_stride=stride)
