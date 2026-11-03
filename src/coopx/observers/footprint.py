"""Camera nominal footprint from own range support and a dense ray fan.

Range rule fixed before looking at target IDs: group camera-visible own LiDAR
returns 0.2--2.2m above an own-cloud ground plane into 0.6m XY tiles. Tiles need
>=5 returns (the camera lift's minimum); declare the 95th percentile of their
horizontal ranges, rounded DOWN to whole metres, bounded by 100m. This avoids
a single distant reflector setting the range. It demonstrates range support
for pedestrian-scale footprints, NOT semantic pedestrian recall or uniform
coverage. No GT, target labels, pedestrian 53, or target-specific range enters
this calculation. No supporting plane/tiles -> range 0 and an empty polygon.

The polygon is nominal camera frustum intersected with that ground plane and
range disk. Occlusion and sparse LiDAR holes remain unknown, not observed-free.
65x33 image rays include the bottom and both sides; an inscribed convex hull
(max 64 vertices, Capability's limit) never expands their planar support.
"""

import math
import numpy as np
from coopx.dataset.v2xreal import camera_image_size, project_camera
from coopx.observers.lidar_real import own_points, ground_model

SUPPORT_TILE_M=.6       # pedestrian-sized horizontal support bin
MIN_SUPPORT_POINTS=5   # unchanged camera lift minimum
RANGE_QUANTILE=.95     # robust tail of ALL supported tiles, not a chosen target
RANGE_LIMIT_M=100.     # computational/domain cap; observed data may only reduce it


def demonstrated_range_m(frame,camera_name,*,min_support=MIN_SUPPORT_POINTS):
    if type(min_support) is not int or min_support<1:raise ValueError('invalid minimum support')
    points=own_points(frame);plane=ground_model(frame,points)
    if plane is None:return 0.
    uv,_,valid=project_camera(frame.to_camera(points,camera_name),frame.cameras[camera_name].intrinsic)
    width,height=camera_image_size(frame.cameras[camera_name])
    above=points[:,2]-(points[:,:2]@plane[:2]+plane[2])
    selected=valid&(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)&(above>=.2)&(above<=2.2)
    points=points[selected]
    if not len(points):return 0.
    bins=np.floor(points[:,:2]/SUPPORT_TILE_M).astype(np.int64)
    _,group,counts=np.unique(bins,axis=0,return_inverse=True,return_counts=True)
    points=points[counts[group]>=min_support]
    if not len(points):return 0.
    # Equal tile weight, so a heavily sampled wall doesn't set the percentile.
    bins=np.floor(points[:,:2]/SUPPORT_TILE_M).astype(np.int64)
    _,group=np.unique(bins,axis=0,return_inverse=True)
    origin=frame.camera_to_site(camera_name)[:2,3]
    ranges=np.linalg.norm(points[:,:2]-origin,axis=1)
    farthest=np.full(group.max()+1,math.inf)
    np.minimum.at(farthest,group,ranges) # every support tile is reached at least this far
    return float(max(0,math.floor(min(RANGE_LIMIT_M,float(np.quantile(farthest,RANGE_QUANTILE))))))


def _hull(points):
    points=sorted(set(tuple(map(float,p)) for p in points))
    if len(points)<3:return ()
    def cross(a,b,c):return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    halves=[]
    for sequence in (points,list(reversed(points))):
        half=[]
        for p in sequence:
            while len(half)>=2 and cross(half[-2],half[-1],p)<=1e-10:half.pop()
            half.append(p)
        halves.append(half[:-1])
    hull=halves[0]+halves[1]
    # Drop the least-area corner, retaining the sharp near-frustum corners.
    # Uniform vertex subsampling would keep the dense far arc and can discard
    # BOTH near corners, accidentally shrinking a fan to an arc segment.
    while len(hull)>64:
        areas=[abs(cross(hull[i-1],hull[i],hull[(i+1)%len(hull)])) for i in range(len(hull))]
        hull.pop(int(np.argmin(areas)))
    return tuple(hull) if len(hull)>=3 else ()


def camera_footprint(frame,camera_name,*,range_m=None):
    demonstrated=demonstrated_range_m(frame,camera_name)
    if range_m is not None and (type(range_m) not in (int,float) or not math.isfinite(range_m) or range_m<=0):
        raise ValueError('range_m must be finite and positive')
    distance=demonstrated if range_m is None else min(float(range_m),demonstrated)
    if distance<=0:return ()
    points=own_points(frame);plane=ground_model(frame,points)
    if plane is None:return ()
    cam=frame.cameras[camera_name];width,height=camera_image_size(cam)
    transform=frame.camera_to_site(camera_name);origin=transform[:3,3]
    pixels=np.array([(u,v,1.) for v in np.linspace(0,height-1,33) for u in np.linspace(0,width-1,65)])
    rays=(pixels@np.linalg.inv(cam.intrinsic).T)@transform[:3,:3].T
    denominator=rays[:,2]-rays[:,:2]@plane[:2]
    numerator=plane[:2]@origin[:2]+plane[2]-origin[2]
    valid=np.abs(denominator)>1e-8
    t=np.divide(numerator,denominator,out=np.full(len(rays),-1.),where=valid)
    hits=origin+rays[t>0]*t[t>0,None]
    if not len(hits):return ()
    delta=hits[:,:2]-origin[:2];ranges=np.linalg.norm(delta,axis=1)
    hits[:,:2]=origin[:2]+delta*np.minimum(1.,distance/np.maximum(ranges,1e-9))[:,None]
    hits[:,2]=hits[:,:2]@plane[:2]+plane[2]
    uv,_,valid=project_camera(frame.to_camera(hits,camera_name),cam.intrinsic)
    valid &= (uv[:,0]>=-1e-6)&(uv[:,0]<=width-1+1e-6)&(uv[:,1]>=-1e-6)&(uv[:,1]<=height-1+1e-6)
    return _hull(hits[valid,:2])
