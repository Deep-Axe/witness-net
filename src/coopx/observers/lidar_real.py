"""Own-LiDAR geometric objects and conservative ray evidence in site_map.

The height band is ABSOLUTE site Z, as required by CoverageContent, not height
above a guessed flat road. Ground removal uses a robust own-cloud plane;
without one, no object clusters are inferred. Rays themselves remain valid
without a fitted plane. A ray only certifies the portion physically inside the
stated height band. No-return directions and the space behind a return stay
unknown. Occupied endpoints win over any crossing free claim in this scan.

A traversed grid cell is a beam-sampled free claim, not an assertion that every
height/point in that voxel was scanned. min_size_m states nominal detectable
size, not a calibrated probability of detection. Multi-height/beam-density
certification and temporal motion compensation remain separate work.
"""

import math
from collections import defaultdict, deque
import numpy as np

from coopx.coverage import CoverageContent, CoverageMessage, GridSpec
from coopx.envelope import ObservationMessage
from coopx.dataset.v2xreal import transform_points
from coopx.observers.camera_lift import _fit_plane

GROUND_MARGIN_M = .15  # reject near-road returns from object clustering
CLUSTER_CELL_M = .6    # pedestrian-scale XY connectivity, no learned classes
MIN_CLUSTER_POINTS = 5 # same minimum range support as the camera lift


def own_points(frame):
    local = frame.lidar()[:, :3]
    local = local[np.isfinite(local).all(1) & (np.linalg.norm(local,axis=1) > .001)]
    return transform_points(local, frame.lidar_to_site)


def ground_model(frame, points=None):
    """Plane z=ax+by+c from own returns within 60m, ignoring padded zeros."""
    points = own_points(frame) if points is None else points
    origin = frame.lidar_to_site[:3,3]
    candidate = points[(np.linalg.norm(points[:,:2]-origin[:2],axis=1)<60)
                       & (points[:,2]<origin[2]-.3)]
    result = _fit_plane(candidate)
    return None if result is None else result[0]


def ray_cells(start, end, spec):
    """Exact positive-length grid crossings (Amanatides-Woo style), ROI clipped.

    Boundary-only touches do not certify an adjacent cell. Runtime depends on
    the finite grid, even if a sensor lies far outside it.
    """
    a,b=np.asarray(start,dtype=float),np.asarray(end,dtype=float)
    delta=b-a
    lo=np.array(spec.bounds_xy[:2]);hi=np.array(spec.bounds_xy[2:])
    t0,t1=0.,1.
    for k in (0,1):
        if abs(delta[k])<1e-12:
            if not lo[k] <= a[k] < hi[k]:return ()
        else:
            u,v=(lo[k]-a[k])/delta[k],(hi[k]-a[k])/delta[k]
            t0,t1=max(t0,min(u,v)),min(t1,max(u,v))
    if t0>=t1:return ()
    a=a+delta*t0;b=np.asarray(start)+delta*t1
    # Probe inside segment, not on ambiguous upper ROI/grid boundaries.
    a=np.nextafter(a,b);b=np.nextafter(b,a)
    ix,iy=spec.cell_at(float(a[0]),float(a[1]));ex,ey=spec.cell_at(float(b[0]),float(b[1]))
    d=b-a;steps=np.sign(d).astype(int)
    cross=[]
    for k,idx in enumerate((ix,iy)):
        boundary=lo[k]+(idx+(1 if steps[k]>0 else 0))*spec.cell_m
        cross.append((boundary-a[k])/d[k] if steps[k] else math.inf)
    increment=np.divide(spec.cell_m,np.abs(d),out=np.full(2,math.inf),where=np.abs(d)>0)
    cells=[]
    for _ in range(spec.nx+spec.ny+2):
        if spec.contains((ix,iy)):cells.append((ix,iy))
        if (ix,iy)==(ex,ey):break
        if abs(cross[0]-cross[1])<1e-12:
            ix+=int(steps[0]);iy+=int(steps[1]);cross[0]+=increment[0];cross[1]+=increment[1]
        elif cross[0]<cross[1]:ix+=int(steps[0]);cross[0]+=increment[0]
        else:iy+=int(steps[1]);cross[1]+=increment[1]
    return tuple(cells)


def _ray_cells_many(starts, ends, spec):
    """Vectorized exact DDA; the same crossings as ray_cells, no sector fill."""
    a,b=np.asarray(starts,dtype=float),np.asarray(ends,dtype=float)
    if not len(a):return set()
    delta=b-a;lo=np.array(spec.bounds_xy[:2]);hi=np.array(spec.bounds_xy[2:])
    t0=np.zeros(len(a));t1=np.ones(len(a));valid=np.linalg.norm(delta,axis=1)>1e-12
    for k in (0,1):
        moving=np.abs(delta[:,k])>=1e-12
        valid &= moving | ((a[:,k]>=lo[k]) & (a[:,k]<hi[k]))
        u=np.divide(lo[k]-a[:,k],delta[:,k],out=np.full(len(a),-math.inf),where=moving)
        v=np.divide(hi[k]-a[:,k],delta[:,k],out=np.full(len(a),math.inf),where=moving)
        t0=np.maximum(t0,np.minimum(u,v));t1=np.minimum(t1,np.maximum(u,v))
    valid &= t0<t1
    start=a[valid]+delta[valid]*t0[valid,None];end=a[valid]+delta[valid]*t1[valid,None]
    if not len(start):return set()
    start=np.nextafter(start,end);end=np.nextafter(end,start)
    cells=np.floor((start-lo)/spec.cell_m).astype(np.int64)
    goal=np.floor((end-lo)/spec.cell_m).astype(np.int64)
    d=end-start;steps=np.sign(d).astype(np.int64)
    boundary=lo+(cells+(steps>0))*spec.cell_m
    cross=np.divide(boundary-start,d,out=np.full_like(d,math.inf),where=d!=0)
    inc=np.divide(spec.cell_m,np.abs(d),out=np.full_like(d,math.inf),where=d!=0)
    active=np.ones(len(start),dtype=bool);bitmap=np.zeros((spec.ny,spec.nx),dtype=bool)
    for _ in range(spec.nx+spec.ny+2):
        inside=active&(cells[:,0]>=0)&(cells[:,0]<spec.nx)&(cells[:,1]>=0)&(cells[:,1]<spec.ny)
        bitmap[cells[inside,1],cells[inside,0]]=True
        active &= (cells!=goal).any(axis=1)
        if not active.any():break
        tie=np.isclose(cross[:,0],cross[:,1],atol=1e-12,rtol=0)
        x=active&((cross[:,0]<cross[:,1])|tie)
        y=active&((cross[:,1]<cross[:,0])|tie)
        cells[x,0]+=steps[x,0];cross[x,0]+=inc[x,0]
        cells[y,1]+=steps[y,1];cross[y,1]+=inc[y,1]
    ys,xs=bitmap.nonzero()
    return set(zip(map(int,xs),map(int,ys)))


def lidar_coverage(frame, spec: GridSpec, *, capture_us, window_us,
                   height_band_m=(.2,2.), max_range_m, min_size_m=.3):
    if (not isinstance(spec,GridSpec) or not math.isfinite(max_range_m) or max_range_m<=0
            or type(capture_us) is not int or capture_us<0
            or type(window_us) is not tuple or len(window_us)!=2 or window_us[1]!=capture_us
            or len(height_band_m)!=2 or not all(math.isfinite(x) for x in height_band_m)
            or height_band_m[0]>=height_band_m[1]):
        raise ValueError('invalid coverage grid, range, height or capture window')
    points=own_points(frame);origin=frame.lidar_to_site[:3,3]
    lower,upper=height_band_m;plane=ground_model(frame,points)
    offset=points-origin;distance=np.linalg.norm(offset[:,:2],axis=1)
    nonzero=distance>1e-6
    points,offset,distance=points[nonzero],offset[nonzero],distance[nonzero]
    hit_cells=set();occupied=set()
    in_range=distance<=max_range_m
    endpoints=points[in_range]
    if len(endpoints):
        ij=np.floor((endpoints[:,:2]-[spec.origin_x,spec.origin_y])/spec.cell_m).astype(np.int64)
        inside=(ij[:,0]>=0)&(ij[:,0]<spec.nx)&(ij[:,1]>=0)&(ij[:,1]<spec.ny)
        hit_cells={tuple(map(int,c)) for c in ij[inside]}
        non_ground=np.ones(len(endpoints),dtype=bool) if plane is None else (
            endpoints[:,2]-(endpoints[:,:2]@plane[:2]+plane[2])>GROUND_MARGIN_M)
        occupied={tuple(map(int,c)) for c in ij[inside&non_ground&(endpoints[:,2]>=lower)&(endpoints[:,2]<=upper)]}
    t0=np.zeros(len(points));t1=np.minimum(1.,max_range_m/distance)
    moving=np.abs(offset[:,2])>=1e-12
    u=np.divide(lower-origin[2],offset[:,2],out=np.full(len(points),-math.inf),where=moving)
    v=np.divide(upper-origin[2],offset[:,2],out=np.full(len(points),math.inf),where=moving)
    t0=np.maximum(t0,np.minimum(u,v));t1=np.minimum(t1,np.maximum(u,v))
    valid=(t0<t1)&(moving|((lower<=origin[2])&(origin[2]<=upper)))
    starts=origin[:2]+offset[valid,:2]*t0[valid,None]
    ends=origin[:2]+offset[valid,:2]*t1[valid,None]
    free=_ray_cells_many(starts,ends,spec)-hit_cells
    # Out-of-band returns can belong to an object extending into the band;
    # their BEV cells remain unknown rather than being declared empty.
    return CoverageContent(spec,window_us,'lidar',tuple(height_band_m),min_size_m,
                           frozenset(free|occupied),frozenset(free),frozenset(occupied))


REL_BAND_M = (.2, 1.)   # above ground: any standing person blocks a ray here
SLICE_M = .25           # ground-height slice per message on sloped sites


def lidar_coverage_sliced(frame, spec: GridSpec, *, capture_us, window_us, max_range_m,
                          rel_band_m=REL_BAND_M, slice_m=SLICE_M, min_size_m=.3):
    """Ground-relative free space, published as absolute-band slices.

    One absolute height band cannot be right across sloped ground (V2X-Real
    scenario 2023-03-17-16-10-12_1_1 falls ~1.6 m over 35 m). Rays are clipped
    once in height above this unit's own fitted ground plane (exact for straight
    rays over a planar ground). Cells are then grouped by their ground height
    into `slice_m` slices; each slice becomes one CoverageContent whose absolute
    band (lo+slice+rel_lo, lo+rel_hi) lies INSIDE the band every one of its
    cells was actually observed in, so no message claims more than was seen.
    Without a ground plane: no free claims (silence is not free space).
    """
    if (not isinstance(spec, GridSpec) or not math.isfinite(max_range_m) or max_range_m <= 0
            or type(capture_us) is not int or window_us[1] != capture_us
            or not 0 < slice_m < rel_band_m[1] - rel_band_m[0]):
        raise ValueError('invalid sliced coverage request')
    points = own_points(frame); origin = frame.lidar_to_site[:3, 3]
    plane = ground_model(frame, points)
    if plane is None:
        return []
    ground = lambda xy: xy @ plane[:2] + plane[2]
    rel = points.copy(); rel[:, 2] = points[:, 2] - ground(points[:, :2])
    rel_origin = origin.copy(); rel_origin[2] = origin[2] - float(ground(origin[:2]))
    offset = rel - rel_origin; distance = np.linalg.norm(offset[:, :2], axis=1)
    keep = distance > 1e-6
    rel, offset, distance, points = rel[keep], offset[keep], distance[keep], points[keep]
    lower, upper = rel_band_m
    in_range = distance <= max_range_m
    ij = np.floor((points[in_range, :2] - [spec.origin_x, spec.origin_y]) / spec.cell_m).astype(np.int64)
    inside = (ij[:, 0] >= 0) & (ij[:, 0] < spec.nx) & (ij[:, 1] >= 0) & (ij[:, 1] < spec.ny)
    hit_cells = {tuple(map(int, c)) for c in ij[inside]}
    h = rel[in_range, 2]
    occupied = {tuple(map(int, c)) for c in ij[inside & (h > GROUND_MARGIN_M) & (h >= lower) & (h <= upper)]}
    t0 = np.zeros(len(rel)); t1 = np.minimum(1., max_range_m / distance)
    moving = np.abs(offset[:, 2]) >= 1e-12
    u = np.divide(lower - rel_origin[2], offset[:, 2], out=np.full(len(rel), -math.inf), where=moving)
    v = np.divide(upper - rel_origin[2], offset[:, 2], out=np.full(len(rel), math.inf), where=moving)
    t0 = np.maximum(t0, np.minimum(u, v)); t1 = np.minimum(t1, np.maximum(u, v))
    valid = (t0 < t1) & (moving | ((lower <= rel_origin[2]) & (rel_origin[2] <= upper)))
    starts = rel_origin[:2] + offset[valid, :2] * t0[valid, None]
    ends = rel_origin[:2] + offset[valid, :2] * t1[valid, None]
    free = _ray_cells_many(starts, ends, spec) - hit_cells
    observed = free | occupied
    groups = defaultdict(lambda: (set(), set()))
    for cell in observed:
        g = float(ground(np.array(spec.cell_centre(cell))))
        groups[math.floor(g / slice_m)][0 if cell in free else 1].add(cell)
    out = []
    for k in sorted(groups):
        f_cells, o_cells = groups[k]
        lo = k * slice_m
        band = (lo + slice_m + lower, lo + upper)
        out.append(CoverageContent(spec, window_us, 'lidar', band, min_size_m,
                                   frozenset(f_cells | o_cells), frozenset(f_cells), frozenset(o_cells)))
    return out


def ground_band(frame, xy, *, rel_band_m=REL_BAND_M, slice_m=SLICE_M):
    """The absolute band a slice message uses for the cell containing xy."""
    plane = ground_model(frame)
    if plane is None:
        return None
    g = float(plane[0] * xy[0] + plane[1] * xy[1] + plane[2])
    lo = math.floor(g / slice_m) * slice_m
    return (lo + slice_m + rel_band_m[0], lo + rel_band_m[1])


def lidar_objects(frame, *, height_band_m=(.2,2.)):
    points=own_points(frame);plane=ground_model(frame,points)
    if plane is None:return []
    height=points[:,2]-(points[:,:2]@plane[:2]+plane[2])
    points=points[(height>GROUND_MARGIN_M)&(points[:,2]>=height_band_m[0])&(points[:,2]<=height_band_m[1])]
    cells=defaultdict(list)
    for index,cell in enumerate(np.floor(points[:,:2]/CLUSTER_CELL_M).astype(int)):
        cells[tuple(cell)].append(index)
    results=[];remaining=set(cells)
    while remaining:
        seed=min(remaining);remaining.remove(seed);queue=deque([seed]);indices=[]
        while queue:
            x,y=queue.popleft();indices.extend(cells[(x,y)])
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    cell=(x+dx,y+dy)
                    if cell in remaining:remaining.remove(cell);queue.append(cell)
        if len(indices)<MIN_CLUSTER_POINTS:continue
        cluster=points[indices];lo,hi=np.quantile(cluster,[.05,.95],axis=0)
        results.append({'xyz':tuple(map(float,np.median(cluster,axis=0))),
                        'dimensions_m':tuple(map(float,np.maximum(.05,hi-lo))),
                        'cls':'UNKNOWN','n_points':len(indices)})
    return results


def range_contradicts(store, coverage_map, detection_ref, coverage_ref, now_us):
    """Proof over admitted originals; shares Assessor's range policy exactly."""
    from coopx.assess import Assessor
    detection,coverage=store.get(detection_ref),store.get(coverage_ref)
    if not isinstance(detection,ObservationMessage) or not isinstance(coverage,CoverageMessage):return False
    if coverage.content.spec!=coverage_map.spec:return False
    return Assessor(store,coverage_map).range_contradiction(detection,coverage_ref,now_us)
