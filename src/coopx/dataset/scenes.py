"""Conservative, GT-assisted clip mining; labels are evaluation-only, not perception.

Targets are the UNION of agents' GT annotations; an ego-only target list
would systematically exclude hidden objects. A visibility candidate combines
actual own-LiDAR support, calibrated camera
projection and segment-vs-GT/LiDAR obstruction. Projection alone is not a
camera-content check: exported overlays require human review before a clip is
certified for a BLOS demo. No SAM/LLM is used. Timestamps are the reader's
nominal 10Hz indices, not measured synchronized camera capture times.

Thresholds are fixed: >=5 in-box returns for visible; <=1 for occluded plus
>=80% of in-image interior sight lines (>=3 of 27) blocked by >=0.5m; >=1s contiguous interval,
<=40m from ego. The gap (2--4 points/conflicting signals) is insufficient
rather than silently hidden. Off-camera targets are always outside_view.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import numpy as np
from PIL import Image, ImageDraw

from coopx.dataset.v2xreal import V2XReal, pose_to_site, transform_points, project_camera, camera_image_size
from coopx.observers.lidar_real import own_points

MIN_VISIBLE_POINTS=5
MAX_HIDDEN_POINTS=1
BLOCKED_FRACTION=.8
OCCLUDER_MARGIN_M=.5
MAX_EGO_DISTANCE_M=40.
MIN_HIDDEN_SECONDS=1.

@dataclass(frozen=True)
class Target:
    gt_id:str
    label:str

@dataclass(frozen=True)
class Visibility:
    label:str
    signals:dict

@dataclass(frozen=True)
class Scene:
    name:str
    scenario:str
    frames:tuple[str,...]
    ego_id:str
    observers:tuple[tuple[str,str,str],...]
    targets:tuple[Target,...]
    visibility:dict
    hidden_interval:dict


def _label(label):
    value=label.lower().replace(' ','').replace('_','')
    if value in ('pedestrian','person'):return 'pedestrian'
    if value in ('cyclist','bicyclerider','bicycle','scooterrider'):return 'cyclist'
    if value in ('car','vehicle'):return 'car'
    return value


def _samples(box):
    offsets=np.array([(x,y,z) for x in (-.8,0,.8) for y in (-.8,0,.8) for z in (-.8,0,.8)])
    return transform_points(np.asarray(box.center)+offsets*np.asarray(box.extent),
                            pose_to_site((*box.location,*box.angle)))


def _box_blocks(origin, targets, box):
    inverse=np.linalg.inv(pose_to_site((*box.location,*box.angle)))
    start=transform_points(np.asarray(origin)[None],inverse)[0]-box.center
    if (np.abs(start)<=np.asarray(box.extent)).all():return np.zeros(len(targets),dtype=bool)
    end=transform_points(targets,inverse)-box.center;delta=end-start
    entry=np.full(len(targets),-math.inf);leave=np.full(len(targets),math.inf)
    for k in range(3):
        parallel=np.abs(delta[:,k])<1e-12
        if abs(start[k])>box.extent[k]:leave[parallel]=-math.inf
        a=np.divide(-box.extent[k]-start[k],delta[:,k],out=np.full(len(targets),-math.inf),where=~parallel)
        b=np.divide(box.extent[k]-start[k],delta[:,k],out=np.full(len(targets),math.inf),where=~parallel)
        entry=np.maximum(entry,np.minimum(a,b));leave=np.minimum(leave,np.maximum(a,b))
    distance=np.linalg.norm(targets-origin,axis=1)
    return ((entry<=leave)&(leave>0)&(entry>.01)&(entry<1-OCCLUDER_MARGIN_M/np.maximum(distance,.01)))


class _View:
    def __init__(self,frame):
        self.frame=frame;self.points=own_points(frame);self.cameras={}
        self.obstacles=[(b,b.center_site(),np.linalg.norm(b.extent)) for b in frame.boxes.values()]

    def camera(self,name):
        if name not in self.cameras:
            uv,depth,valid=project_camera(self.frame.to_camera(self.points,name),self.frame.cameras[name].intrinsic)
            self.cameras[name]=(camera_image_size(self.frame.cameras[name]),uv,depth,valid)
        return self.cameras[name]


def visibility(frame, target, *, _view=None):
    """Provisional per-agent/frame label. GT target may be a box or its ID."""
    box=frame.boxes[str(target)] if isinstance(target,(str,int)) else target
    view=_view or _View(frame)
    center=box.center_site();corners=box.corners_site()
    # AABB cull before rotating the few nearby returns into box coordinates.
    selected=((view.points>=corners.min(0))&(view.points<=corners.max(0))).all(1)
    local=transform_points(view.points[selected],np.linalg.inv(pose_to_site((*box.location,*box.angle))))
    inside=np.zeros(len(view.points),dtype=bool)
    inside[np.flatnonzero(selected)]=(np.abs(local-box.center)<=np.asarray(box.extent)).all(1)
    count=int(inside.sum())
    choices=[]
    for name,cam in frame.cameras.items():
        if not cam.image_path.is_file():continue
        size,_,_,_=view.camera(name);width,height=size
        uv,_,valid=project_camera(frame.to_camera(corners,name),cam.intrinsic)
        if not valid.any():continue
        lo,hi=uv[valid].min(0),uv[valid].max(0)
        area=max(0,min(width,hi[0])-max(0,lo[0]))*max(0,min(height,hi[1])-max(0,lo[1]))
        if area>0:choices.append((float(area),name,lo,hi))
    complete=bool(frame.cameras) and all(c.image_path.is_file() for c in frame.cameras.values())
    signals={'points':count,'obstruction':False,'in_camera':bool(choices),'camera_checked':False,'camera_data_complete':complete}
    if not choices:
        return Visibility('outside_view' if complete else 'insufficient_evidence',signals)
    def inspect(choice):
        area,name,lo,hi=choice
        origin=frame.camera_to_site(name)[:3,3];samples=_samples(box)
        blocked=np.zeros(len(samples),dtype=bool)
        distance=np.linalg.norm(center-origin)
        for other,other_center,radius in view.obstacles:
            if other.object_id==box.object_id or np.linalg.norm(other_center-origin)>distance+radius:continue
            blocked|=_box_blocks(origin,samples,other)
        # Own returns projected into this camera provide a separate positive
        # obstruction hint, never an inference from missing target returns.
        size,uv,depth,valid=view.camera(name)
        sample_uv,sample_depth,sample_valid=project_camera(frame.to_camera(samples,name),frame.cameras[name].intrinsic)
        radius=max(2.,min(6.,float((hi-lo).min()/12)))
        nearby=valid&~inside&(uv[:,0]>=lo[0]-radius)&(uv[:,0]<=hi[0]+radius)&(uv[:,1]>=lo[1]-radius)&(uv[:,1]<=hi[1]+radius)
        pixels,depths=uv[nearby],depth[nearby]
        lidar_blocked=np.zeros(len(samples),dtype=bool)
        for i in np.flatnonzero(sample_valid):
            in_cone=(np.abs(pixels-sample_uv[i])<=radius).all(1)
            lidar_blocked[i]=bool(np.any(depths[in_cone]<sample_depth[i]-OCCLUDER_MARGIN_M))
        width,height=size
        in_view=(sample_valid & (sample_uv[:,0]>=0) & (sample_uv[:,0]<width)
                 & (sample_uv[:,1]>=0) & (sample_uv[:,1]<height))
        n_view=int(in_view.sum())
        # Rays outside the image cannot establish camera occlusion. A tiny
        # projected sliver with <3 interior samples stays insufficient.
        fraction=float((blocked|lidar_blocked)[in_view].mean()) if n_view else 0.
        obstruction=n_view>=3 and fraction>=BLOCKED_FRACTION
        clear=n_view>=3 and not obstruction
        return area,name,lo,hi,obstruction,fraction,float(blocked.mean()),float(lidar_blocked.mean()),n_view,clear
    examined=[inspect(choice) for choice in choices]
    # An agent is hidden only when EVERY camera with image overlap is blocked.
    # The largest projected box alone must not override a clear second camera.
    area,name,lo,hi,_,fraction,gt_fraction,lidar_fraction,n_view,_=min(examined,key=lambda row:(not row[9],-row[0],row[1]))
    obstruction=all(row[4] for row in examined)
    signals.update(obstruction=obstruction,camera=name,blocked_fraction=fraction,
                   gt_blocked_fraction=gt_fraction,lidar_blocked_fraction=lidar_fraction,
                   in_view_samples=n_view,
                   cameras={row[1]:{'blocked_fraction':row[5],'obstruction':row[4],'in_view_samples':row[8]} for row in examined},
                   box_xyxy=[*map(float,lo),*map(float,hi)])
    if count>=MIN_VISIBLE_POINTS and any(row[9] for row in examined):label='visible'
    elif count<=MAX_HIDDEN_POINTS and obstruction and complete:label='occluded'
    else:label='insufficient_evidence'
    return Visibility(label,signals)


def _scene_json(scene):
    return {'name':scene.name,'scenario':scene.scenario,'frames':list(scene.frames),'ego_id':scene.ego_id,
            'observers':[list(v) for v in scene.observers],
            'targets':[{'gt_id':t.gt_id,'label':t.label} for t in scene.targets],
            'visibility':[{'gt_id':g,'agent_id':a,'frame_id':f,'label':v.label,'signals':v.signals}
                          for (g,a,f),v in sorted(scene.visibility.items())],
            'hidden_interval':scene.hidden_interval}


def load_scenes(path='data/scenes/scenes.json'):
    raw=json.loads(Path(path).read_text());rows=raw['scenes'] if isinstance(raw,dict) else raw
    scenes=[]
    for row in rows:
        v={(v['gt_id'],v['agent_id'],v['frame_id']):Visibility(v['label'],v['signals']) for v in row['visibility']}
        scenes.append(Scene(row['name'],row['scenario'],tuple(row['frames']),row['ego_id'],
                            tuple(tuple(o) for o in row['observers']),tuple(Target(**t) for t in row['targets']),
                            v,{k:tuple(value) for k,value in row['hidden_interval'].items()}))
    return scenes


def _overlay(scene,dataset,out):
    target=scene.targets[0];fid=scene.hidden_interval[target.gt_id][0]
    observer=next((o for o in scene.observers if scene.visibility.get((target.gt_id,o[1],fid),Visibility('',{})).label=='visible'),None)
    if observer is None:return
    panels=[]
    for agent,cam in ((scene.ego_id,None),(observer[1],observer[2])):
        frame=dataset.read(scene.scenario,agent,fid)
        vis=scene.visibility[(target.gt_id,agent,fid)]
        cam=cam or vis.signals.get('camera') or sorted(frame.cameras)[0]
        image=frame.cameras[cam].image();pil=Image.fromarray(image);draw=ImageDraw.Draw(pil)
        target_box=frame.boxes.get(target.gt_id)
        if target_box is None:
            for source in dataset.agents(scene.scenario):
                candidate=dataset.read(scene.scenario,source,fid).boxes.get(target.gt_id)
                if candidate is not None:
                    target_box=candidate;break
        if target_box is None:continue
        uv,_,valid=project_camera(frame.to_camera(target_box.corners_site(),cam),frame.cameras[cam].intrinsic)
        if valid.all():
            for a,b in ((0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)):
                draw.line([tuple(uv[a]),tuple(uv[b])],fill='orange' if agent==scene.ego_id else 'lime',width=4)
        draw.rectangle((0,0,pil.width,65),fill='black')
        draw.text((12,12),f'{scene.name} frame {fid} target {target.gt_id} | agent {agent} {cam} | {vis.label}',fill='white')
        draw.text((12,35),f'own points={vis.signals["points"]} blocked={vis.signals.get("blocked_fraction",0):.2f} | human check pending',fill='white')
        pil.thumbnail((960,540));panels.append(pil)
    combined=Image.new('RGB',(sum(p.width for p in panels),max(p.height for p in panels)))
    x=0
    for p in panels:combined.paste(p,(x,0));x+=p.width
    Path(out).mkdir(parents=True,exist_ok=True);combined.save(Path(out)/f'{scene.name}-{fid}.png')


def _interval_candidates(scenario,frames,history,nominal_hz=10.):
    candidates=[]
    frame_index={f:i for i,f in enumerate(frames)}
    short_intervals=[]
    for (ego,tid,label),records in history.items():
        run=[];runs=[]
        for fid in frames:
            value=records.get(fid)
            hidden=value and value[0][ego].label=='occluded' and value[1]
            contiguous=not run or int(fid)==int(run[-1])+1
            if hidden and contiguous:run.append(fid)
            else:
                if run:runs.append(run)
                run=[fid] if hidden else []
        if run:runs.append(run)
        for run in runs:
            duration=(int(run[-1])-int(run[0]))/nominal_hz
            if duration<MIN_HIDDEN_SECONDS:
                short_intervals.append({'ego_id':ego,'gt_id':tid,'label':label,'first_frame':run[0],
                                        'last_frame':run[-1],'seconds':duration})
                continue
            first_visible=next((f for f in frames[frame_index[run[-1]]+1:]
                                if f in records and records[f][0][ego].label=='visible'),None)
            last=frame_index[first_visible] if first_visible else frame_index[run[-1]]
            chosen=frames[max(0,frame_index[run[0]]-3):min(len(frames),last+4)]
            labels={(tid,a,f):v for f in chosen if f in records for a,v in records[f][0].items()}
            observers=tuple(sorted({o for f in run for o in records[f][1]}))
            candidates.append((duration,first_visible is not None,Scene('',scenario,tuple(chosen),ego,observers,
                                (Target(tid,label),),labels,{tid:(run[0],first_visible)})))
    return candidates,short_intervals


def _mine_scenario(args):
    dataset_root,scenario,*artifacts=args
    dataset=V2XReal(dataset_root);candidates=[]
    agents=dataset.agents(scenario);egos=[a for a in agents if not a.startswith('-')]
    infrastructure=[a for a in agents if a.startswith('-')]
    if not egos or not infrastructure:
        return [],{'scenario':scenario,'frames':0,'reason':'no vehicle/infrastructure pair'}
    frames=dataset.aligned_frame_ids(scenario,agents);history={}
    for fid in frames:
        views={a:_View(dataset.read(scenario,a,fid)) for a in agents}
        union={}
        for agent in agents:
            for tid,target in views[agent].frame.boxes.items():union.setdefault(tid,(target,agent))
        measured={}
        for ego in egos:
            frame=views[ego].frame;ego_xy=frame.lidar_to_site[:2,3]
            targets=[b for b,source in union.values() if _label(b.label) in ('pedestrian','cyclist','car')
                     and np.linalg.norm(b.center_site()[:2]-ego_xy)<=MAX_EGO_DISTANCE_M
                     and (_label(b.label)!='car' or np.linalg.norm(b.center_site()[:2]-ego_xy)>2)]
            for target in targets:
                key=(ego,target.object_id,_label(target.label));labels={};observers=[]
                for agent in (ego,*infrastructure):
                    other=views[agent].frame
                    cache_key=(agent,target.object_id)
                    if cache_key not in measured:
                        box=other.boxes.get(target.object_id,target)
                        value=visibility(other,box,_view=views[agent])
                        value.signals['gt_origin_agent']=agent if target.object_id in other.boxes else union[target.object_id][1]
                        value.signals['gt_shared']=target.object_id not in other.boxes
                        measured[cache_key]=value
                    label=measured[cache_key];labels[agent]=label
                    if agent in infrastructure and label.label=='visible':
                        observers.append((f'rsu{agent}-{label.signals["camera"]}',agent,label.signals['camera']))
                history.setdefault(key,{})[fid]=(labels,tuple(observers))
    candidates,short_intervals=_interval_candidates(scenario,frames,history,dataset.nominal_hz)
    # Preserve every measured candidate label, including rejected/short cases,
    # so an empty shortlist is auditable rather than erasing the evidence.
    from collections import Counter
    tallies=Counter()
    if artifacts:
        directory=Path(artifacts[0]);directory.mkdir(parents=True,exist_ok=True)
        with (directory/(scenario+'.jsonl')).open('w') as handle:
            for (ego,tid,label),records in sorted(history.items()):
                for fid,(labels,observers) in sorted(records.items()):
                    for agent,v in labels.items():tallies[f'{label}/{agent}/{v.label}']+=1
                    row={'scenario':scenario,'ego_id':ego,'gt_id':tid,'label':label,'frame_id':fid,
                         'observers':observers,'visibility':{a:{'label':v.label,'signals':v.signals} for a,v in labels.items()}}
                    handle.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
    short_intervals.sort(key=lambda r:(r['label']!='car',r['seconds'],r['gt_id']),reverse=True)
    return candidates,{'scenario':scenario,'frames':len(frames),'targets':len(history),
                       'visibility_counts':dict(sorted(tallies.items())),
                       'short_intervals':short_intervals[:20]}


def mine_scenes(dataset_root,*,out='data/scenes/scenes.json'):
    from concurrent.futures import ProcessPoolExecutor
    dataset=V2XReal(dataset_root);candidates=[];counts={};scanned=[]
    scenarios=dataset.scenarios()
    # Native model clients are not imported or called in these workers. Only
    # the parent writes files; ordering is deterministic across worker counts.
    with ProcessPoolExecutor(max_workers=max(1,min(4,len(scenarios)))) as pool:
        for found,stats in pool.map(_mine_scenario,[(str(dataset_root),s,str(Path(out).parent/'labels')) for s in scenarios]):
            candidates.extend(found);scanned.append(stats)
            print(f"mined {stats['scenario']}: {stats['frames']} frames",flush=True)
    return _write_mining_result(dataset,candidates,scanned,out)


def _write_mining_result(dataset,candidates,scanned,out):
    counts={}
    candidates.sort(key=lambda r:(r[2].targets[0].label!='car',r[1],r[0],r[2].scenario,r[2].ego_id),reverse=True)
    for _,_,scene in candidates:counts[scene.targets[0].label]=counts.get(scene.targets[0].label,0)+1
    selected=[];per_class={}
    for duration,transition,scene in candidates:
        kind='car' if scene.targets[0].label=='car' else 'vulnerable'
        if per_class.get(kind,0)>=(3 if kind=='car' else 5):continue
        per_class[kind]=per_class.get(kind,0)+1
        name=f'blos_{scene.targets[0].label}_{per_class[kind]:03d}'
        selected.append(Scene(name,scene.scenario,scene.frames,scene.ego_id,scene.observers,scene.targets,scene.visibility,scene.hidden_interval))
    path=Path(out);path.parent.mkdir(parents=True,exist_ok=True)
    payload={'schema':'coopx.scenes/1','mining':{'scenarios':scanned,'qualifying_intervals':counts,
              'min_hidden_seconds':MIN_HIDDEN_SECONDS,'visibility_rule':'union-GT, any-clear-camera/all-blocked-cameras',
              'human_review':'pending','timestamp_basis':'nominal frame index at 10Hz'},
             'scenes':[_scene_json(s) for s in selected]}
    path.write_text(json.dumps(payload,allow_nan=False,indent=2)+'\n')
    for scene in selected:_overlay(scene,dataset,path.parent/'overlays')
    return selected
