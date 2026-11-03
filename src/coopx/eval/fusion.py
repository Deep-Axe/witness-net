"""Object-report fusion, Muse off, same admitted inputs for every policy.

Geo is a deterministic object-level baseline inspired by ROBOSAC consensus
(https://arxiv.org/abs/2303.09495) and CAD collaborative anomaly detection
(https://www.usenix.org/conference/usenixsecurity24/presentation/zhang-qingzhao),
NOT a reproduction of either feature-level method. For each report, count
physical observers among class/anchor/frame-compatible reports within 1.5m
(pedestrians), 3m (vehicles), and 300ms; two are confirmation. At least half
its footprint in eligible recent range free space contradicts it, taking
precedence over consensus. Unverified reports still create bounded caution.
Geometric grouping uses configured physical bodies, not lineage/common-cause
proofs. Witness uses authenticated roots and the existing Assessor, retains
contested evidence, and protects every local detection. No model is called.
"""
from dataclasses import dataclass
import math
from coopx.envelope import ObservationMessage, evidence_ref
from coopx.coverage import evidence_eligible
from coopx.domain import ObjectClass

FUSION_MODES=('ego','union','geo','witness')
PED_GATE_M=1.5  # project plan: conservative cross-view pedestrian association
VEHICLE_GATE_M=3. # surface anchors can differ across opposite vehicle faces
COMPATIBLE_US=300_000 # three nominal 10Hz frames; fixed across all modes
MIN_WITNESSES=2 # single-source reports remain caution, not confirmation
FREE_FRACTION=.5 # same full-footprint contradiction threshold as Assessor

@dataclass(frozen=True)
class Belief:
    xy:tuple[float,float]
    cls:str
    local:bool
    state:str
    refs:tuple[str,...]


def _belief(message,local,state,refs):
    return Belief(tuple(message.observation.xyz_m[:2]),message.observation.cls.name,
                  local,state,tuple(sorted(set(refs))))


def fuse(mode,*,ego_observations,remote_admitted,store,coverage_map,assessor,now_us):
    if mode not in FUSION_MODES:raise ValueError('unknown fusion mode')
    if type(now_us) is not int or now_us<0:raise ValueError('invalid fusion time')
    if assessor is None:
        from coopx.assess import Assessor
        assessor=Assessor(store,coverage_map)
    elif assessor.store is not store or assessor.coverage_map is not coverage_map:
        raise ValueError('fusion evidence store and assessment map must match')
    def active(message):
        if not isinstance(message,ObservationMessage):return False
        e=message.envelope
        return (e.spatial_frame_id==coverage_map.spatial_frame_id
                and e.capture_time_us<=now_us<=min(e.valid_until_us,e.capture_time_us+coverage_map.ttl_us))
    local=[m for m in ego_observations if active(m)]
    remote=[m for m in remote_admitted if active(m)]
    # Select the latest update per object, not a stream of repeated votes.
    def latest(messages):
        result={}
        for m in messages:
            old=result.get(m.observation.key)
            if old is None or m.envelope.capture_time_us>old.envelope.capture_time_us:result[m.observation.key]=m
        return list(result.values())
    local,remote=latest(local),latest(remote)
    if mode=='ego':return [_belief(m,True,'confirmed',[evidence_ref(m)]) for m in local]
    messages=local+remote
    for message in messages:store.add(message)
    # All admitted observations must exist before assessing the first one, so
    # corroboration cannot depend on input iteration order.
    history=[m for m in store.recent_observations(now_us-coverage_map.ttl_us,now_us) if active(m)]
    result=[]
    for m in messages:
        is_local=m in local;ref=evidence_ref(m)
        if mode=='union':state='confirmed' if is_local else 'unverified';refs=(ref,)
        elif mode=='witness':
            if is_local:state,refs='confirmed',(ref,)
            else:
                assessment=assessor.assess(m,now_us)
                state='confirmed' if assessment.state=='corroborated' else assessment.state
                refs=(ref,*assessment.supporting,*assessment.contradicting)
        else:
            def agrees(c):
                a,b=c.observation,m.observation
                gate=VEHICLE_GATE_M if b.cls==ObjectClass.VEHICLE else PED_GATE_M
                return (a.cls==b.cls and a.anchor==b.anchor and c.label_id==m.label_id
                        and c.envelope.spatial_frame_id==m.envelope.spatial_frame_id
                        and abs(a.captured_sim_us-b.captured_sim_us)<=COMPATIBLE_US
                        and math.dist(a.xyz_m,b.xyz_m)<=gate)
            agreeing=[c for c in history if agrees(c)]
            bodies={a.physical_observer for c in agreeing
                    if (a:=store.anchors.get(c.envelope.source_id)) is not None
                    and a.physical_observer is not None and evidence_eligible(a)}
            refs=tuple(evidence_ref(c) for c in agreeing)
            # Same verified geometry as Witness, but disagreement rejects a
            # report even when geometric consensus supports it (three states).
            from coopx.assess import footprint
            cells,denominator=footprint(m,coverage_map.spec)
            free,free_refs,_=assessor._range_evidence(m,cells,now_us)
            conflict=bool(cells) and len(free)/denominator>=FREE_FRACTION
            state='contradicted' if conflict else 'confirmed' if len(bodies)>=MIN_WITNESSES else 'unverified'
            refs=(*refs,*free_refs)
        result.append(_belief(m,is_local,state,refs))
    return result
