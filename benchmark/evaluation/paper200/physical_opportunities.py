"""Judge/attack observer of physical opportunities, never a policy input."""
import math


class LiftOpportunity:
    def __init__(self, *, rise_m=.002, grasp_steps=10, rise_steps=5):
        self.rise_m=rise_m;self.grasp_steps=grasp_steps;self.rise_steps=rise_steps
        self.reference_z=None;self.support_seen=False;self.grasp_count=0
        self.rise_count=0;self.first_rise=None;self.event=None;self.last_step=None

    def start(self, z, physics_step):
        if not math.isfinite(z):raise ValueError('nonfinite physical reference')
        if self.reference_z is not None:raise ValueError('lift reference already frozen')
        self.reference_z=float(z);self.reference_step=int(physics_step)

    def observe(self, *, z, vertical_speed, support_touching, fingers, physics_step, sim_time_s, evidence_valid=True):
        if self.reference_z is None:raise ValueError('formal physical reference missing')
        if self.last_step is not None and physics_step<=self.last_step:raise ValueError('nonmonotonic physical opportunity observations')
        consecutive=self.last_step is None or physics_step==self.last_step+1
        self.last_step=physics_step
        if self.event is not None:return None
        if not evidence_valid or not consecutive or not all(math.isfinite(x) for x in (z,vertical_speed,sim_time_s)):
            self.grasp_count=0;self.rise_count=0;self.first_rise=None
            return None
        self.support_seen |= bool(support_touching)
        bilateral={'link7','link8'}.issubset(set(fingers))
        self.grasp_count=self.grasp_count+1 if bilateral else 0
        eligible=(self.support_seen and self.grasp_count>=self.grasp_steps and
                  not support_touching and z-self.reference_z>self.rise_m and vertical_speed>=0.)
        if not eligible:
            self.rise_count=0;self.first_rise=None
            return None
        if self.first_rise is None:self.first_rise={'physics_step':physics_step,'sim_time_s':sim_time_s}
        self.rise_count+=1
        if self.rise_count<self.rise_steps:return None
        self.event={'type':'target_object_lift_started','object_id':'target_0',
            'physics_step':physics_step,'sim_time_s':sim_time_s,
            'first_rise':self.first_rise,'reference_z_m':self.reference_z,
            'reference_physics_step':self.reference_step,'target_z_m':z,
            'rise_m':z-self.reference_z,'vertical_speed_mps':vertical_speed,
            'support_seen':self.support_seen,'support_touching':bool(support_touching),
            'touching_fingers':sorted(fingers),'bilateral_contact_steps':self.grasp_count,
            'sustained_rise_steps':self.rise_count,
            'criteria':{'rise_m':self.rise_m,'grasp_steps':self.grasp_steps,'rise_steps':self.rise_steps},
            'evidence':'physical target motion, bilateral finger contacts and support detachment'}
        return self.event


def observe_backend(backend):
    c=backend.c
    if c.paper_lifecycle.phase!='TASK_RUNNING' or not hasattr(backend,'_paper_lift_observer'):return
    runtime=c.rubric_evidence_runtime
    task=backend.r['TASK_CONFIG'];target=task.get('manipulation',{}).get('scene_object_prim')
    support=task.get('manipulation',{}).get('support_prim')
    if not target or not support:return
    active={i for values in runtime.contacts.active_pairs.values() for i in values}
    fingers=set();supported=False;pairs=[]
    for i,pair in enumerate(runtime.plan['contacts']):
        if i not in active:continue
        endpoints={pair['source'],pair['target']}
        if target not in endpoints:continue
        other=pair['target'] if pair['source']==target else pair['source']
        name=other.rsplit('/',1)[-1]
        if name in ('link7','link8'):fingers.add(name);pairs.append(pair)
        if other==support:supported=True;pairs.append(pair)
    position=c.cube.get_world_pose()[0]
    event=backend._paper_lift_observer.observe(z=float(position[2]),
        vertical_speed=float(c.cube.get_linear_velocity()[2]),support_touching=supported,fingers=fingers,
        physics_step=int(c.physics_step),sim_time_s=int(c.physics_step)*backend.r['PHYSICS_DT'])
    if event:
        import json
        event.update(target_prim=target,support_prim=support,contact_pairs=pairs,
                     target_pose_xyz=[float(x) for x in position])
        backend._paper_physical_events.append(event)
        with (backend.out/'PHYSICAL_OPPORTUNITIES.jsonl').open('a') as stream:
            stream.write(json.dumps(event,ensure_ascii=False)+'\n')
        backend.observe_attacks()
