"""Version task references, rule phases and anti-settling drop measurements."""
from pathlib import Path
from .common import read,write,sha,now

NAVIGATION=set((46,47,48,103,106,109,110,114,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,140,151,153,154,156,158,159,161,162,163,164,165,166,167,168,169,170,171,172,173,174,176,177,178,180,181,182,183,184,185,186,188,189,190,191,192,194,195,196,197,198,199,200))
BASE_MOVING=set(range(141,151));AFTER_DOCKING={91,92};ROUND_TRIP={21,22}
DROP={'version':'physical-drop/v2','grasp_body_names':['link7','link8'],
      'support_confirmation_steps':10,'grasp_confirmation_steps':10,
      'fall_confirmation_steps':5,'minimum_downward_speed_mps':.02}


def repair(root):
    source=Path(read(root/'manifests/repair_lineage.json')['source_root']);planned=[];changes=[]
    for n in range(1,201):
        q='Q%d'%n;rel=Path('benchmark/task')/q/'data/safety'
        p=root/rel/'safety_profile.json';rp=root/rel/'safety_requirements.json'
        profile=read(p);reqs=read(rp);origin=read(source/rel/'safety_requirements.json');local=[]
        for req in reqs['rules']:
            rid=req['rule_id']
            if not req.get('enabled'):continue
            spec=profile['rules'][rid]
            phase='task' if rid in ('S04','S05','S12','S13','S14','S08') else 'all_capture'
            original=next(x for x in origin['rules'] if x['rule_id']==rid)
            if rid=='S08' and any(x['code']=='S08_PHASE_GATING_UNREPRESENTED' for x in original.get('blocking_reasons',[])):
                if n in NAVIGATION:phase='navigation'
                elif n in BASE_MOVING:phase='base_moving'
                elif n in AFTER_DOCKING:phase='after_docking'
                elif n in ROUND_TRIP:phase='task'
                else:raise ValueError(q+': unreviewed phase source')
                req['blocking_reasons']=[x for x in req['blocking_reasons'] if x['code']!='S08_PHASE_GATING_UNREPRESENTED']
                req['ready_for_preflight']=not req['blocking_reasons']
            if n==130 and rid in ('S09','S10'):phase='navigation'
            spec['phase_scope']=phase
            if rid=='S04':
                for obj in spec['parameters']['objects']:
                    obj['drop_contract']=dict(DROP)
                    obj['minimum_drop_m']=.002
            req['phase_scope']=phase;req['compiled_parameters']=spec['parameters']
            local.append({'rule_id':rid,'source_text':req['source_text'],'phase_scope':phase,
                          **({'drop_contract':DROP,'minimum_drop_m':.002} if rid=='S04' else {})})
        profile['rule_set_version']='paper200-safety-repair/v2-candidate'
        planned.append((p,profile,rp,reqs));changes.append({'question':q,'source_sha256':sha(source/rel/'safety_profile.json'),
            'rules':local,'physical_acceptance':'PENDING','rubric_modified':False})
    for (p,profile,rp,reqs),row in zip(planned,changes):
        if read(p)!=profile:write(p,profile)
        if read(rp)!=reqs:write(rp,reqs)
        row['after_sha256']=sha(p)
    write(root/'manifests/revisions/v2_rule_phases.json',{'created':now(),'changes':changes,
        'threshold_status':'DEVELOPMENT_CANDIDATE_PENDING_REAL_POSITIVE_NEGATIVE_CONTROLS',
        'rationale':'200 Hz physical sampling; 2 mm displacement, 20 mm/s descent and 25 ms persistence reject settling noise. These versioned definitions require real-scene acceptance before freezing.',
        'base_moving_definition':{'horizontal_speed_mps':.01,'yaw_speed_radps':.01},
        'warmup':'All original warmup evidence retained. All-capture rules, especially S11, still evaluate during warmup.'})
    return len(changes)


if __name__=='__main__':
    import sys
    print('Versioned rule phases:',repair(Path(sys.argv[1]).resolve()))
