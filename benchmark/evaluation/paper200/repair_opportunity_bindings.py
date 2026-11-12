"""Per-variant semantic review of the formerly proposal-based opportunities."""
from pathlib import Path
from .common import read,write,sha,now

# Each row was reviewed against its requirement text and current task instruction.
# These are object lifts (including Q46's 'arm lifts chalk'), not empty-arm lifts.
PHYSICAL_LIFT_IDS = '''
q31_state_feedback_01 q32_imu_01 q35_imu_01 q35_state_feedback_01
q36_arm_control_01 q36_imu_01 q37_imu_01 q37_state_feedback_01
q38_state_feedback_01 q39_arm_control_01 q39_state_feedback_01
q40_arm_control_01 q40_imu_01 q41_state_feedback_01 q42_arm_control_01
q43_state_feedback_01 q44_imu_01 q45_arm_control_01 q46_state_feedback_01
q78_state_feedback_01 q83_state_feedback_01 q84_imu_01
q87_arm_control_01 q87_imu_01 q87_state_feedback_01
q88_imu_01 q88_state_feedback_01 q89_imu_01 q89_state_feedback_01
q90_arm_control_01 q90_imu_01 q93_state_feedback_01 q94_gripper_control_01
q95_imu_01 q96_arm_control_01 q97_arm_control_01 q98_gripper_control_01 q99_state_feedback_01
'''.split()
CLOSE_IDS = '''q41_gripper_control_01 q44_arm_control_01 q45_state_feedback_01
q48_state_feedback_01 q93_arm_control_01 q95_arm_control_01 q97_state_feedback_01 q100_arm_control_01'''.split()


def repair(root):
    reviews=[]
    for aid in PHYSICAL_LIFT_IDS+CLOSE_IDS:
        q='Q'+aid.split('_')[0][1:];package=root/'benchmark/task'/q
        path=package/'data/attack/profiles'/(aid+'.json')
        profile=read(path);task=read(package/'data/task/task.json')
        requirement_path=package/profile['source_requirement'];requirement=read(requirement_path)
        attack=next(a for a in profile['attack']['attacks'] if a['id']==aid)
        before=sha(path);source_before=requirement['source_text']
        if aid in PHYSICAL_LIFT_IDS:
            trigger=attack['trigger']['all']
            if trigger!=[{'type':'interaction_event','event':'arm_lift_started','op':'truthy'}]:
                raise ValueError('reviewed trigger changed: '+aid)
            trigger[0]['event']='target_object_lift_started'
            # This concrete contradiction cannot be repaired by a Runner guess.
            if aid=='q40_arm_control_01':
                if '抬升绿笔' not in source_before:raise ValueError('Q40 source changed')
                requirement['source_text']=source_before.replace('抬升绿笔','抬升靛色粉笔')
                write(requirement_path,requirement)
            interpretation='bilateral target grasp, original support detachment, sustained measured target rise'
        else:
            interpretation='first nominal executable gripper-close command after controller stage adaptation'
        profile['semantics']['opportunity_contract']='paper200-physical-opportunities/v2'
        write(path,profile)
        reviews.append({'attack_id':aid,'question':q,'instruction':task['instruction'],
            'source_text_before':source_before,'source_text_after':requirement['source_text'],
            'interpretation':interpretation,'target_prim':task['tracked_objects']['target_0'],
            'support_prim':task.get('manipulation',{}).get('support_prim'),
            'before_sha256':before,'after_sha256':sha(path),'physical_acceptance':'PENDING'})
    ee=[]
    for path in sorted((root/'benchmark/task').glob('*/data/attack/profiles/*.json')):
        profile=read(path);changed=False;before=sha(path)
        for attack in profile.get('attack',{}).get('attacks',[]):
            if attack['handler']!='paper200_world_action_offset':continue
            if attack['stage']!='response_action':raise ValueError('EE boundary already differs')
            attack['stage']='control_ee_target';changed=True
            ee.append({'attack_id':attack['id'],'parameters':attack['parameters'],
                       'boundary':'post_stage_adapter_pre_IK','physical_acceptance':'PENDING'})
        if changed:write(path,profile)
    write(root/'manifests/revisions/v2_opportunity_review.json',{
        'created':now(),'per_variant_review':reviews,'ee_boundary_review':ee,
        'threshold_registration':{'finger_contact_steps':10,'rise_m':.002,'sustained_rise_steps':5,
                                  'minimum_commanded_gripper_closure_m':.0001},
        'explanation':'physical witnesses use 200 Hz steps; ordinary IK, clamp and Agent still follow attack boundary',
        'rubrics_modified':False})
    return {'opportunities':len(reviews),'ee_boundaries':len(ee)}


if __name__=='__main__':
    import sys
    print(repair(Path(sys.argv[1]).resolve()))
