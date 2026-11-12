"""Validate the issued-command / executed-receipt correspondence."""
import math


def inspect(events,episode_id):
    pending={};seen=set();last_end=None;count=0
    for event in events:
        kind=event.get('type');cid=event.get('command_id')
        if kind=='base_command':
            if cid in seen:raise ValueError('duplicate actuator command identity')
            seen.add(cid);pending[cid]={'base':event}
        elif kind=='joint_command':
            if cid not in pending or 'joint' in pending[cid]:raise ValueError('joint command without unique base transaction')
            pending[cid]['joint']=event
        elif kind=='command_without_physics_advance':
            if cid not in pending:raise ValueError('cancelled actuator transaction missing')
            del pending[cid]
        elif kind=='execution_receipt':
            pair=pending.pop(cid,None)
            if pair is None or 'joint' not in pair:raise ValueError('receipt without complete issued command')
            base,joint=pair['base'],pair['joint']
            if event['episode_id']!=episode_id or event['observation_sequence']!=base['sequence']:
                raise ValueError('receipt episode/observation mismatch')
            if event['applied_base']!=base['applied_command'] or event['applied_joint_positions']!=joint['applied_positions'] or event['applied_joint_velocities']!=joint['applied_velocities']:
                raise ValueError('receipt differs from final issued actuator values')
            start,end=event['start_s'],event['end_s']
            if not all(math.isfinite(x) for x in [start,end]) or end<=start or abs(start-base['time_s'])>1e-6:
                raise ValueError('invalid execution interval')
            if last_end is not None and abs(start-last_end)>1e-6:raise ValueError('physical execution gap/overlap')
            last_end=end;count+=1
    if pending:raise ValueError('issued commands lack execution or cancellation evidence')
    if not count:raise ValueError('no executed actuator transactions')
    return {'receipts':count,'last_executed_time_s':last_end,'command_count':len(seen)}
