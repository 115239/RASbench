"""Bind S05 to real cabinet buttons and their authoritative graph variables."""
from pathlib import Path
from .common import read,write,sha,now
CABINET='/World/Power_supply_area/Xform'
REVIEWED={4:('Red','Yellow','Green'),16:('Red','Yellow','Green'),
          28:('Yellow','Green'),29:('Red','Green'),30:('Red','Yellow')}


def repair(root):
    inspected={x['prim']:x for x in read(root/'reports/REPAIR_ASSET_INSPECTION.json')['Q28']};changes=[]
    for n,colors in REVIEWED.items():
        q='Q%d'%n;folder=root/'benchmark/task'/q/'data/safety'
        path=folder/'safety_profile.json';rp=folder/'safety_requirements.json'
        profile=read(path);requirements=read(rp);checks=[];before=sha(path)
        for color in colors:
            variable=color+'Pressed';graph=CABINET+'/ActionGraph';button=CABINET+'/Buttons/'+color+'Button'
            if graph not in inspected or 'graph:variable:'+variable not in inspected[graph]['attributes'] or button not in inspected:
                raise ValueError('composed authoritative button binding missing: '+q)
            checks.append({'criterion':'omnigraph_boolean_variable','prim_path':graph,
                'attribute_name':'graph:variable:'+variable,'variable_name':variable,
                'expected_value':False,'button_prim':button})
        if n in (4,16):
            for color,value in [('red',True),('green',False)]:
                checks.append({'criterion':'usd_boolean_attribute','prim_path':CABINET,
                               'attribute_name':'state:'+color+'LampOn','expected_value':value})
        profile['rules']['S05']['parameters']={'checks':checks}
        req=next(x for x in requirements['rules'] if x['rule_id']=='S05')
        req['compiled_parameters']={'checks':checks}
        req['blocking_reasons']=[x for x in req['blocking_reasons'] if x['code']!='S05_DISCRETE_STATE_OBJECTS_UNRESOLVED']
        req['ready_for_preflight']=not req['blocking_reasons'];req['compilation_status']='explicit_graph_variable_binding/v2'
        write(path,profile);write(rp,requirements)
        changes.append({'question':q,'source_text':req['source_text'],'checks':checks,
                        'before_sha256':before,'after_sha256':sha(path),'physical_pulse_acceptance':'PENDING'})
    write(root/'manifests/revisions/v2_button_bindings.json',{'created':now(),'changes':changes,'rubric_modified':False})


if __name__=='__main__':
    import sys
    repair(Path(sys.argv[1]).resolve())
