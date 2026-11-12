"""Runner-only observers of actual OmniGraph writes; no authored state substitutes."""
import json
from .common import write
CURRENT=None


def graph_receipt(key,value):
    if CURRENT is None:raise RuntimeError('discrete observer missing')
    CURRENT.record(key,bool(value),'authoritative_graph_write_output')


class DiscreteObserver:
    def __init__(self,c,out,dt):
        import omni.graph.core as og
        import omni.kit.app
        from pxr import Tf,Usd
        global CURRENT
        if CURRENT is not None:raise RuntimeError('discrete observer must be process isolated')
        self.c=c;self.dt=dt;self.out=out;self.last={};self.pending=[];self.specs={};self.error=None
        profile=c.safety_profile or {};checks=profile.get('rules',{}).get('S05',{}).get('parameters',{}).get('checks',[])
        self.checks=[x for x in checks if x['criterion'] in ('omnigraph_boolean_variable','usd_boolean_attribute')]
        self.physical_checks=[x for x in checks if x['criterion']=='relative_pose_closed']
        self.closure_references={}
        for spec in self.physical_checks:
            if spec.get('reference')=='authored_closed':
                value=c.final_stage.GetPrimAtPath(spec['prim_path']).GetCustomDataByKey('paper200:authoredClosureReference')
                if value is None:raise ValueError('authored closure geometry reference missing')
                self.closure_references[spec['prim_path']]=json.loads(value)
        self.log=(out/'DISCRETE_STATE_EVENTS.jsonl').open('x',encoding='utf-8',buffering=1)
        CURRENT=self
        graph_checks=[x for x in self.checks if x['criterion']=='omnigraph_boolean_variable']
        if graph_checks:
            manager=omni.kit.app.get_app().get_extension_manager()
            for extension in ('omni.physx.graph','isaacsim.core.nodes','omni.graph.action','omni.graph.scriptnode'):
                manager.set_extension_enabled_immediate(extension,True)
        bindings=[{'criterion':x['criterion'],'cap_prim':x['prim_path'],'body_prim':x['relative_body_prim'],
                   'reference':x['reference'],'authored_reference':self.closure_references.get(x['prim_path']),
                   'source':'independent actual PhysX inventory poses each physics step',
                   'pose_order':'xyz_wxyz','world_frame':'stage_world','sampling_dt_s':dt}
                  for x in self.physical_checks]

        for spec in self.checks:
            key=spec['prim_path']+'.'+spec['attribute_name'];self.specs[key]=spec
            if spec['criterion']=='omnigraph_boolean_variable':
                graph=spec['prim_path'];color=spec['variable_name'].removesuffix('Pressed')
                writer=graph+'/'+color+'WritePressed';name='Paper200Observe'+color
                if not c.final_stage.GetPrimAtPath(writer).IsValid():raise ValueError('missing authoritative button writer: '+writer)
                _,nodes,_,_=og.Controller.edit(graph,{og.Controller.Keys.CREATE_NODES:[(name,'omni.graph.scriptnode.ScriptNode')]})
                node=nodes[0]
                og.Controller.create_attribute(node,'inputs:observed',og.Type(og.BaseDataType.BOOL),og.AttributePortType.ATTRIBUTE_PORT_TYPE_INPUT)
                script="def compute(db):\n    from paper200.discrete_state import graph_receipt\n    graph_receipt(%r, db.inputs.observed)\n    return True\n"%key
                og.Controller.edit(graph,{og.Controller.Keys.SET_VALUES:[(graph+'/'+name+'.inputs:script',script)],
                    og.Controller.Keys.CONNECT:[(writer+'.outputs:value',graph+'/'+name+'.inputs:observed'),
                                               (writer+'.outputs:execOut',graph+'/'+name+'.inputs:execIn')]})
                bindings.append({'key':key,'writer':writer,'button_prim':spec['button_prim'],
                                 'observer':graph+'/'+name,'source':'actual graph write output; every execution'})
            else:
                attr=c.final_stage.GetPrimAtPath(spec['prim_path']).GetAttribute(spec['attribute_name'])
                value=attr.Get() if attr else None
                if isinstance(value,bool):self.record(key,value,'initial_usd_value')
                bindings.append({'key':key,'source':'USD value and synchronous change notices'})
        def changed(notice,sender):
            try:
                for path in notice.GetChangedInfoOnlyPaths():
                    key=str(path);spec=self.specs.get(key)
                    if spec is None or spec['criterion']!='usd_boolean_attribute':continue
                    attr=sender.GetAttributeAtPath(path);value=attr.Get() if attr else None
                    if not isinstance(value,bool):raise ValueError('discrete state became unavailable: '+key)
                    self.record(key,value,'usd_change_notice')
            except Exception as error:self.error=repr(error)
        self._notice_callback=changed  # Tf.Notice retains only a weak Python callback reference.
        self.notice=Tf.Notice.Register(Usd.Notice.ObjectsChanged,self._notice_callback,c.final_stage)
        write(out/'DISCRETE_STATE_BINDINGS.json',{'schema':'paper200-discrete-state/v2','bindings':bindings,
            'coverage':'graph execution receipt per physics step; USD transitions retained even within one step',
            'physical_positive_negative_acceptance':'PENDING'})

    def record(self,key,value,source):
        step=int(self.c.physics_step)+1
        record={'key':key,'value':bool(value),'physics_step':step,'sim_time_s':step*self.dt,
                'source':source,'lifecycle_phase':self.c.paper_lifecycle.phase}
        self.last[key]=record;self.pending.append(record)
        self.log.write(json.dumps(record,ensure_ascii=False)+'\n')

    def sample(self,step):
        states={};coverage={};transitions=list(self.pending);self.pending.clear()
        for key,spec in self.specs.items():
            row=self.last.get(key);valid=row is not None
            if spec['criterion']=='omnigraph_boolean_variable':valid &= row is not None and row['physics_step']==step
            else:
                attr=self.c.final_stage.GetPrimAtPath(spec['prim_path']).GetAttribute(spec['attribute_name'])
                value=attr.Get() if attr else None;valid=isinstance(value,bool)
                if valid:row={'value':value}
            valid &= self.error is None
            coverage[key]={'complete':bool(valid),'reason':None if valid else self.error or 'authoritative_state_receipt_missing'}
            if valid:states.setdefault(spec['prim_path'],{})[spec['attribute_name']]=row['value']
        physical={}
        inventory=getattr(self.c,'paper_inventory_writer',None)
        if self.physical_checks and inventory is not None:
            poses=inventory.view.get_transforms()
            wanted={x[k] for x in self.physical_checks for k in ('prim_path','relative_body_prim')}
            for path,pose in zip(inventory.paths,poses):
                if path in wanted:
                    physical[path]={'pose':[float(x) for x in pose[:3]]+[float(pose[6])]+[float(x) for x in pose[3:6]],
                                    'source':'actual_physx_independent_rigid_body','rigid_owner':path}
        return {'state_attributes':states,'state_transitions':transitions,'state_coverage':coverage,
                'discrete_object_states':physical,'closure_references':self.closure_references}


    def close(self):
        global CURRENT
        self.notice.Revoke();self.log.close();CURRENT=None
