"""Task scoring must not count warmup success or parking-tail failure."""
import json,tempfile
from pathlib import Path
import h5py,numpy as np,pytest
from v2.common.task_trajectory import TaskTrajectoryView
from robolab_action_judge.errors import ActionJudgeError


@pytest.fixture(autouse=True)
def isolate_projection_from_unrelated_robot_schema(monkeypatch):
    # These cases exercise time/reference selection; full robot schema is
    # independently enforced by the real collection protocol validator.
    import robolab_action_judge.trajectory_evidence as evidence
    monkeypatch.setattr(evidence,'validate_submission_structure',lambda file: None)


def fixture(path):
    f=h5py.File(path,'w');root=f.create_group('data/demo_0')
    root.create_dataset('actions',data=np.zeros((5,10),np.float32))
    t=root.create_group('time')
    for key in ['action_time_s','state_time_s']:t.create_dataset(key,data=np.arange(5,dtype=np.float64))
    t.create_dataset('step_index',data=np.arange(5,dtype=np.uint64))
    f.create_group('metadata').attrs.update(schema_name='robolab-action-hdf5',schema_version=1)
    g=f.create_group('extensions/paper200_collection');g.attrs['schema']='paper200-collection/v2'
    result={'task_started':True,'task_start_sim_time_s':1.,'task_start_physics_step':200,'termination_time_s':3.}
    g.create_dataset('result_json',data=np.frombuffer(json.dumps(result).encode(),np.uint8))
    r=f.create_group('extensions/bboard_rubric_v04');r.attrs.update(formal_reference_phase='TASK_RUNNING',formal_reference_sim_time_s=1.,formal_reference_physics_step=200)
    root.create_dataset('states/rigid_object/%2FTarget/root_pose',data=np.array([[99,0,0,1,0,0,0],[1,0,0,1,0,0,0],[2,0,0,1,0,0,0],[3,0,0,1,0,0,0],[-99,0,0,1,0,0,0]],np.float32))
    root.create_dataset('initial_state/rigid_object/%2FTarget/root_pose',data=[99,0,0,1,0,0,0])
    r.create_dataset('objects/%2FTarget/formal_initial/root_pose',data=np.array([1,0,0,1,0,0,0],np.float32))
    return f


def test_task_interval_and_reference(tmp_path):
    with fixture(tmp_path/'e.h5') as f:
        view=TaskTrajectoryView(f,None)
        path='/data/demo_0/states/rigid_object/%2FTarget/root_pose'
        series=view.load({'path':path,'slice':[0,3]})
        aligned=view.align({'position':series})
        assert aligned.time_s.tolist()==[1.,2.,3.]
        assert aligned.control_steps.tolist()==[0,1,2]
        assert aligned.values['position'][:,0].tolist()==[1.,2.,3.]
        assert view.load({'path':path,'sample':'first','slice':[0,3]}).values.tolist()==[1.,0.,0.]
        initial=view.load({'path':path.replace('/states/','/initial_state/'),'sample':'first','slice':[0,3]})
        assert initial.values.tolist()==[1.,0.,0.]
        assert f[path][0,0]==99 and f[path][-1,0]==-99
        assert view.align({'position':series},after_step=0).time_s.tolist()==[2.,3.]


def test_missing_reference_not_warmup_fallback(tmp_path):
    with fixture(tmp_path/'e.h5') as f:
        del f['extensions/bboard_rubric_v04'].attrs['formal_reference_sim_time_s']
        with pytest.raises(ActionJudgeError,match='TASK_PHASE_REFERENCE_UNAVAILABLE'):TaskTrajectoryView(f,None)
