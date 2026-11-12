import concurrent.futures
import json
from pathlib import Path
import tempfile
import unittest
from .common import SYSTEMS, read
from .queue import initialize, claim, finish, connect, report
from .scheduler import admission, select_gpu,profile_key
from .prepare import integrate_agent, primitives, stages


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        questions=[{'question':'Q1','variants':['prompt_01','lidar_01']}]
        from .common import write
        write(self.root/'manifests/questions.json',questions)
        initialize(self.root,questions,'test')
    def tearDown(self):self.tmp.cleanup()
    def test_exact_matrix_no_unrequested_repeats(self):
        with connect(self.root/'queue.sqlite3') as db:
            rows=[dict(r) for r in db.execute('SELECT * FROM runs')]
        self.assertEqual(len(rows),9)
        self.assertEqual({r['seed'] for r in rows},{82026001})
        self.assertEqual({r['system'] for r in rows},set(SYSTEMS))
    def test_concurrent_claims_are_unique(self):
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
            jobs=list(ex.map(lambda i:claim(self.root,str(i)),range(12)))
        self.assertEqual(len({r['run_id'] for r in jobs if r}),9)
        self.assertEqual(sum(r is None for r in jobs),3)
    def test_task_failure_is_not_infrastructure_failure(self):
        job=claim(self.root,'a')
        finish(self.root,job,{'termination':'MODEL_ABSTAIN','task_score':None})
        self.assertEqual(report(self.root)['main_collected'],1)
    def test_attempts_retained_and_retries_bounded(self):
        job=claim(self.root,'a');rid=job['run_id']
        for i in range(3):
            if i:job=claim(self.root,'a',run_id=rid)
            finish(self.root,job,{},error='test infrastructure failure')
        self.assertIsNone(claim(self.root,'a',run_id=rid))
        with connect(self.root/'queue.sqlite3') as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM attempts WHERE run_id=?',(rid,)).fetchone()[0],3)
            self.assertEqual(db.execute('SELECT status FROM runs WHERE run_id=?',(rid,)).fetchone()[0],'infrastructure_failure')
    def test_wrong_owner_cannot_finalize(self):
        job=claim(self.root,'a');job['owner']='b'
        with self.assertRaises(RuntimeError):finish(self.root,job,{})
    def test_final_audit_detects_same_count_wrong_condition_and_auxiliary_omission(self):
        from .common import write
        from .final_audit import matrix_issues
        write(self.root/'manifests/questions.json',[{'question':'Q1','variants':['prompt_01','lidar_01']}])
        write(self.root/'manifests/assistance.json',{'questions':[{'question':'Q1'}]})
        with connect(self.root/'queue.sqlite3') as db:rows=[dict(r) for r in db.execute('SELECT * FROM runs')]
        self.assertEqual(matrix_issues(self.root,rows,'main'),[])
        rows[0]['seed']=72026001
        self.assertTrue(matrix_issues(self.root,rows,'main'))
        self.assertTrue(matrix_issues(self.root,[],'assisted'))


class ResourceTests(unittest.TestCase):
    def test_mature_profile_uses_measured_peak_without_arbitrary_task_floor(self):
        from .scheduler import measured_envelope
        self.assertEqual(measured_envelope(10000,5,'groot_agent_off'),11024)
        self.assertGreaterEqual(measured_envelope(30000,5,'groot_agent_off'),31500)
        self.assertGreater(measured_envelope(10000,1,'groot_agent_off'),11024)
    def test_exited_wrapper_still_cleans_stubborn_child(self):
        import subprocess,sys
        from .launch import stop_group,live_group
        child="import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(60)"
        wrapper="import subprocess,sys; subprocess.Popen([sys.executable,'-c',"+repr(child)+"])"
        p=subprocess.Popen([sys.executable,'-c',wrapper],stdout=subprocess.PIPE,start_new_session=True)
        try:
            self.assertEqual(p.stdout.readline(),b'ready\n');p.wait(timeout=5)
            self.assertTrue(live_group(p.pid))
            stop_group(p,grace_seconds=1)
            self.assertFalse(live_group(p.pid))
        finally:
            stop_group(p,grace_seconds=1);p.stdout.close()
    def test_split_profile_uses_conservative_completed_envelope(self):
        from .scheduler import predicted_peak
        row={'scene':'market','task_type':'navigation','stages':['NAV_OUTBOUND']}
        profiles={'groot_agent_off:market:navigation':{'samples':10,'predicted_peak_mib':19000}}
        self.assertEqual(predicted_peak(profiles,'groot_agent_off',row),19000)
        self.assertEqual(predicted_peak(profiles,'groot_agent_off',row,'assisted'),30000)
    def test_assisted_networks_follow_unchanged_mission_stages(self):
        row={'scene':'market','task_type':'grasp','stages':['PICK']}
        self.assertTrue(profile_key('groot_agent_off',row).endswith(':grasp'))
        self.assertTrue(profile_key('groot_agent_off',row,purpose='assisted').endswith(':grasp'))
        row['stages']=['NAV_OUTBOUND','PICK']
        self.assertTrue(profile_key('groot_agent_off',row,purpose='assisted').endswith(':grasp+unified'))
    def test_reserves_future_peak(self):
        self.assertLess(admission(40000,24000,[{'predicted_peak_mib':30000,'current_mib':10000}]),0)
    def test_more_than_two_tasks_per_gpu_permitted(self):
        active=[{'gpu':3,'current_mib':10000,'predicted_peak_mib':10000} for _ in range(3)]
        self.assertEqual(select_gpu([{'index':3,'total_mib':73415,'used_mib':30000}],active,10000),3)
    def test_busy_gpu_does_not_block_other_cards(self):
        self.assertEqual(select_gpu([{'index':0,'total_mib':73415,'used_mib':69000},
                                    {'index':7,'total_mib':73415,'used_mib':12000}],[],30000),7)


class BindingTests(unittest.TestCase):
    def test_world_offset_rotation_roundtrip(self):
        import numpy as np
        from .attack_runtime import rotation_wxyz
        r=rotation_wxyz([2**-.5,0,0,2**-.5])
        np.testing.assert_allclose(r.T@np.array([.02,0,0]),[0,-.02,0],atol=1e-12)
        np.testing.assert_allclose(r@r.T,np.eye(3),atol=1e-12)
    def test_three_dimensional_gripper_distance_ignores_base_shortcut(self):
        from .attack_runtime import distance
        self.assertEqual(distance({'object_id':'x','subject':'gripper','metric':'prim_origin_3d'},
            {'grasp':{'tcp_xyz':[0,0,1]},'objects':{'x':{'position_xyz':[0,0,3]}},'distances':{'x':0}}),2.)
    def test_surface_distance_uses_both_bounds(self):
        from .attack_runtime import distance
        s={'subjects':{'robot_base':{'world_aabb':[[0,0,0],[2,2,2]]}},
           'objects':{'x':{'world_aabb':[[3,0,0],[4,1,1]]}}}
        self.assertEqual(distance({'object_id':'x','metric':'world_aabb_surface_xy'},s),1.)
    def test_navigation_return_is_not_silently_dropped(self):
        self.assertEqual(stages({'instruction':'导航至目标并返回出生点','execution_mode':'navigation_only'},
                                {'task_type':'navigation'}),['NAV_OUTBOUND','NAV_RETURN'])
    def test_mixed_manipulation_stages(self):
        self.assertEqual(stages({'instruction':'抓取物体返回出生点并放下','execution_mode':'mobile_manipulation'},
                                {'task_type':'mobile_manipulation'}),['NAV_OUTBOUND','PICK','HANDOFF','NAV_RETURN','PLACE'])
    def test_rubric_references(self):
        self.assertEqual(primitives({'text':'目标（/World_01/table/body）及 /World/B/cube。'}),
                         ['/World/B/cube','/World_01/table/body'])


class AttackAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.episode=Path(self.tmp.name)
        (self.episode/'resolved_attack_profile').mkdir();(self.episode/'attack_evidence').mkdir()
        (self.episode/'resolved_attack_profile/all_attack_definitions.json').write_text(json.dumps({'attack':{'attacks':[
            {'id':'q1_prompt_01','enabled':True,'effect_enabled':True},
            {'id':'q1_imu_01','enabled':True,'effect_enabled':False}]}}))
    def tearDown(self):self.tmp.cleanup()
    def audit(self,events,variant):
        from .chain_evidence import inspect
        (self.episode/'attack_evidence/unified_attack_audit.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        return inspect(self.episode,{'question':'Q1','variant':variant})
    def test_clean_condition_marker_is_not_effect(self):
        v=self.audit([{'event':'attack_runtime_initialized','attack_enabled':False},
            {'event':'condition_triggered','attack_id':'q1_prompt_01'}],'clean')
        self.assertFalse(v['configured']);self.assertFalse(v['effective']);self.assertFalse(v['triggered'])
    def test_untriggered_attack_is_preserved(self):
        v=self.audit([{'event':'attack_runtime_initialized','attack_enabled':True}],'prompt_01')
        self.assertTrue(v['configured']);self.assertFalse(v['triggered']);self.assertFalse(v['effective'])
    def test_wrong_attack_is_rejected(self):
        with self.assertRaises(ValueError):
            self.audit([{'event':'attack_runtime_initialized','attack_enabled':True},
                {'event':'effect_first_applied','effect_applied':True,'attack_id':'q1_imu_01',
                 'time_s':1.,'evidence':{'target_before_sha256':'a','target_after_sha256':'b','executor':'imu_sample_transform'}}],'prompt_01')


class GeneratedCacheTests(unittest.TestCase):
    def test_cleanup_preserves_raw_evidence(self):
        from .common import write
        from .runtime_cache import finish as dispose
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'attempt';cache=out/'private_runtime/cache/ov/texturecache'
            cache.mkdir(parents=True);(cache/'generated.bin').write_bytes(b'generated')
            write(out/'PROCESS_CLEANUP.json',{'completed':True});write(out/'COLLECTION_VALIDATION.json',{})
            write(out/'processes.json',{});raw=out/'submission.hdf5';raw.write_bytes(b'raw physical evidence')
            result=dispose(root,out,'factory')
            self.assertEqual(result['logical_bytes'],9)
            self.assertFalse(cache.exists());self.assertEqual(raw.read_bytes(),b'raw physical evidence')
            self.assertTrue((out/'XDG_CACHE_DISPOSAL.json').exists())
    def test_unknown_cache_content_is_preserved(self):
        from .common import write
        from .runtime_cache import finish as dispose
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);out=root/'attempt';cache=out/'private_runtime/cache'
            cache.mkdir(parents=True);unknown=cache/'unknown_evidence';unknown.write_bytes(b'keep')
            write(out/'PROCESS_CLEANUP.json',{'completed':True});write(out/'COLLECTION_VALIDATION.json',{})
            write(out/'processes.json',{})
            with self.assertRaises(ValueError):dispose(root,out,'factory')
            self.assertEqual(unknown.read_bytes(),b'keep')


class AuxiliaryApproachTests(unittest.TestCase):
    def test_far_target_uses_bounded_lidar_command_and_respects_guard_stop(self):
        import numpy as np
        from types import SimpleNamespace
        from .assistance import ManipulationProposal
        helper=ManipulationProposal.__new__(ManipulationProposal)
        calls=[];command=np.array([.4,-.3,.8])
        def guard(goal):calls.append(goal.copy());return command.copy(),{'state':'track'}
        helper.c=SimpleNamespace(world=SimpleNamespace(current_time=1.),
            cube=SimpleNamespace(get_world_pose=lambda:(np.array([1.,0.,.8]),np.array([1.,0.,0.,0.]))),
            link_transform=lambda body:(np.zeros(3),np.array([1.,0.,0.,0.])),
            gripper_body=1,arm_body=0,lidar_navigation_command=guard)
        helper.r={'quaternion_to_rotation_matrix':lambda q:np.eye(3),'TCP_OFFSET':np.zeros(3)}
        helper.cfg={};helper.phase=None;helper.center_local=np.zeros(3);helper.calls=0
        action,detail=helper.action('PICK')
        np.testing.assert_allclose(action[:3],[.08,-.08,.2]);np.testing.assert_array_equal(action[3:],0)
        np.testing.assert_array_equal(calls[0],[1.,0.]);self.assertEqual(detail['state'],'position_base')
        command[:]=0
        action,_=helper.action('PICK');np.testing.assert_array_equal(action,0)
        helper.c.world.current_time=7.
        action,detail=helper.action('PICK')
        np.testing.assert_array_equal(action[:3],0);self.assertGreater(np.linalg.norm(action[3:6]),0)
        self.assertEqual(detail['state'],'approach')


class FrozenBindingTests(unittest.TestCase):
    def test_symbolic_model_binding_cannot_switch_to_another_same_content_target(self):
        from .common import write,sha
        from .provenance import file_stat,verify_frozen
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);a=root/'a';b=root/'b';a.write_text('weights');b.write_text('weights')
            link=root/'binding';link.symlink_to('a')
            write(root/'manifests/files_final.json',{'a':sha(a)})
            write(root/'manifests/file_stats_final.json',{'a':file_stat(a)})
            write(root/'manifests/model_files_verified.json',{'files':{}})
            write(root/'manifests/symlinks_final.json',{'binding':{'target':'a','resolved':str(a)}})
            write(root/'manifests/FROZEN.json',{'files_manifest_sha256':sha(root/'manifests/files_final.json'),
                'stats_manifest_sha256':sha(root/'manifests/file_stats_final.json')})
            verify_frozen(root)
            link.unlink();link.symlink_to('b')
            with self.assertRaisesRegex(ValueError,'symbolic binding'):verify_frozen(root)


if __name__=='__main__':unittest.main()
