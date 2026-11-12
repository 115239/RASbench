"""Gate rejection regressions; these are not physical acceptance evidence."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from unittest.mock import patch
from paper200 import release_v2 as gate
from paper200.common import write,sha


class ReleaseGateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.source=self.root/'runtime/runner/test.py';self.source.parent.mkdir(parents=True);self.source.write_text('original')
        self.capture=self.root/'pilot/runtime_snapshot/test.py';self.capture.parent.mkdir(parents=True);self.capture.write_text('original')
        self.raw=self.root/'pilot/trace.jsonl';self.raw.write_text('{"physics_step":1}\n')
        self.review=self.root/'reports/review.json';write(self.review,{'assertions':{'positive':True,'negative':True}})
        self.spec=gate._requirement('physical_control','positive negative','runner/test.py',roles=['physics_trace'])
        self.case={'id':'scoped','scope':'physical_control','report':self.entry(self.review),
            'assertions':{'positive':'/assertions/positive','negative':'/assertions/negative'},
            'sources':{'runner/test.py':{'sha256':sha(self.source),'snapshot_path':str(self.capture)}},
            'evidence':[dict(self.entry(self.raw),role='physics_trace')]}
    def entry(self,path):return {'path':str(path),'sha256':sha(path)}
    def check(self):return gate.check_case(self.root,self.case,self.spec)
    def test_explicit_current_scoped_evidence_validates(self):self.assertEqual([],self.check())
    def test_missing_manifest_rejects_legacy_pass(self):
        self.assertFalse(gate.check_acceptance(self.root,{'passed':True})[0]['passed'])
    def test_passed_report_without_assertions_is_insufficient(self):
        write(self.review,{'passed':True});self.case['report']=self.entry(self.review)
        self.assertTrue(self.check())
    def test_noop_raw_mutation_rejected(self):
        self.raw.write_text('changed');self.assertTrue(any('SHA256' in x for x in self.check()))
    def test_new_runtime_invalidates_prior_physical_proof(self):
        self.source.write_text('new');self.assertTrue(any('candidate source changed' in x for x in self.check()))
    def test_unverified_retrospective_hash_does_not_bind_capture(self):
        del self.case['sources']['runner/test.py']['snapshot_path']
        self.assertTrue(any('producer identity missing' in x for x in self.check()))
    def test_capture_source_mismatch_rejected(self):
        self.capture.write_text('other');self.assertTrue(any('capture source differs' in x for x in self.check()))
    def test_synthetic_cannot_replace_physics(self):
        self.case['synthetic']=True;self.assertTrue(self.check())
    def test_control_cannot_replace_full_runner(self):
        self.spec['scope']='full_runner';self.assertTrue(self.check())
    def test_old_release_evidence_rejected_even_with_valid_hash(self):
        with tempfile.TemporaryDirectory() as external:
            p=Path(external)/'trace.jsonl';p.write_text('old')
            self.case['evidence']=[dict(self.entry(p),role='physics_trace')]
            self.assertTrue(any('this v2 collection' in x for x in self.check()))
    def test_question_system_cross_product_coverage_required(self):
        spec=dict(self.spec,questions=['Q84'],systems=list(gate.SYSTEMS))
        self.case.update(requirement='r',question='Q84',system='groot_agent_off')
        with patch.dict(gate.REQUIREMENTS,{'r':spec},clear=True):
            result=gate.check_acceptance(self.root,{'schema':gate.SCHEMA,'cases':[self.case]})
        self.assertFalse(result[0]['passed']);self.assertEqual(len(result[0]['problems']),2)
    def test_producer_identity_from_hashed_report(self):
        write(self.review,{'assertions':{'positive':True,'negative':True},'source':sha(self.source)})
        self.case['report']=self.entry(self.review)
        self.case['sources']['runner/test.py']={'sha256':sha(self.source),'report_sha256_pointer':'/source'}
        self.assertEqual([],self.check())
    def test_unknown_false_assertion_is_not_pass(self):
        write(self.review,{'assertions':{'positive':True,'negative':None}});self.case['report']=self.entry(self.review)
        self.assertTrue(self.check())

    def expedited_fixture(self):
        rows=[dict(requirement=k,passed=k in gate.EXPEDITED_REQUIRED) for k in gate.REQUIREMENTS]
        review=self.root/'reports/core.json'
        write(review,{'systems':list(gate.SYSTEMS),'current_implementation_inventory_sha256':'candidate',
            'assertions':dict.fromkeys(gate.EXPEDITED_CORE,True),'evidence':[self.entry(self.raw)],
            'statistically_eligible':False})
        policy={'mode':'expedited_collection','user_authorized':True,'user_instruction':'Start collection; defer noncritical checks.',
            'core_review':self.entry(review),'deferred_requirements':{k:'deferred, not passed' for k in gate.REQUIREMENTS if k not in gate.EXPEDITED_REQUIRED}}
        write(self.root/'manifests/expedited_collection_policy.json',policy)
        return rows,review,policy

    def test_expedited_preserves_incomplete_acceptance(self):
        rows,_,_=self.expedited_fixture()
        policy,issues,_=gate.expedited_policy(self.root,rows,'candidate')
        self.assertEqual([],issues);self.assertIsNotNone(policy)
        self.assertFalse(all(x['passed'] for x in rows))

    def test_expedited_does_not_waive_core_or_current_identity(self):
        rows,review,policy=self.expedited_fixture()
        self.assertTrue(gate.expedited_policy(self.root,rows,'changed')[1])
        doc=json.loads(review.read_text());doc['assertions']['clean_no_attack']=False
        write(review,doc);policy['core_review']=self.entry(review)
        write(self.root/'manifests/expedited_collection_policy.json',policy)
        self.assertTrue(any('clean_no_attack' in p for p in gate.expedited_policy(self.root,rows,'candidate')[1]))

    def test_expedited_requires_authorization_and_named_deferrals(self):
        rows,_,policy=self.expedited_fixture();policy['user_authorized']=False
        policy['deferred_requirements'].pop('formal_pairing')
        write(self.root/'manifests/expedited_collection_policy.json',policy)
        errors=gate.expedited_policy(self.root,rows,'candidate')[1]
        self.assertTrue(any('user instruction' in p for p in errors));self.assertTrue(any('formal_pairing' in p for p in errors))

    def test_expedited_rejects_changed_raw_evidence(self):
        rows,_,_=self.expedited_fixture();self.raw.write_text('mutated')
        self.assertTrue(any('SHA256' in p for p in gate.expedited_policy(self.root,rows,'candidate')[1]))

    def drained_source(self):
        import sqlite3
        source=self.root/'source_v1';source.mkdir()
        with sqlite3.connect(str(source/'queue.sqlite3')) as db:
            db.execute('CREATE TABLE runs (status TEXT)')
        write(self.root/'manifests/source_drain.json',{'drained':True,'source':str(source),'parked':946})

    def test_formal_dispatch_and_freeze_never_follow_diagnostic_pass(self):
        import sqlite3
        with sqlite3.connect(str(self.root/'queue.sqlite3')) as db:
            db.execute('CREATE TABLE runs (run_id TEXT, purpose TEXT, status TEXT)')
        self.drained_source()
        (self.root/'STOP').write_text('retain')
        with patch.object(gate,'check_contracts',return_value=([],'contract','implementation')), \
             patch.object(gate,'check_acceptance',return_value=[]):
            result=gate.release_check(self.root)
        self.assertTrue(result['evidence_collection_ready'])
        self.assertFalse(result['statistically_eligible'])
        self.assertFalse(result['dispatch_changed'])
        self.assertTrue((self.root/'STOP').exists())
        self.assertFalse((self.root/'manifests/FROZEN.json').exists())
        self.assertFalse((self.root/'reports/SCORING_READINESS.json').exists())

    def test_freeze_refuses_running_development_jobs(self):
        import sqlite3
        with sqlite3.connect(str(self.root/'queue.sqlite3')) as db:
            db.execute('CREATE TABLE runs (run_id TEXT, purpose TEXT, status TEXT)')
            db.execute("INSERT INTO runs VALUES ('pilot-1','pilot','running')")
        self.drained_source()
        with patch.object(gate,'check_contracts',return_value=([],'contract','implementation')), \
             patch.object(gate,'check_acceptance',return_value=[]):
            with self.assertRaisesRegex(ValueError,'drain development'):
                gate.release_check(self.root,freeze=True)
        self.assertFalse((self.root/'manifests/FROZEN.json').exists())

    def test_formal_running_blocks_diagnostic_release(self):
        import sqlite3
        with sqlite3.connect(str(self.root/'queue.sqlite3')) as db:
            db.execute('CREATE TABLE runs (run_id TEXT, purpose TEXT, status TEXT)')
            db.execute("INSERT INTO runs VALUES ('main-1','main','running')")
        self.drained_source()
        with patch.object(gate,'check_contracts',return_value=([],'contract','implementation')), \
             patch.object(gate,'check_acceptance',return_value=[]):
            result=gate.release_check(self.root)
        self.assertFalse(result['passed'])
        self.assertIn('formal jobs still running',result['problems'])

    def judge_capture(self):
        current=self.root/'runtime/judge/v2/tools/run.py';current.parent.mkdir(parents=True);current.write_text('judge')
        snapshot=self.root/'pilot/judge_runtime_snapshot'
        file=snapshot/'v2/tools/run.py';file.parent.mkdir(parents=True);file.write_text('judge')
        manifest=self.root/'pilot/JUDGE_RUNTIME_SNAPSHOT.json'
        write(manifest,{'snapshot':str(snapshot),'files':{'v2/tools/run.py':sha(file)}})
        invocation=self.root/'pilot/ENTRYPOINT_INVOCATIONS.json'
        write(invocation,{name:{'argv':['python',str(file)]} for name in ('single','manifest','directory','archive')})
        return {'judge_snapshot':self.entry(manifest),'judge_invocations':self.entry(invocation)},current,file

    def test_judge_requires_preexecution_snapshot_not_posthoc_hash_label(self):
        issues=gate.check_judge_capture(self.root,{'sources':{'judge/v2/tools/run.py':{'sha256':'label'}}})
        self.assertTrue(issues)

    def test_judge_complete_snapshot_and_actual_invocations(self):
        case,current,file=self.judge_capture()
        self.assertEqual([],gate.check_judge_capture(self.root,case))

    def test_changed_judge_dependency_invalidates_capture(self):
        case,current,file=self.judge_capture()
        dependency=current.parent/'dependency.py';dependency.write_text('new dependency')
        self.assertIn('Judge complete candidate differs from executed snapshot',gate.check_judge_capture(self.root,case))

    def test_judge_invocation_must_use_frozen_copy(self):
        case,current,file=self.judge_capture()
        path=self.root/'pilot/ENTRYPOINT_INVOCATIONS.json'
        write(path,{name:{'argv':['python',str(current)]} for name in ('single','manifest','directory','archive')})
        case['judge_invocations']=self.entry(path)
        self.assertTrue(any('not bound' in x for x in gate.check_judge_capture(self.root,case)))


if __name__=='__main__':unittest.main()
