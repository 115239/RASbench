"""Resumable formal batch execution, audits and deliverables after release."""
import fcntl,os,traceback
from .common import read,write,now
from .provenance import verify_frozen
from .scheduler import run as schedule
from .final_audit import audit
from .collection_report import build


def run(root):
    with (root/'formal_pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not read(root/'reports/MAIN_RELEASE_GATE.json').get('passed'):
            raise ValueError('formal batch requires the completed frozen release gate')
        verify_frozen(root)
        if (root/'manifests/source_drain.json').exists():
            from .resume_repaired import verify_prepared
            verify_prepared(root)
        if (root/'STOP').exists():
            raise ValueError('dispatch remains stopped')
        state={'pid':os.getpid(),'started':now(),'status':'RUNNING','scoring':'NOT_SCORED'}
        def update(**values):
            state.update(values,updated=now());write(root/'reports/FORMAL_PIPELINE.json',state)
        try:
            for purpose in ['main','assisted']:
                update(stage='collect_'+purpose)
                schedule(root,purpose)
                update(stage='audit_'+purpose)
                result=audit(root,purpose)
                build(root)
                if not result['passed']:
                    update(status='INCOMPLETE',issues=result['issues'],required_action='inspect retained failures; resume only after repair within frozen-version rules')
                    return state
            result=build(root)
            if result['legacy_audit_required']:
                if not result['remaining_all_collection_complete']:
                    raise RuntimeError('remaining matrix collection is incomplete')
                from .resume_repaired import request_legacy_audit
                legacy=request_legacy_audit(root)
                update(status='LEGACY_AUDIT_REQUIRED',stage='audit_parked_source',
                       remaining_collection_completed=now(),parked=len(legacy['records']),
                       whole_paper_complete=False,required_action=legacy['required_action'])
                return state
            if not result['all_deliverables_complete']:raise RuntimeError('final collection report did not certify all deliverables')
            update(status='COMPLETE_PENDING_SCORING',stage='done',completed=now())
        except Exception:
            update(status='ERROR',error=traceback.format_exc())
            raise
        return state
