"""Cooperative claim-wide wall budget and low-overhead monotonic timings.

The launcher owns the absolute hard deadline. This observer never interrupts
native code, an incomplete physics sample, or the stop/export path.
"""
import math
import time
from functools import wraps
from .common import write
from .wallclock_budget import EXPORT_RESERVE_S


class WallClock:
    def __init__(self, cfg, *, unix_clock=time.time, monotonic_clock=time.monotonic):
        self.unix_clock=unix_clock;self.monotonic_clock=monotonic_clock
        self.attached_unix=unix_clock();self.attached_monotonic=monotonic_clock()
        self.started=cfg.get('wall_clock_started_unix_s')
        self.deadline=cfg.get('wall_clock_deadline_unix_s')
        self.reserve=float(cfg.get('wall_clock_export_reserve_s',EXPORT_RESERVE_S))
        self.soft_monotonic=None
        if self.deadline is not None:
            if self.started is None or not all(math.isfinite(float(x)) for x in (self.started,self.deadline,self.reserve)):
                raise ValueError('wall-clock budget requires finite launcher claim start/deadline/reserve')
            if self.deadline<=self.started or self.reserve<0 or self.reserve>=self.deadline-self.started:
                raise ValueError('invalid claim-wide wall-clock budget')
            self.soft_monotonic=self.attached_monotonic+(self.deadline-self.reserve-self.attached_unix)
        self.expired_at=None;self.components={};self.milestones={};self.phase_times={}
        self.phase=None;self.phase_started=self.attached_monotonic

    def stamp(self):
        now=self.monotonic_clock()
        return {'unix_s':self.attached_unix+now-self.attached_monotonic,
                'monotonic_s':now,'claim_elapsed_s':None if self.started is None else self.attached_unix+now-self.attached_monotonic-self.started}

    def mark(self,name,**details):
        self.milestones.setdefault(name,{**self.stamp(),**details})

    def phase_change(self,phase):
        if phase==self.phase:return
        now=self.monotonic_clock()
        if self.phase is not None:self.phase_times[self.phase]=self.phase_times.get(self.phase,0.)+now-self.phase_started
        self.phase=str(phase);self.phase_started=now

    def poll(self,phase,physics_step):
        if phase in ('STOPPING','FINALIZED'):return False
        if self.expired_at is None and self.soft_monotonic is not None and self.monotonic_clock()>=self.soft_monotonic:
            self.expired_at={**self.stamp(),'lifecycle_phase':phase,'physics_step':int(physics_step)}
        return self.expired_at is not None

    def wrap(self,obj,name,category):
        original=getattr(obj,name,None)
        if original is None:return
        @wraps(original)
        def measured(*args,**kwargs):
            started=self.monotonic_clock()
            try:return original(*args,**kwargs)
            finally:
                row=self.components.setdefault(category,{'calls':0,'wall_s':0.})
                row['calls']+=1;row['wall_s']+=self.monotonic_clock()-started
        setattr(obj,name,measured)

    def snapshot(self):
        phases=dict(self.phase_times)
        if self.phase is not None:phases[self.phase]=phases.get(self.phase,0.)+self.monotonic_clock()-self.phase_started
        return {'schema':'paper200-wall-clock-timing/v1','claim_started_unix_s':self.started,
                'hard_deadline_unix_s':self.deadline,'soft_deadline_unix_s':None if self.deadline is None else self.deadline-self.reserve,
                'export_reserve_s':self.reserve,'clock_semantics':'launcher claim deadline, mapped once to local monotonic; no signal timers',
                'setup_elapsed_at_attach_s':None if self.started is None else self.attached_unix-self.started,
                'milestones':self.milestones,'soft_limit_observed':self.expired_at,'snapshot':self.stamp(),
                'phase_wall_s':phases,'components':self.components,
                'component_semantics':'inclusive nested wall durations, not additive; no per-step file writes',
                'final_export_and_validation_timing':'launcher ledger; occurs after rollout recording closes'}


def install(backend):
    c=backend.c;clock=WallClock(backend.cfg);c.paper_wall_clock=clock
    clock.mark('rollout_attached',physics_step=int(c.physics_step))
    for obj,name,category in [(c.world,'step','simulation'),(c.world,'render','renderer'),
            (c,'capture','capture_and_video_frames'),(c,'_observe_safety','safety_observation'),
            (c.safety_geometry_tracker,'sample','geometry'),
            (c.paper_inventory_writer,'step','physical_state_evidence'),
            (c.paper_geometry_writer,'observe','geometry_evidence'),
            (c.rubric_evidence_writer,'step','rubric_evidence'),
            (backend,'infer','model_roundtrip'),(backend,'plan','planning')]:
        if obj is not None:clock.wrap(obj,name,category)
    original=c.paper_physical_observer
    def physics_boundary():
        original()
        clock.phase_change(c.paper_lifecycle.phase+':'+backend.phase)
        # Observe every completed physics sample; raising waits until the
        # backend has appended the complete control transaction and receipts.
        clock.poll(c.paper_lifecycle.phase,c.physics_step)
    c.paper_physical_observer=physics_boundary
    return clock


def check(backend):
    clock=getattr(backend.c,'paper_wall_clock',None)
    if clock is not None and clock.poll(backend.c.paper_lifecycle.phase,backend.c.physics_step):
        persist(backend,rollout_recording_closed=False)
        from mission import Stop
        raise Stop('WALL_CLOCK_LIMIT')


def persist(backend,**extra):
    clock=getattr(backend.c,'paper_wall_clock',None)
    if clock is not None:
        write(backend.out/'WALL_CLOCK_TIMING.json',{**clock.snapshot(),
            'physics_steps':int(backend.c.physics_step),'simulation_time_s':int(backend.c.physics_step)*float(backend.r['PHYSICS_DT']),
            'model_calls':int(backend.request_count),**extra})
