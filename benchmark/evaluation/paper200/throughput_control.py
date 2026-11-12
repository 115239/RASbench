"""Bounded concurrency trials driven by valid output rate and service contention.

VRAM and disk admission remain independent hard constraints. Lower targets drain
naturally; this controller never interrupts active work or caps tasks per GPU.
"""
import statistics,time
from pathlib import Path
from .common import read,write


def decide(state,metrics,minimum=1,maximum=64):
    target=state.get('target',8);previous=state.get('baseline')
    count=metrics.get('completed_valid',0);bad=metrics.get('wallclock_timeouts',0)
    io=metrics.get('io_full_avg10',0);latency=metrics.get('model_p95_latency_s');speed=metrics.get('median_sim_seconds_per_wall_second')
    degraded=bad>0 or io>=10
    if previous:
        prior_latency=previous.get('model_p95_latency_s');prior_speed=previous.get('median_sim_seconds_per_wall_second')
        if latency is not None and prior_latency and latency>prior_latency*1.5:degraded=True
        if speed is not None and prior_speed and speed<prior_speed*.7:degraded=True
        if count>=2 and previous.get('completed_valid',0)>=2 and metrics['valid_completions_per_hour']<previous['valid_completions_per_hour']*.8:
            degraded=True
    if degraded:
        new=max(minimum,target-max(1,target//4));reason='reduce admission after timeout, I/O saturation or degraded measured throughput/latency/simulation speed'
    elif count>=2:
        new=min(maximum,target+max(1,target//4));reason='bounded upward trial after valid completions without measured contention'
    else:
        new=target;reason='hold until a complete outcome window; GPU free space alone does not justify another trial'
    return {'target':new,'baseline':metrics if count>=2 else previous,'reason':reason,'previous_target':target}


def last_sim_time(out):
    path=out/'episode/PHYSICAL_TRACE.jsonl'
    try:
        import json
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-65536));lines=stream.read().splitlines()
        for line in reversed(lines):
            try:return float(json.loads(line)['time_s'])
            except (ValueError,KeyError):continue
    except OSError:return None


def io_pressure():
    try:
        for line in Path('/proc/pressure/io').read_text().splitlines():
            fields=line.split()
            if fields[0]=='full':return float(dict(x.split('=') for x in fields[1:])['avg10'])
    except (OSError,KeyError,ValueError):pass
    return 0.


class Controller:
    def __init__(self,root,purpose,window_s=120,initial=8):
        self.root=Path(root);self.purpose=purpose;self.window_s=window_s;self.last=time.time();self.probes={}
        self.path=self.root/'reports/THROUGHPUT_CONTROL.json'
        previous=read(self.path) if self.path.exists() else {}
        self.state=previous.get('state',{'target':initial,'baseline':None})
        self.history=previous.get('history',[])[-100:]
    @property
    def target(self):return self.state['target']
    def observe(self,db,active):
        epoch=time.time()
        if epoch-self.last<self.window_s:return self.target
        completed=db.execute('SELECT COUNT(*) FROM attempts a JOIN runs r ON r.run_id=a.run_id WHERE r.purpose=? AND a.status="collected" AND a.finished>?',(self.purpose,self.last)).fetchone()[0]
        caps=db.execute('SELECT COUNT(*) FROM attempts a JOIN runs r ON r.run_id=a.run_id WHERE r.purpose=? AND a.finished>? AND a.reason LIKE "%WALLCLOCK_LIMIT_600S%"',(self.purpose,self.last)).fetchone()[0]
        latencies=[];speeds=[]
        for item in active:
            row=db.execute('SELECT output FROM runs WHERE run_id=?',(item['run_id'],)).fetchone()
            if not row or not row[0]:continue
            out=Path(row[0]);sim=last_sim_time(out);before=self.probes.get(item['run_id'])
            if sim is not None:
                if before and sim>=before[1]:speeds.append((sim-before[1])/(epoch-before[0]))
                self.probes[item['run_id']]=(epoch,sim)
            for path in sorted((out/'model').glob('call_*.json'))[-4:]:
                try:latencies.append(float(read(path)['latency_s']))
                except (OSError,ValueError,KeyError):pass
        p95=sorted(latencies)[min(len(latencies)-1,int(len(latencies)*.95))] if latencies else None
        metrics={'window_seconds':epoch-self.last,'completed_valid':completed,'valid_completions_per_hour':completed*3600/(epoch-self.last),
                 'wallclock_timeouts':caps,'model_p95_latency_s':p95,'median_sim_seconds_per_wall_second':statistics.median(speeds) if speeds else None,
                 'io_full_avg10':io_pressure(),'active':len(active),'observed_epoch':epoch}
        self.state=decide(self.state,metrics);self.last=epoch;self.history.append({'state':self.state,'metrics':metrics})
        write(self.path,{'purpose':self.purpose,'state':self.state,'history':self.history[-100:],
                        'policy':'bounded outcome-rate trials; global admission target, no per-GPU task count cap; hard VRAM/disk guards remain active'})
        return self.target
