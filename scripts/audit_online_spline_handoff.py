"""Wall-clock reference-loop engineering audit; NOT Isaac/hardware execution.

Frozen KAN + one optimization request. A late or invalid replacement is
rejected without repair; the approved original curve continues to rest.
Future reference history is explicitly predicted, never measured telemetry.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import sys,json,argparse,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from src.planning.normalized_spline_planner import NormalizedSplinePlanner
from src.planning.benchmark_gpu_lock import BenchmarkGPULock
from src.planning.cubic_execution_curve import CubicExecutionCurve
from src.planning.c2_trajectory_buffer import C2TrajectoryBuffer
from src.planning.moving_horizon_guess import moving_horizon_guess
from src.planning.contact_validation_pool import ContactValidationPool
from src.data.dagger_buffer import ActionHistory


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lead-seconds',type=float,default=1.6)
    p.add_argument('--parallel-verification',action='store_true')
    p.add_argument('--all-development-cases',action='store_true')
    args=p.parse_args();assert not args.output.exists();args.output.mkdir(parents=True)
    assert np.isfinite(args.lead_seconds) and args.lead_seconds>0
    indices=[341,15,329,13,227,165,159,198,51,69,286,63]
    if args.all_development_cases:
        protocol30=json.loads((ROOT/'results/sota_simple_v1/moving_clock_dev30_v1/protocol.json').read_text())
        indices=[i for i,_ in protocol30['selected']]
    protocol=dict(indices=indices,lead_seconds=args.lead_seconds,control_hz=30,
        scope='Selected DEVELOPMENT engineering cases incl. known difficult cases; no independent success-rate claim',
        environment='CPU wall-clock ideal reference execution; no Isaac, payload or physical dynamics',
        target_offset_rad=.05,minimum_duration_s=2.7,retries=0,parallel_verification=args.parallel_verification,
        history='Executed reference samples kept separately from predicted future-reference history')
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    audit=json.loads((ROOT/'results/sota_simple_v1/moving_cpu_curve_identity_v1.json').read_text())
    assert audit['gate'] and audit['source_sha256']==hashlib.sha256((ROOT/'src/planning/cubic_execution_curve.py').read_bytes()).hexdigest()
    planner=NormalizedSplinePlanner(ROOT,ROOT/'results/sota_simple_v1/confirmation_decision_v1.json',contact_only_verification=True)
    manifest=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())
    original=ROOT/'results/sota_simple_v1/kan_normalized_tcp01_k32_i200_dev350'
    reference={r['index']:r for r in json.loads((original/'status.json').read_text())['rows']}
    for index in [15,128]*2:
        task=manifest['tasks'][index]['task']['left'];planner.plan(task['q_start'],task['q_goal'],index=index)
    rows=[]
    if args.parallel_verification:
        pool_audit=json.loads((ROOT/'results/sota_simple_v1/contact_validation_pool_audit_v2.json').read_text())
        assert pool_audit['gate']
        for name,sha in pool_audit['sources_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    context=ContactValidationPool(ROOT,sys.executable,workers=2) if args.parallel_verification else nullcontext(None)
    with context as verification_pool,ThreadPoolExecutor(max_workers=1) as worker:
        planner.verification_pool=verification_pool
        for index in indices:
            assert reference[index]['success'];task=manifest['tasks'][index]['task']['left']
            with np.load(original/f'trajectory_{index:04d}.npz') as old:
                duration=(len(old['q'])-1)*float(old['dt'])
                curve=CubicExecutionCurve(old['native_knots']+planner.model.q_ref,
                    old['native_start']+planner.model.q_ref,old['native_goal']+planner.model.q_ref,duration)
            buffer=C2TrajectoryBuffer(curve);switch=args.lead_seconds;boundary=buffer.reference_at(switch)
            predicted=ActionHistory(128);times=np.arange(0,int(round(switch*30))+1)/30
            positions=np.array([buffer.reference_at(t)['position'] for t in times])
            for dq in np.diff(positions,axis=0):predicted.append(dq)
            predicted_before=predicted.padded();executed=ActionHistory(128)
            goal=np.asarray(task['q_goal'])+np.random.default_rng(2026091219+index).uniform(-.05,.05,8)
            horizon=moving_horizon_guess(boundary['position'],goal,boundary['velocity'],boundary['acceleration'],minimum_duration=2.7)
            begun=time.perf_counter()
            future=worker.submit(planner.plan,boundary['position'],goal,index=index,history=predicted,
                start_velocity=boundary['velocity'],start_acceleration=boundary['acceleration'],fixed_duration=horizon)
            handled=False;outcome=None;handoff=None;ready=None;end=duration;samples=[];tick=0;previous=None
            while True:
                now=time.perf_counter()-begun
                if future.done() and not handled:
                    outcome=future.result();ready=time.perf_counter()-begun;handled=True
                    assert all(np.array_equal(a,b) for a,b in zip(predicted_before,predicted.padded()))
                    if outcome['success']:
                        handoff=buffer.schedule(outcome['curve'],switch,ready,independently_verified=True)
                        if handoff['accepted']:end=switch+outcome['curve'].duration
                    else:handoff=dict(accepted=False,reason='planner_rejected',planner_reason=outcome['reason'])
                now=time.perf_counter()-begun
                if now>=tick/30:
                    state=buffer.advance(now)
                    if previous is not None:executed.append(state['position']-previous)
                    previous=state['position'].copy()
                    samples.append(dict(time=now,scheduled=tick/30,**state))
                    # No fabricated intermediate commands after a scheduling gap.
                    tick=max(tick+1,int(np.floor(now*30))+1)
                if handled and now>=end+.05:break
                if now>60:raise TimeoutError('Reference audit exceeded 60s; no replacement forced')
                time.sleep(min(.003,max(0,tick/30-now)))
            arrays=outcome.pop('arrays',None);outcome.pop('curve',None)
            if arrays is not None:np.savez_compressed(args.output/f'candidate_{index:04d}.npz',**arrays)
            data={k:np.array([s[k] for s in samples]) for k in samples[0]}
            np.savez_compressed(args.output/f'reference_loop_{index:04d}.npz',**data)
            error=float(np.max(abs((data['position'][-1]-goal+np.pi)%(2*np.pi)-np.pi)))
            step_dt=np.diff(data['time']);jitter=data['time']-data['scheduled']
            record=dict(index=index,requested_switch_s=switch,ready_s=ready,
                moving_boundary=bool(np.max(abs(boundary['velocity']))>.001 or np.max(abs(boundary['acceleration']))>.01),
                handoff=handoff,updated_goal_reached=bool(handoff['accepted'] and error<.01),terminal_joint_error=error,
                command_count=len(samples),command_dt_max_s=float(step_dt.max()),
                scheduling_lateness_p95_s=float(np.quantile(jitter,.95)),
                max_reference_acceleration=float(np.max(abs(data['acceleration']))),
                max_reference_jerk=float(np.max(abs(data['jerk']))),
                executed_history_valid=int(executed.padded()[1].sum()),planner=outcome)
            rows.append(record)
            (args.output/'status.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2),encoding='utf-8')
            print(index,record['moving_boundary'],handoff,ready,flush=True)
    summary=dict(count=len(rows),accepted=sum(r['handoff']['accepted'] for r in rows),
        updated_goal_reached=sum(r['updated_goal_reached'] for r in rows),moving_boundaries=sum(r['moving_boundary'] for r in rows),
        scope=protocol['scope'],no_unsafe_or_late_forced_handoff=all(not r['handoff']['accepted'] or
            (r['planner']['success'] and r['ready_s']<=r['requested_switch_s']) for r in rows))
    sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ['scripts/audit_online_spline_handoff.py','src/planning/normalized_spline_planner.py',
         'src/planning/fixed_spline_clock.py','src/planning/moving_horizon_guess.py','src/planning/c2_trajectory_buffer.py',
         'src/planning/contact_validation_pool.py','scripts/contact_validation_server.py']}
    (args.output/'status.json').write_text(json.dumps(dict(status='complete',summary=summary,rows=rows,sources_sha256=sources),indent=2),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':
    with BenchmarkGPULock(ROOT/'results/sota_simple_v1/benchmark_gpu.lock'):main()
