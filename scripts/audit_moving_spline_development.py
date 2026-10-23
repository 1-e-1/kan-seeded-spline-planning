"""Moving-boundary clock DEVELOPMENT sweep, never independent confirmation.

Each horizon is a separate reported experiment, not a hidden retry policy.
The source states come from already-approved development reference curves.
"""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from src.planning.normalized_spline_planner import NormalizedSplinePlanner
from src.planning.benchmark_gpu_lock import BenchmarkGPULock
from src.planning.cubic_execution_curve import CubicExecutionCurve
from src.data.dagger_buffer import ActionHistory
from src.planning.moving_horizon_guess import moving_horizon_guess


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=30)
    p.add_argument('--state-clock',action='store_true')
    p.add_argument('--minimum-duration',type=float,default=.9)
    p.add_argument('--target-offset',type=float,default=0.)
    p.add_argument('--fraction',type=float,default=.35)
    p.add_argument('--contact-only',action='store_true')
    args=p.parse_args();assert not args.output.exists();args.output.mkdir(parents=True)
    original=ROOT/'results/sota_simple_v1/kan_normalized_tcp01_k32_i200_dev350'
    records=json.loads((original/'status.json').read_text())['rows']
    rng=np.random.default_rng(2026091217);selected=[]
    for difficulty,fraction in [('easy',.2),('medium',.6),('hard',.2)]:
        pool=[r['index'] for r in records if r['success'] and r['difficulty']==difficulty]
        selected.extend((int(i),difficulty) for i in rng.choice(pool,int(args.count*fraction),replace=False))
    assert len(selected)==args.count
    protocol=dict(scope='Conditional moving-reference DEVELOPMENT; not deployment success rate or confirmation',
        count=args.count,selected=selected,fractions=[args.fraction],duration_factors=[1.] if args.state_clock else [1.,1.5,2.],
        state_clock=args.state_clock,minimum_duration=args.minimum_duration,target_offset_rad=args.target_offset,contact_only=args.contact_only,
        history='Synthetic approved-reference replay; no simulator execution',
        seed=2026091217,no_retry_selection=True)
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    planner=NormalizedSplinePlanner(ROOT,ROOT/'results/sota_simple_v1/confirmation_decision_v1.json',
                                    contact_only_verification=args.contact_only)
    manifest=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())
    for index in [15,128]*2:
        task=manifest['tasks'][index]['task']['left'];planner.plan(task['q_start'],task['q_goal'],index=index)
    rows=[]
    for index,difficulty in selected:
        task=manifest['tasks'][index]['task']['left']
        with np.load(original/f'trajectory_{index:04d}.npz') as old:
            duration=(len(old['q'])-1)*float(old['dt'])
            curve=CubicExecutionCurve(old['native_knots']+planner.model.q_ref,
                old['native_start']+planner.model.q_ref,old['native_goal']+planner.model.q_ref,duration)
        stamp=np.floor(args.fraction*duration*30)/30;state=curve.evaluate(stamp)
        goal=np.asarray(task['q_goal'])+np.random.default_rng(2026091219+index).uniform(-args.target_offset,args.target_offset,8)
        history=ActionHistory(128);times=np.arange(max(0,int(round(stamp*30))-128),int(round(stamp*30))+1)/30
        for action in np.diff(curve.evaluate(times)['position'],axis=0):history.append(action)
        before=history.padded()
        for factor in protocol['duration_factors']:
            requested_duration=(moving_horizon_guess(state['position'],goal,state['velocity'],state['acceleration'],minimum_duration=args.minimum_duration)
                                if args.state_clock else (duration-stamp)*factor)
            result=planner.plan(state['position'],goal,index=index,history=history,
                start_velocity=state['velocity'],start_acceleration=state['acceleration'],
                fixed_duration=requested_duration)
            assert all(np.array_equal(a,b) for a,b in zip(before,history.padded()))
            arrays=result.pop('arrays',None);result.pop('curve',None)
            name=f'moving_{index:04d}_factor{factor:g}'
            if arrays is not None:np.savez_compressed(args.output/(name+'.npz'),**arrays)
            row=dict(index=index,difficulty=difficulty,factor=factor,reference_duration=duration,
                     reference_timestamp=stamp,**result)
            rows.append(row)
            (args.output/(name+'.json')).write_text(json.dumps(row,indent=2),encoding='utf-8')
            (args.output/'status.json').write_text(json.dumps(dict(status='running',rows=rows),indent=2),encoding='utf-8')
            print(index,factor,result['success'],result['reason'],flush=True)
    summary={str(f):dict(count=sum(r['factor']==f for r in rows),success=sum(r['success'] for r in rows if r['factor']==f))
             for f in protocol['duration_factors']}
    sources={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in
        ['scripts/audit_moving_spline_development.py','src/planning/normalized_spline_planner.py',
         'src/planning/fixed_spline_clock.py','src/planning/moving_horizon_guess.py']}
    (args.output/'status.json').write_text(json.dumps(dict(status='complete',summary=summary,rows=rows,
        sources_sha256=sources,decision_sha256=planner.decision_sha256),indent=2),encoding='utf-8')
    print(summary,flush=True)


if __name__=='__main__':
    with BenchmarkGPULock(ROOT/'results/sota_simple_v1/benchmark_gpu.lock'):main()
