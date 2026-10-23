"""Independent adapter replay + moving-start DEVELOPMENT audit, not holdout."""
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


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decision',type=Path,default=ROOT/'results/sota_simple_v1/confirmation_decision_v1.json')
    p.add_argument('--moving-duration-guess',action='store_true',help='Use approved reference remaining time only for moving-start initialization')
    p.add_argument('--moving-fixed-clock',action='store_true',help='Fixed aligned horizon, no post-optimization clock scaling')
    args=p.parse_args();assert not args.output.exists();args.output.mkdir(parents=True)
    planner=NormalizedSplinePlanner(ROOT,args.decision)
    manifest=json.loads((ROOT/'results/astronaut_full_v3/evaluation_350_seed985350.json').read_text())
    original=ROOT/'results/sota_simple_v1/kan_normalized_tcp01_k32_i200_dev350'
    reference={r['index']:r for r in json.loads((original/'status.json').read_text())['rows']}
    def save(name,outcome):
        arrays=outcome.pop('arrays',None);outcome.pop('curve',None)
        if arrays is not None:np.savez_compressed(args.output/(name+'.npz'),**arrays)
        (args.output/(name+'.json')).write_text(json.dumps(outcome,indent=2),encoding='utf-8')
    for ordinal,index in enumerate([15,128]*2):
        task=manifest['tasks'][index]['task']['left'];result=planner.plan(task['q_start'],task['q_goal'],index=index)
        save(f'warmup_{ordinal}',result)
    comparisons=[]
    for index in (15,32,94,164,219,315):
        task=manifest['tasks'][index]['task']['left'];result=planner.plan(task['q_start'],task['q_goal'],index=index)
        with np.load(original/f'trajectory_{index:04d}.npz') as old:
            errors={k:float(np.max(abs(result['arrays'][k]-old[k]))) if result['arrays'][k].shape==old[k].shape else None
                    for k in ('q','native_knots','native_knot_dt','dense_q')}
        gate=result['success']==reference[index]['success'] and all(v is not None and v<1e-5 for v in errors.values())
        comparisons.append(dict(index=index,gate=gate,success=result['success'],maximum_errors=errors))
        save(f'static_{index:04d}',result);print('STATIC_IDENTITY',index,gate,flush=True)
    moving=[]
    for index in (15,94):
        task=manifest['tasks'][index]['task']['left']
        with np.load(original/f'trajectory_{index:04d}.npz') as old:
            duration=(len(old['q'])-1)*float(old['dt'])
            curve=CubicExecutionCurve(old['native_knots']+planner.model.q_ref,
                old['native_start']+planner.model.q_ref,old['native_goal']+planner.model.q_ref,duration)
        stamp=np.floor(.35*duration*30)/30;state=curve.evaluate(stamp)
        history=ActionHistory(128)
        times=np.arange(max(0,int(round(stamp*30))-128),int(round(stamp*30))+1)/30
        positions=curve.evaluate(times)['position']
        for action in np.diff(positions,axis=0):history.append(action)
        before=history.padded()
        result=planner.plan(state['position'],task['q_goal'],index=index,history=history,
                            start_velocity=state['velocity'],start_acceleration=state['acceleration'],
                            initial_duration=duration-stamp if args.moving_duration_guess else None,
                            fixed_duration=duration-stamp if args.moving_fixed_clock else None)
        assert all(np.array_equal(a,b) for a,b in zip(before,history.padded()))
        moving.append(dict(index=index,success=result['success'],reason=result['reason'],start_errors=result.get('start_errors'),
            planner_wall_s=result.get('planner_wall_s'),end_to_end_wall_s=result.get('end_to_end_wall_s'),
            history='synthetic approved-reference replay for adapter audit; not simulator execution telemetry'))
        save(f'moving_{index:04d}',result);print('MOVING_START',index,moving[-1],flush=True)
    index=32;task=manifest['tasks'][index]['task']['left']
    result=planner.plan(task['q_start'],task['q_goal'],index=index)
    with np.load(original/f'trajectory_{index:04d}.npz') as old:
        errors={k:float(np.max(abs(result['arrays'][k]-old[k]))) if result['arrays'][k].shape==old[k].shape else None
                for k in ('q','native_knots','native_knot_dt','dense_q')}
    gate=result['success']==reference[index]['success'] and all(v is not None and v<1e-5 for v in errors.values())
    comparisons.append(dict(index=index,after_moving=True,gate=gate,success=result['success'],maximum_errors=errors))
    save('static_after_moving',result)
    report=dict(static_identity_gate=all(r['gate'] for r in comparisons),static=comparisons,moving=moving,
        source_sha256=hashlib.sha256((ROOT/'src/planning/normalized_spline_planner.py').read_bytes()).hexdigest(),
        decision_sha256=planner.decision_sha256,scope='Adapter and moving-boundary development only. No payload, simulator, real-time handoff, or independent success-rate certification.')
    (args.output/'audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report,flush=True)


if __name__=='__main__':
    with BenchmarkGPULock(ROOT/'results/sota_simple_v1/benchmark_gpu.lock'):main()
