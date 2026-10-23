"""Paired analytic curve comparison, preserving native timing and equal duration.

Raw trial files are never overwritten. Older dense metrics may have used the
native utility's implicit rerounding of duration; only the saved knots and
original execution samples determine these independently recomputed metrics.
"""
from pathlib import Path
import sys,argparse,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from src.planning.native_spline_sampling import sample_native
from src.geometry.portable_scene import PortableScene
from src.geometry.kinematic_jets import tcp_jets,derivative_metrics
from src.planning.paired_statistics import paired_success_interval


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--quality-only',action='store_true',help='Explicit untimed DEVELOPMENT analysis; never allowed for holdout')
    args=p.parse_args()
    model=PortableScene(ROOT/'results/sota_simple_v1/scene_model.json')
    reports=[json.loads((d/'status.json').read_text()) for d in (args.baseline,args.candidate)]
    protocols=[json.loads((d/'protocol.json').read_text()) for d in (args.baseline,args.candidate)]
    rows=[{r['index']:r for r in report['rows']} for report in reports]
    if args.quality_only:
        assert not any(p.get('heldout',False) for p in protocols),'Cannot remove deadlines from confirmation'
        for group in rows:
            for index,original in list(group.items()):
                row=dict(original)
                quality=bool(row.get('native_success',False) and row.get('collision') is None
                    and row.get('terminal_joint_error',float('inf'))<.01
                    and row.get('terminal_tcp_m',float('inf'))<.005
                    and row.get('terminal_orientation_rad',float('inf'))<.02
                    and row.get('stopped',False) and row.get('kinematic_gate',{}).get('passed',False))
                row['success']=quality;group[index]=row
    shared=sorted(set(rows[0])&set(rows[1]));common=[i for i in shared if all(r[i]['success'] for r in rows)]
    def metrics(directory,index):
        file=directory/f'trajectory_{index:04d}.npz';z=np.load(file);duration=(len(z['q'])-1)*float(z['dt'])
        intervals=(z['native_knots'].shape[-2]+int(z['native_degree'])+1)*100
        dt=duration/intervals
        s=sample_native(z['native_knots'],z['native_knot_dt'],int(z['native_degree']),z['native_start'],z['native_goal'],dt,duration=duration)
        tcp=tcp_jets(model,s['position']+model.q_ref,s['velocity'],s['acceleration'],s['jerk'])
        m={'duration_s':duration,'joint':{},'tcp':{},'source_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
        for k,name in enumerate(['position','velocity','acceleration','jerk']):
            if not k:continue
            m['joint'][name]=derivative_metrics(s[name],dt)
            m['tcp'][name]=derivative_metrics(tcp[k],dt,True)
        return m
    paired=[]
    for index in common:
        a,b=[metrics(d,index) for d in (args.baseline,args.candidate)]
        equal={}
        for space in ('joint','tcp'):
            for derivative,order in [('acceleration',2),('jerk',3)]:
                for statistic in ('rms','max'):
                    key=f'{space}_{derivative}_{statistic}'
                    equal[key]=b[space][derivative][statistic]/a[space][derivative][statistic]*(b['duration_s']/a['duration_s'])**order
        paired.append(dict(index=index,difficulty=rows[0][index]['difficulty'],baseline=a,candidate=b,equal_duration_ratios=equal))
    rng=np.random.default_rng(2026091203);summary={}
    if paired:
        for key in paired[0]['equal_duration_ratios']:
            values=np.array([r['equal_duration_ratios'][key] for r in paired]);logs=np.log(np.maximum(values,1e-12))
            bootstrap=np.exp(np.mean(logs[rng.integers(0,len(logs),(10000,len(logs)))],axis=1))
            summary[key]=dict(geometric_mean_ratio=float(np.exp(np.mean(logs))),median_ratio=float(np.median(values)),
                              paired_bootstrap_95ci=np.quantile(bootstrap,[.025,.975]).tolist(),improved_tasks=int(sum(values<1)),paired_tasks=len(values))
    success_delta=np.array([int(rows[1][i]['success'])-int(rows[0][i]['success']) for i in shared])
    delta_bootstrap=success_delta[rng.integers(0,len(shared),(10000,len(shared)))].mean(1) if shared else np.array([np.nan])
    report=dict(status='complete' if all(r['status']=='complete' for r in reports) else 'partial_development_only',
                acceptance_mode='untimed_development_quality_only' if args.quality_only else 'original_trial_success',
                baseline=str(args.baseline),candidate=str(args.candidate),requested_counts=[r['total'] for r in reports],
                completed_counts=[r['completed'] for r in reports],success_counts=[sum(v['success'] for v in group.values()) for group in rows],
                original_trial_success_counts=[r['successes'] for r in reports],
                task_sets_equal=set(rows[0])==set(rows[1]),
                same_manifest=bool(protocols[0].get('manifest_sha256')) and protocols[0].get('manifest_sha256')==protocols[1].get('manifest_sha256'),
                both_heldout=all(p.get('heldout',False) for p in protocols),
                protocol_equality={key:all(key in p for p in protocols) and protocols[0].get(key)==protocols[1].get(key) for key in ('scene_sha256','effective_robot_sha256','world_sha256','planner_budget_s','warmup_tasks','warmup_indices','cpu_validation_workers','gpu_benchmark_serialization','kinematic_gate','collision_acceptance','goal_joint_max_rad','goal_tcp_m','goal_orientation_rad')},
                shared_completed_tasks=len(shared),shared_successes=[sum(r[i]['success'] for i in shared) for r in rows],
                success_difference=float(success_delta.mean()) if shared else None,success_difference_bootstrap_95ci=np.quantile(delta_bootstrap,[.025,.975]).tolist(),
                success_conservative_95ci=paired_success_interval([rows[0][i]['success'] for i in shared],[rows[1][i]['success'] for i in shared]) if shared else None,
                common_successful_tasks=len(common),analytic_normalized_summary=summary,paired=paired,
                interpretation='Ratios < 1 favor candidate. Same-duration comparison equals uniform time scaling to max(native durations); no output path modification. Selection on common successes can be biased; failures and all-request success remain separate. Low impact is kinematic only, not measured force.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:v for k,v in report.items() if k not in ('paired','interpretation')},flush=True)


if __name__=='__main__':main()
