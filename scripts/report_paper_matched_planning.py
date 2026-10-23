"""Per-seed paired KAN/MLP analysis; never pool 3 x 350 as independent tasks."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from scipy.stats import beta


def paired_success(treatment,control,alpha=.05):
    a,b=np.asarray(treatment),np.asarray(control)
    if a.ndim!=1 or a.shape!=b.shape or not len(a) or not np.isin(a,[0,1]).all() or not np.isin(b,[0,1]).all():
        raise ValueError('Matching nonempty binary paired outcomes required')
    if not 0<alpha<1:raise ValueError('Invalid significance level')
    d=a.astype(int)-b.astype(int);n=len(d)
    def interval(k):
        return (0. if k==0 else float(beta.ppf(alpha/4,k,n-k+1)),
                1. if k==n else float(beta.ppf(1-alpha/4,k+1,n-k)))
    plus=interval(int(sum(d==1)));minus=interval(int(sum(d==-1)))
    return dict(tasks=n,difference_pp=float(d.mean()*100),treatment_only=int(sum(d==1)),
                control_only=int(sum(d==-1)),confidence=1-alpha,
                conservative_interval_pp=[100*(plus[0]-minus[1]),100*(plus[1]-minus[0])])


def quality_vector(row):
    values={};duration=row['analytic_smoothness']['duration_s']
    if not np.isfinite(duration) or duration<=0:raise ValueError('Invalid execution duration')
    for space in ('joint','tcp'):
        for derivative,order in [('acceleration',2),('jerk',3)]:
            for measure,key in [('rms','rms'),('peak','max')]:
                raw=(row['analytic_smoothness'][derivative+'_'+key] if space=='joint'
                     else row['tcp_analytic_smoothness'][derivative][key])
                if not np.isfinite(raw) or raw<0:raise ValueError('Invalid analytic motion metric')
                values[f'{space}_{derivative}_{measure}']=raw*duration**order
    return values


def quality_ratios(treatment,control,indices,seed):
    if not indices:return dict(common_successes=0,metrics=None)
    a=[quality_vector(treatment[i]) for i in indices];b=[quality_vector(control[i]) for i in indices]
    rng=np.random.default_rng(seed);draws=rng.integers(0,len(indices),(10000,len(indices)))
    metrics={}
    for key in a[0]:
        left=np.array([v[key] for v in a]);right=np.array([v[key] for v in b])
        # Do not add epsilon to manufacture ratios for stationary trajectories.
        if np.any(left<=0) or np.any(right<=0):
            metrics[key]=dict(ratio=None,reason='zero_metric; no epsilon or silent case deletion')
            continue
        logs=np.log(left)-np.log(right)
        metrics[key]=dict(ratio=float(np.exp(logs.mean())),
                         exploratory_bootstrap95=np.exp(np.quantile(logs[draws].mean(1),[.025,.975])).tolist())
    return dict(common_successes=len(indices),indices=indices,metrics=metrics)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise RuntimeError('Fresh analysis directory required')
    state=json.loads((args.suite/'status.json').read_text())
    if state.get('status')!='complete' or len(state['completed'])!=17:
        raise RuntimeError('Wait for the full frozen suite, not a favorable subset')
    protocol=json.loads((args.suite/'protocol.json').read_text())
    completed={r['name']:r for r in state['completed']}
    sources={}
    for stage in protocol['stages']:
        path=Path(stage['result']);digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==completed[stage['name']]['result_sha256'], 'Suite result changed'
        sources[str(path)]=digest
    bindings=json.loads((args.suite/'trained_actor_bindings.json').read_text())
    action_protocol=json.loads((args.suite/'actions/protocol.json').read_text())
    training={r['name']:r for r in action_protocol['actors']}
    data={};groups={}
    for seed in (42,43,44):
        for family in ('kan','mlp'):
            name=f'{family}_seed{seed}'
            result=json.loads((args.suite/'matched_planning'/name/'status.json').read_text())
            assert result['status']=='complete' and len(result['rows'])==350
            data[name]={r['index']:r for r in result['rows']}
            assert set(data[name])==set(range(350))
            actor=json.loads((args.suite/'matched_planning'/name/'actor.json').read_text())
            assert actor['actor_family']==family
            assert actor['sha256']==bindings[name]['checkpoint_sha256']==training[name]['checkpoint_sha256']
            sources[str(args.suite/'matched_planning'/name/'actor.json')]=hashlib.sha256(
                (args.suite/'matched_planning'/name/'actor.json').read_bytes()).hexdigest()
            timing={}
            for field in ('planner_wall_s','independent_validation_wall_s','end_to_end_wall_s'):
                values=[r[field] for r in result['rows'] if field in r]
                assert all(np.isfinite(v) and v>=0 for v in values)
                timing[field]=dict(count=len(values),median=float(np.median(values)),p95=float(np.quantile(values,.95))) if values else None
            groups[name]=dict(successes=sum(r['success'] for r in result['rows']),tasks=350,timing=timing,
                              actor_sha256=actor['sha256'],selected_epoch=training[name]['selected_epoch'],
                              training_updates=training[name]['updates'],
                              parameter_count=training[name]['protocol']['parameter_count'],
                              trainable_parameter_count=training[name]['protocol']['trainable_parameter_count'])
    common_all=[i for i in range(350) if all(rows[i]['success'] for rows in data.values())]
    contrasts={}
    for seed in (42,43,44):
        a,b=data[f'kan_seed{seed}'],data[f'mlp_seed{seed}']
        common=[i for i in range(350) if a[i]['success'] and b[i]['success']]
        contrasts[str(seed)]=dict(success=paired_success([a[i]['success'] for i in range(350)],
            [b[i]['success'] for i in range(350)],alpha=.05/3),
            paired_common_quality=quality_ratios(a,b,common,2026091400+seed),
            all_six_common_quality=quality_ratios(a,b,common_all,2026091500+seed))
    report=dict(groups=groups,contrasts=contrasts,common_all_six=len(common_all),input_hashes=sources,
        scope='Development task comparison; three paired training seeds, not 1050 independent tasks',
        success_inference='Per-seed paired conservative intervals, Bonferroni across three seed contrasts',
        smoothness_inference='Exploratory individual 95% bootstrap intervals, no multiplicity correction; conditioned on common success',
        timing='All available records including failures; missing values counted, not replaced by zero',
        limitations=['Three training seeds do not support strong claims about the training-seed population.',
                     'Normalized quality is equal-duration comparison only, never an executed retiming.',
                     'Matched updates/data, not parameter counts, training seconds or anchor training history.',
                     'No new best checkpoint, no test-based retuning, no global SOTA or real-hardware claim.'])
    resources=args.suite/'RESOURCE_OBSERVATIONS.json'
    if resources.exists():
        report['resource_observations']=json.loads(resources.read_text(encoding='utf-8'))
        report['input_hashes'][str(resources)]=hashlib.sha256(resources.read_bytes()).hexdigest()
        report['limitations'].append('Recorded external workload overlap; strict suite resource isolation is unproven. Timing and time-budgeted success cannot establish isolated runtime fairness.')
    args.output.mkdir(parents=True)
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    lines=['# KAN / MLP 匹配训练规划对照','',
           '固定 350 个开发请求，三个训练种子分别配对；不将重复任务池化为 1,050 个独立样本。','',
           '| 种子 | KAN 成功 | MLP 成功 | 成功率差 pp | 三组校正后的保守区间 pp |',
           '|---|---:|---:|---:|---:|']
    for seed in (42,43,44):
        result=contrasts[str(seed)]['success'];lo,hi=result['conservative_interval_pp']
        lines.append(f'| {seed} | {groups[f"kan_seed{seed}"]["successes"]}/350 | {groups[f"mlp_seed{seed}"]["successes"]}/350 | {result["difference_pp"]:+.3f} | [{lo:+.3f}, {hi:+.3f}] |')
    lines+=['',f'六组共同成功 {len(common_all)} 项。八项等时长运动指标及逐种子共同成功结果见 report.json。',
            '', '运动指标区间仅为探索性、未经多重比较校正；共同成功条件可能引入选择偏差。不能由表中点估计宣称 KAN 必需或显著优于 MLP。',
            '', '三种子不能充分刻画训练随机性，训练参数量和耗时也未匹配。热规划、独立验收与端到端时间分开报告，不将验收队列耗时称为在线规划延迟。']
    if resources.exists():
        lines+=['', '资源限制：套件存在外部任务并发记录，详见 report.json。不能宣称全程严格隔离；耗时与限时成功率比较需考虑该限制，尚不足以完成公平时延验证。']
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':main()
