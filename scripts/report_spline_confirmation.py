"""Readable report/figure from COMPLETE paired results, never interim tuning."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--paired',type=Path,required=True)
    p.add_argument('--assessment',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replication',action='store_true',help='Label secondary replication separately from the prespecified primary test')
    args=p.parse_args();assert not args.output.exists(),'Keep earlier reports immutable'
    paired=json.loads(args.paired.read_text());assessment=json.loads(args.assessment.read_text())
    assert paired['status']=='complete'
    assert assessment['paired_source_sha256']==hashlib.sha256(args.paired.read_bytes()).hexdigest()
    directories=[Path(paired[name]) for name in ('baseline','candidate')]
    states=[json.loads((d/'status.json').read_text()) for d in directories]
    assert all(s['status']=='complete' and s['completed']==s['total'] for s in states)
    assert paired['task_sets_equal'] and paired['same_manifest']
    args.output.mkdir(parents=True)
    groups=['all','easy','medium','hard'];counts=[]
    for group in groups:
        counts.append([(sum(r['success'] for r in s['rows'] if group=='all' or r['difficulty']==group),
                        sum(group=='all' or r['difficulty']==group for r in s['rows'])) for s in states])
    metrics=paired['analytic_normalized_summary'];keys=list(metrics)
    fig,axes=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    x=np.arange(len(groups));colors=['#59656f','#176b9b']
    for role,label in enumerate(('cuRoboV2 baseline','KAN + normalized spline')):
        values=[100*c[role][0]/c[role][1] for c in counts]
        bars=axes[0].bar(x+(role-.5)*.36,values,width=.36,color=colors[role],label=label)
        for bar,(success,total) in zip(bars,[c[role] for c in counts]):
            axes[0].text(bar.get_x()+bar.get_width()/2,bar.get_height()+1+role*3,f'{success}/{total}',ha='center',fontsize=8)
    axes[0].set(xticks=x,xticklabels=groups,ylim=(0,112),ylabel='On-time accepted requests (%)',title='All requested tasks retained')
    axes[0].legend(loc='lower left',fontsize=8);axes[0].grid(axis='y',alpha=.2)
    centers=np.array([metrics[k]['geometric_mean_ratio'] for k in keys])
    bounds=np.array([metrics[k]['paired_bootstrap_95ci'] for k in keys])
    labels=[k.replace('joint_','Joint ').replace('tcp_','TCP ').replace('acceleration','accel.').replace('_',' ') for k in keys]
    axes[1].errorbar(centers,np.arange(len(keys)),xerr=np.stack((centers-bounds[:,0],bounds[:,1]-centers)),
                     fmt='o',capsize=3,color=colors[1])
    axes[1].axvline(1,color='#bb4444',linestyle='--');axes[1].set(yticks=np.arange(len(keys)),yticklabels=labels,
        xlabel='Candidate / baseline; equal duration (lower is better)',title=f'Common successes only: n={paired["common_successful_tasks"]}')
    axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.2)
    fig.suptitle(('Independent replication (secondary)' if args.replication else 'Frozen static confirmation')
                 if paired['both_heldout'] else 'DEVELOPMENT ONLY - not independent confirmation')
    fig.savefig(args.output/'comparison.png',dpi=180);plt.close(fig)
    n=paired['shared_completed_tasks'];ci=paired['success_conservative_95ci']['ci']
    independent=paired['both_heldout'];passed=assessment['gate']
    lines=['# 简洁轨迹算法：'+(('独立复验（次要结果）' if args.replication else '独立静态确认') if independent else '开发结果（非独立确认）'),' ',
        '结论：'+('通过预设静态验收。' if passed else '未通过全部预设静态验收。')+
        '这不是全领域 SOTA、实时闭环或递送完成声明。','',
        '## 成功率','', '| 难度 | cuRoboV2 对照 | KAN + 归一化样条 |','|---|---:|---:|']
    for group,count in zip(groups,counts):
        lines.append('| '+group+' | '+' | '.join(f'{a}/{b}（{a/b:.2%}）' for a,b in count)+' |')
    for label,state in zip(('对照','候选'),states):
        invalid=sum(r['reason'] in ('invalid_start','invalid_goal') for r in state['rows'])
        lines.append(f'\n{label}：无效端点请求 {invalid}/{state["total"]}，仍记作失败；有效端点请求成功 {state["successes"]}/{state["total"]-invalid}。')
    lines.extend(['',f'同一 {n} 个请求；配对成功率差 {paired["success_difference"]*100:+.3f} 个百分点，保守 95% 区间 [{ci[0]*100:.3f}, {ci[1]*100:.3f}]。',
        '', '## 统一时长后的平滑性','',
        '下列比值小于 1 表示候选更低，只在共同成功任务上配对；不因某方法失败就静默从成功率分母剔除。',
        '', '| 指标 | 几何均值比值 | 个别配对 bootstrap 95% 区间 |','|---|---:|---:|'])
    for key,value in metrics.items():
        lo,hi=value['paired_bootstrap_95ci'];lines.append(f'| {key} | {value["geometric_mean_ratio"]:.5f} | [{lo:.5f}, {hi:.5f}] |')
    lines.extend(['','## 验收项目','', '| 项目 | 通过 |','|---|---|'])
    for name,ok in assessment['checks'].items():lines.append(f'| {name} | {"是" if ok else "否"} |')
    lines.extend(['','## 实测耗时（秒）','',
        '| 统计量 | 对照 median / P95 / P99 | 候选 median / P95 / P99 |','|---|---:|---:|'])
    for key in ('planner_wall_s','independent_validation_wall_s','end_to_end_wall_s'):
        cells=[]
        for state in states:
            values=[r[key] for r in state['rows'] if key in r]
            cells.append(' / '.join(f'{v:.3f}' for v in np.quantile(values,[.5,.95,.99])) if values else '未记录')
        lines.append('| '+key+' | '+' | '.join(cells)+' |')
    lines.extend(['','求解及几何复核统计仅使用有对应耗时记录的请求，包含失败；无效端点通常不会调用求解器。end_to_end_wall_s 还包含离线批处理的验证排队，受任务提交速率影响，不能据此直接推断在线执行延迟优势。'])
    lines.extend(['','## 算法与适用边界','',
        '冻结 KAN-Transformer 生成短虚拟运动学初值，一个多初值 B-spline 优化器同时处理碰撞、终点和导数限制。形状损失对均匀改变时长不变；没有串联 RRT、伺服、shortcut 或输出后重定时修复。',
        '', '对照是本机同场景复现的 [cuRoboV2](https://github.com/NVlabs/curobo/releases)，不是直接引用其其他机器人/任务集的论文成绩。',
        '', '2 s 预算只计求解，完整几何验收及验证排队另计，不能写成 2 s 内已取得安全可执行路径。低冲击只用运动学加速度和 jerk 代理，不代表测得接触力或力矩。',
        '', '该样本分布按简单/中等/困难 20/60/20 固定，不能当作已经测量的实际工况频率；端点安全不代表预先证明路径存在。共同成功条件下的平滑比较仍有选择效应，已另报全请求成功率。',
        '', '三次样条是 C2，不保证结点处 jerk 连续。在线状态衔接、实际执行历史和带吸附物体的递送需另行验证。',
        '', '![比较图](comparison.png)','',
        f'数据 SHA256：`{hashlib.sha256(args.paired.read_bytes()).hexdigest()}`。'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(args.output/'REPORT.md')


if __name__=='__main__':main()
