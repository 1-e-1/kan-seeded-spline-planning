"""Audit native basis with moving boundaries before any online execution.

Prescribed boundary derivatives do not establish path feasibility. No scene,
controller, checkpoint or existing static trial is modified by this audit.
"""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from src.planning.native_spline_sampling import sample_native


def main():
    rows=[];rng=np.random.default_rng(2026091217)
    file=ROOT/'results/sota_simple_v1/kan_k32_j6500_graph_dev350/trajectory_0015.npz'
    with np.load(file) as z:
        for degree,stop_goal in ((3,True),(5,True),(3,False),(5,False)):
            segments=z['native_knots'].shape[-2]+degree+1
            dt=.005;duration=segments*20*dt
            boundary={f'{end}_{name}':rng.uniform(-magnitude,magnitude,8)
                      for end in ('start','goal') for name,magnitude in [('velocity',.3),('acceleration',.5),('jerk',1.)]}
            if stop_goal:
                for name in ('velocity','acceleration','jerk'):boundary['goal_'+name]=np.zeros(8)
            sampled=sample_native(z['native_knots'],z['native_knot_dt'],degree,z['native_start'],z['native_goal'],dt,duration=duration,**boundary)
            dense=sample_native(z['native_knots'],z['native_knot_dt'],degree,z['native_start'],z['native_goal'],dt/4,duration=duration,**boundary)
            consistency={k:float(np.max(abs(dense[k][::4]-sampled[k]))) for k in sampled}
            endpoint_error={f'{end}_{name}':float(np.max(abs(sampled[name][i]-boundary[f'{end}_{name}'])))
                            for i,end in [(0,'start'),(-1,'goal')] for name in ('velocity','acceleration')}
            endpoint_error.update(start_position=float(np.max(abs(sampled['position'][0]-z['native_start']))),
                                  goal_position=float(np.max(abs(sampled['position'][-1]-z['native_goal']))))
            rows.append(dict(degree=degree,stop_goal=stop_goal,sampling_errors=consistency,boundary_errors=endpoint_error,
                             gate=max([*consistency.values(),*endpoint_error.values()])<2e-4))
    report=dict(rows=rows,gate=all(r['gate'] for r in rows),moving_start_rest_goal_gate=all(r['gate'] for r in rows if r['stop_goal']),scope='boundary and sampling identity only; not online feasibility or jerk continuity certification')
    (ROOT/'results/sota_simple_v1/moving_spline_boundary_audit_v2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':main()
