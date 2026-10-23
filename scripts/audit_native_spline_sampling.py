"""Densification must reproduce both the native samples and their real clock."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from src.planning.native_spline_sampling import sample_native

rows=[]
for directory in ['curobo_curve35','kan_esdf4_j500_c10_edge4']:
    for index in [15,286,315]:
        file=ROOT/'results/sota_simple_v1'/directory/f'trajectory_{index:04d}.npz'
        z=np.load(file);dt=float(z['dt']);duration=(len(z['q'])-1)*dt
        samples=sample_native(z['native_knots'],z['native_knot_dt'],z['native_degree'],z['native_start'],z['native_goal'],dt/4,duration=duration)
        errors={k:float(np.max(abs(samples[k][::4]-z['q' if k=='position' else k]))) for k in samples}
        row=dict(directory=directory,index=index,maximum_errors=errors,duration_error=abs((len(samples['position'])-1)*dt/4-duration))
        row['gate']=max(errors.values())<1e-5 and row['duration_error']<1e-8;rows.append(row)
report=dict(gate=all(r['gate'] for r in rows),rows=rows)
(ROOT/'results/sota_simple_v1/native_sampling_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(report)
