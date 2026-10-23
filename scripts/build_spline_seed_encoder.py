"""Construct and audit a fixed seed representation map from the native basis."""
from pathlib import Path
import sys,json,argparse,hashlib
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from src.planning.native_spline_sampling import sample_native
from src.planning.spline_seed_encoding import build_encoding,encode_seed
from src.planning.benchmark_gpu_lock import BenchmarkGPULock


def main():
    p=argparse.ArgumentParser();p.add_argument('--knots',type=int,default=32);p.add_argument('--degree',type=int,default=3)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    n=args.knots;segments=n+args.degree+1;h=segments*8+1;dt=1/(h-1)
    zero=np.zeros((n,8));start=np.zeros(8);goal=np.zeros(8)
    def sample(c,a,b):return sample_native(c,np.array([1/segments]),args.degree,a,b,dt,duration=1.)
    sr=sample(zero,np.ones(8),goal)['position'][:,0].astype(float)
    gr=sample(zero,start,np.ones(8))['position'][:,0].astype(float)
    basis=np.zeros((h,n))
    for offset in range(0,n,8):
        pulses=zero.copy();count=min(8,n-offset)
        for j in range(count):pulses[offset+j,j]=1
        basis[:,offset:offset+count]=sample(pulses,start,goal)['position'][:,:count]
    # Native rest boundaries impose a constant first/last knot interval.
    u=np.clip((np.linspace(0,1,h)-1/segments)/(1-2/segments),0,1)
    phase=10*u**3-15*u**4+6*u**5
    encoding=build_encoding(basis,sr,gr,phase)
    rng=np.random.default_rng(2026091223);errors=[]
    for _ in range(4):
        a=rng.normal(size=8);b=rng.normal(size=8);c=rng.normal(size=(n,8))
        expected=basis@c+sr[:,None]*a+gr[:,None]*b
        errors.append(float(np.max(abs(sample(c,a,b)['position']-expected))))
    linear=np.linspace(0,1,n)[:,None]*np.ones((1,8))
    encoded=encode_seed(linear,np.zeros(8),np.ones(8),encoding)
    curve=sample(encoded,np.zeros(8),np.ones(8))
    positional_error=float(np.max(abs(curve['position'][:,0]-phase)))
    partition_error=float(np.max(abs(basis.sum(1)+sr+gr-1)))
    report=dict(knots=n,degree=args.degree,rank=encoding['rank'],linearity_errors=errors,
        partition_error=partition_error,quintic_progress_fit_max_error=positional_error,
        gate=max(errors)<3e-6 and partition_error<1e-6 and positional_error<.002,
        scope='static seed encoding only; no claim that a curved path minimizes integrated jerk or is collision free')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    assert not args.output.exists(),'Keep audited encodings immutable'
    np.savez_compressed(args.output,**encoding,position_basis=basis,start_response=sr,goal_response=gr,phase=phase)
    report['matrix_sha256']=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(report)


if __name__=='__main__':
    with BenchmarkGPULock(ROOT/'results/sota_simple_v1/benchmark_gpu.lock'):main()
