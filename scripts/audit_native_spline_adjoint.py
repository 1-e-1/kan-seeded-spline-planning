"""Finite-difference audit of native spline q/v/a/j optimization gradients."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'external/research_packages'),str(ROOT/'external/curobo_v0_8_0')]
import numpy as np,torch,warp as wp
wp.config.kernel_cache_dir=str(ROOT/'results/sota_simple_v1/warp_cache')
import curobo._src.runtime as runtime
runtime.cache_dir=str(ROOT/'results/sota_simple_v1/curobo_cache')
from curobo.types import JointState,DeviceCfg
from curobo._src.types.control_space import ControlSpace
from curobo._src.transition.fns_state_transition import StateFromBSplineKnot
from src.planning.native_spline_sampling import sample_native


def main():
    torch.set_num_threads(4);torch.manual_seed(2026091219)
    device=DeviceCfg();rows=[]
    for n,degree in ((16,3),(32,3),(64,3),(16,5)):
        h=(n+degree+1)*4+1
        transition=StateFromBSplineKnot(device,8,batch_size=1,horizon=h,n_knots=n,
            control_space=ControlSpace(degree),interpolation_steps=4)
        start=JointState.zeros((1,8),device);goal=JointState.zeros((1,8),device)
        goal.position.fill_(.8);goal.dt=torch.tensor([.08],device='cuda')
        idx=torch.zeros(1,device='cuda',dtype=torch.int32);implicit=torch.ones(1,device='cuda',dtype=torch.uint8)
        knots=torch.linspace(0,.8,n,device='cuda')[None,:,None].expand(1,n,8).clone()
        knots+=.02*torch.randn_like(knots)
        def evaluate(value,field):
            state=JointState.zeros((1,h,8),device);state.dt=torch.zeros(1,device='cuda')
            out=transition.forward(start,value,state,start_state_idx=idx,goal_state=goal,goal_state_idx=idx,use_implicit_goal_state=implicit)
            return (getattr(out,field).double()**2).mean(),out
        for field in ('position','velocity','acceleration','jerk'):
            value=knots.clone().requires_grad_(True);loss,out=evaluate(value,field)
            gradient=torch.autograd.grad(loss,value)[0].detach().clone()
            errors=[]
            for _ in range(4):
                direction=torch.randn_like(knots);direction/=torch.linalg.vector_norm(direction)
                actual=float((gradient*direction).sum())
                with torch.no_grad():
                    upper,_=evaluate(knots+.01*direction,field)
                    lower,_=evaluate(knots-.01*direction,field)
                fd=float((upper-lower)/.02)
                errors.append(dict(autograd=actual,finite_difference=fd,relative_error=abs(actual-fd)/max(abs(actual),abs(fd),1e-6)))
            rows.append(dict(knots=n,degree=degree,field=field,checks=errors,gate=max(x['relative_error'] for x in errors)<.005))
        _,out=evaluate(knots,'position')
        # goal.dt in the rollout is the collocation sample interval, NOT the
        # knot interval. With four interpolation steps, knot_dt = 4 * dt.
        sample=sample_native(knots.cpu().numpy(),np.array([.32]),degree,np.zeros(8),np.full(8,.8),.08,duration=(h-1)*.08)
        errors={k:float(np.max(abs(sample[k]-getattr(out,k)[0].detach().cpu().numpy()))) for k in sample}
        rows.append(dict(knots=n,degree=degree,clock_errors=errors,reported_state_dt=out.dt.cpu().numpy().tolist(),gate=max(errors.values())<.005))
    report=dict(gate=all(r['gate'] for r in rows),rows=rows)
    (ROOT/'results/sota_simple_v1/native_spline_adjoint_audit_v2.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(report)


if __name__=='__main__':main()
