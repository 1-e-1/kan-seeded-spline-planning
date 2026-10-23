"""Optional TCP acceleration term inside the same spline optimization problem.

No path post-processing or network changes. The objective uses collocation
differences; independent evaluation uses analytic POE time derivatives instead.
"""
from types import MethodType
import torch
import torch.nn.functional as F


def tcp_acceleration_cost(position,dt):
    """[B,H,L,3] tool positions -> [B,H,1] squared acceleration cost.

The experiment has fixed rest-to-rest boundaries; their true acceleration is
zero. Interior central differences are evaluated at the rollout's actual dt.
"""
    step=dt.reshape(-1,1,1,1).clamp_min(1e-6)
    acceleration=(position[:,2:]-2*position[:,1:-1]+position[:,:-2])/(step*step)
    value=(acceleration*acceleration).sum(dim=(-1,-2)).unsqueeze(-1)
    return F.pad(value,(0,0,1,1))


def tcp_jerk_cost(position,dt,stride=2):
    """Third differences on a fixed wider stencil; proxy, not measured jerk.

The wider stencil reduces cancellation of float32 world-space positions.
Independent benchmark metrics remain analytic POE derivatives.
"""
    if position.shape[1]<=3*stride:raise ValueError('Insufficient collocation points')
    step=dt.reshape(-1,1,1,1).clamp_min(1e-6)*stride
    difference=(position[:,3*stride:]-3*position[:,2*stride:-stride]
                +3*position[:,stride:-2*stride]-position[:,:-3*stride])/step**3
    value=difference.square().sum(dim=(-1,-2)).unsqueeze(-1)
    left=3*stride//2
    return F.pad(value,(0,0,left,3*stride-left))


def add_tcp_acceleration_objective(planner,weight):
    assert weight>=0
    if weight==0:return dict(weight=0.,managers=0)
    seen=set()
    for rollout in planner.trajopt_solver.get_all_rollout_instances(include_optimizer_rollouts=True,include_auxiliary_rollout=True):
        if rollout.config.cost_cfg is None:continue
        for name in ('cost_manager','metrics_cost_manager'):
            manager=getattr(rollout,name)
            if id(manager) in seen:continue
            seen.add(id(manager));original=manager.compute_costs
            def wrapped(self,state,*args,_original=original,**kwargs):
                costs=_original(state,*args,**kwargs)
                costs.add(weight*tcp_acceleration_cost(state.tool_poses.position,state.joint_state.dt),'tcp_acceleration')
                return costs
            manager.compute_costs=MethodType(wrapped,manager)
    return dict(weight=weight,managers=len(seen),semantics='additional squared TCP acceleration objective, same optimizer; collocation differences, fixed rest-to-rest boundaries')
