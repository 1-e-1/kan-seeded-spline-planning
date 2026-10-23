"""Duration-normalized smoothness inside the ONE trajectory optimizer.

q''(s)=T^2 q''(t), q'''(s)=T^3 q'''(t), s=t/T. This objective
targets curve shape; increasing duration alone cannot reduce its value.
The network, collision oracle, physical derivative limits and solver remain.
"""
from types import MethodType
import torch
from src.planning.tcp_spline_objective import tcp_acceleration_cost,tcp_jerk_cost


def normalized_smoothness_cost(position,acceleration,jerk,dt,tcp_position=None,
                               jerk_ratio=.002,tcp_ratio=0.,tcp_length_m=.5,tcp_jerk_ratio=0.):
    duration=(position.shape[1]-1)*dt.reshape(-1,1,1).clamp_min(1e-6)
    # Endpoints are fixed constraints, not optimization variables. Scaling by
    # requested displacement gives comparable weights for short/long tasks.
    displacement=(position[:,-1]-position[:,0]).square().sum(-1).clamp_min(.01).detach()[:,None,None]
    cost=duration**4*acceleration.square().sum(-1,keepdim=True)/displacement
    cost=cost+jerk_ratio*duration**6*jerk.square().sum(-1,keepdim=True)/displacement
    if tcp_ratio:
        if tcp_position is None:raise ValueError('TCP positions required')
        cost=cost+tcp_ratio*duration**4*tcp_acceleration_cost(tcp_position,dt)/(tcp_length_m**2)
    if tcp_jerk_ratio:
        if tcp_position is None:raise ValueError('TCP positions required')
        cost=cost+tcp_jerk_ratio*duration**6*tcp_jerk_cost(tcp_position,dt)/(tcp_length_m**2)
    return cost


def add_normalized_smoothness(planner,weight,jerk_ratio,tcp_ratio,tcp_jerk_ratio=0.):
    assert weight>0 and jerk_ratio>=0 and tcp_ratio>=0 and tcp_jerk_ratio>=0
    seen=set()
    for rollout in planner.trajopt_solver.get_all_rollout_instances(include_optimizer_rollouts=True,include_auxiliary_rollout=True):
        if rollout.config.cost_cfg is None:continue
        for name in ('cost_manager','metrics_cost_manager'):
            manager=getattr(rollout,name)
            if id(manager) in seen:continue
            seen.add(id(manager));original=manager.compute_costs
            def wrapped(self,state,*args,_original=original,**kwargs):
                costs=_original(state,*args,**kwargs);joint=state.joint_state
                costs.add(weight*normalized_smoothness_cost(joint.position,joint.acceleration,joint.jerk,
                    joint.dt,state.tool_poses.position,jerk_ratio,tcp_ratio,tcp_jerk_ratio=tcp_jerk_ratio),'normalized_smoothness')
                return costs
            manager.compute_costs=MethodType(wrapped,manager)
    return dict(weight=weight,jerk_ratio=jerk_ratio,tcp_ratio=tcp_ratio,tcp_jerk_ratio=tcp_jerk_ratio,tcp_length_m=.5,managers=len(seen),
                scope='single optimizer shape objective; not output retiming or repair')
