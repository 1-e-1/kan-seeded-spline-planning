"""Resample saved cuRobo curves through their original analytic CUDA basis.

This module changes neither a path nor its timing. It is an evaluation adapter,
not trajectory post-processing. Caller sets up the isolated cuRobo dependency.
"""
import numpy as np
import torch
from curobo.types import JointState, DeviceCfg
from curobo._src.types.control_space import ControlSpace
from curobo._src.util.trajectory import get_batch_interpolated_trajectory, TrajInterpolationType
from curobo._src.curobolib.cuda_ops.trajectory import get_bspline_interpolation


def sample_native(knots, knot_dt, degree, start, goal, dt, implicit_goal=True, duration=None,
                  start_velocity=None, start_acceleration=None, start_jerk=None,
                  goal_velocity=None, goal_acceleration=None, goal_jerk=None):
    device = DeviceCfg()
    tensor = lambda x: torch.as_tensor(np.asarray(x), device=device.device, dtype=torch.float32).contiguous()
    knots = tensor(knots).reshape(1, -1, len(start))
    raw = JointState.from_position(knots)
    raw.knot = knots
    raw.knot_dt = tensor(knot_dt).reshape(1)
    raw.control_space = ControlSpace(int(degree))
    a = JointState.zeros((1, len(start)), device)
    b = JointState.zeros((1, len(start)), device)
    a.position.copy_(tensor(start).reshape_as(a.position))
    b.position.copy_(tensor(goal).reshape_as(b.position))
    for state,values in ((a,(start_velocity,start_acceleration,start_jerk)),
                         (b,(goal_velocity,goal_acceleration,goal_jerk))):
        for name,value in zip(('velocity','acceleration','jerk'),values):
            if value is not None:
                if np.asarray(value).shape != (len(start),) or not np.isfinite(value).all():
                    raise ValueError(f'Invalid {name} boundary; expected finite joint vector')
                getattr(state,name).copy_(tensor(value).reshape_as(state.position))
    idx = torch.zeros(1, device=device.device, dtype=torch.int32)
    implicit=torch.full((1,), int(implicit_goal), device=device.device, dtype=torch.uint8)
    if duration is None:
        out, steps = get_batch_interpolated_trajectory(
            raw, tensor([dt]), kind=TrajInterpolationType.BSPLINE_KNOTS_CUDA,
            device_cfg=device, current_state=a, goal_state=b, start_idx=idx, goal_idx=idx,
            use_implicit_goal_state=implicit)
        count = int(steps[0])
    else:
        # Native utility rounds each knot interval UP to an integer number of
        # sampling steps, silently changing duration at every requested dt.
        # The direct basis kernel preserves the already-executed clock instead.
        intervals=int(round(float(duration)/dt));segments=knots.shape[-2]+int(degree)+1
        assert intervals%segments==0 and abs(intervals*dt-duration)<1e-6
        count=intervals+1
        out=JointState.zeros((1,count,len(start)),device)
        get_bspline_interpolation(raw,out,tensor([dt]),a,b,idx,idx,implicit,
            torch.tensor([intervals],device=device.device,dtype=torch.int32),bspline_degree=int(degree))
    return {k: getattr(out, k)[0, :count].detach().cpu().numpy().copy()
            for k in ('position', 'velocity', 'acceleration', 'jerk')}
