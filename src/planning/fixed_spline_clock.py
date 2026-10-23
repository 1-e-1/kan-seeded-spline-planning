"""Fixed-horizon worker mode for moving boundaries, not output retiming.

The native rest-to-rest utility adds one interpolation interval even when a
knot interval is already exactly on the sampling grid. With nonzero start
derivatives this changes boundary control coefficients, not just speed.
This scoped adapter requires alignment and preserves that optimization clock.
It never modifies the shared cuRobo source or the frozen static benchmark.
"""
from contextlib import contextmanager
import torch
import math


def aligned_duration(duration,quantum):
    """Ceiling on a clock grid, without adding a period for roundoff only."""
    if not all(math.isfinite(v) and v>0 for v in (duration,quantum)):
        raise ValueError('Positive finite duration and quantum required')
    quotient=duration/quantum;nearest=round(quotient)
    if abs(quotient-nearest)<=8*math.ulp(max(1.,abs(quotient))):quotient=float(nearest)
    return max(1,math.ceil(quotient))*quantum


def aligned_bspline_steps(knot_dt,sample_dt,horizon):
    quotient=knot_dt/sample_dt;intervals=torch.round(quotient)
    if not bool(torch.all(torch.isfinite(quotient) & (abs(quotient-intervals)<1e-5) & (intervals>=1))):
        raise ValueError('Fixed knot clock must align with sampling clock; no implicit time change allowed')
    counts=((int(horizon)-1)*intervals.to(torch.int64)).to(torch.int32)+1
    return counts,counts.max()


@contextmanager
def fixed_native_clock(solver,collocation_dt):
    import curobo._src.util.trajectory as utility
    old_step=utility.calculate_traj_steps
    old_min=solver.config.minimum_trajectory_dt;old_max=solver.config.maximum_trajectory_dt
    def steps(raw_dt,interpolation_dt,horizon,nearest_int=False):
        if nearest_int:return aligned_bspline_steps(raw_dt,interpolation_dt,horizon)
        return old_step(raw_dt,interpolation_dt,horizon,nearest_int=nearest_int)
    try:
        solver.config.minimum_trajectory_dt=float(collocation_dt)
        solver.config.maximum_trajectory_dt=float(collocation_dt)
        utility.calculate_traj_steps=steps
        yield
    finally:
        utility.calculate_traj_steps=old_step
        solver.config.minimum_trajectory_dt=old_min;solver.config.maximum_trajectory_dt=old_max
