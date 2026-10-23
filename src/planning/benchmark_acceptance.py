"""Shared kinematic acceptance, separate from any planner's success flag.

This validates supplied analytic samples, not extrema between samples and not
physical torque or impact. It never repairs or rescales a trajectory.
"""
import numpy as np


def kinematic_gate(position, velocity, acceleration, jerk, *,
                   position_limit=2*np.pi, velocity_limit=1.5,
                   acceleration_limit=4., jerk_limit=30., tolerance=1e-5):
    arrays = [np.asarray(x) for x in (position, velocity, acceleration, jerk)]
    if any(x.ndim != 2 or x.shape != arrays[0].shape or x.size == 0 for x in arrays):
        raise ValueError('Expected nonempty matching [time, joint] analytic arrays')
    finite = all(np.isfinite(x).all() for x in arrays)
    names = ('position', 'velocity', 'acceleration', 'jerk')
    limits = (position_limit, velocity_limit, acceleration_limit, jerk_limit)
    peaks = {k: float(np.max(np.abs(x))) if np.isfinite(x).all() else None
             for k, x in zip(names, arrays)}
    violations = [k for k, limit in zip(names, limits)
                  if peaks[k] is None or peaks[k] > limit+tolerance]
    return dict(passed=bool(finite and not violations), finite=bool(finite),
                absolute_peak=peaks, limits=dict(zip(names, limits)),
                absolute_tolerance=tolerance, violations=violations,
                scope='analytic samples only; not a continuous extrema certificate')


def finish_trial_row(row,collision_audit):
    row={**row,**collision_audit}
    hit=row['collision'];stopped=row['stopped']
    goal_ok=row['terminal_joint_error']<.01 and row['terminal_tcp_m']<.005 and row['terminal_orientation_rad']<.02
    row['failure_flags']=dict(native_failed=not row['native_success'],exact_collision=hit is not None,
        terminal_error=not goal_ok,not_stopped=not stopped,kinematic_limit=not row['kinematic_gate']['passed'],planner_deadline=not row['planner_budget_ok'])
    row['planning_quality_success']=bool(row['native_success'] and hit is None and goal_ok and stopped and row['kinematic_gate']['passed'])
    row['success']=bool(row['planning_quality_success'] and row['planner_budget_ok'])
    row['reason']='success' if row['success'] else 'exact_collision' if hit else 'terminal_error' if not goal_ok else 'not_stopped' if not stopped else 'kinematic_limit' if not row['kinematic_gate']['passed'] else 'planner_deadline' if not row['planner_budget_ok'] else 'native_failed'
    return row
