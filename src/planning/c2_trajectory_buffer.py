"""One queued curve handoff, with no path repair, blend, or time scaling.

This is an execution scheduling primitive, not a motion planner or safety
oracle. A caller must independently verify a candidate's entire swept path,
limits, payload geometry, and actual state before execution. Kinematic-only.
"""
import numpy as np


class C2TrajectoryBuffer:
    def __init__(self,curve,start_time=0.,position_tolerance=1e-5,
                 velocity_tolerance=1e-5,acceleration_tolerance=1e-4):
        self.curve=curve;self.start_time=float(start_time);self.pending=None
        self.tolerances=dict(position=position_tolerance,velocity=velocity_tolerance,acceleration=acceleration_tolerance)
        if not np.isfinite(self.start_time) or any(not np.isfinite(v) or v<0 for v in self.tolerances.values()):
            raise ValueError('Finite clock and nonnegative finite tolerances required')
        self.last_time=self.start_time

    @staticmethod
    def _sample(curve,start_time,time):
        if not np.isfinite(time):raise ValueError('Finite timestamp required')
        elapsed=time-start_time
        if elapsed<0:raise ValueError('Cannot execute before curve start')
        if elapsed<=curve.duration:return curve.evaluate(elapsed)
        state=curve.evaluate(curve.duration)
        # Only stationary-goal curves are supported by CubicExecutionCurve.
        for key in ('velocity','acceleration','jerk'):state[key]=np.zeros_like(state[key])
        return state

    def reference_at(self,time):
        """Future approved reference, NOT measured state or executed history."""
        if self.pending is not None and time>=self.pending[1]:
            return self._sample(*self.pending,time)
        return self._sample(self.curve,self.start_time,time)

    def schedule(self,candidate,switch_time,ready_time,independently_verified=False):
        """Reject late/unchecked/discontinuous curves; never edit either path."""
        if not independently_verified:return dict(accepted=False,reason='not_verified')
        if not np.isfinite([switch_time,ready_time]).all():return dict(accepted=False,reason='invalid_clock')
        if ready_time>switch_time or ready_time<self.last_time:return dict(accepted=False,reason='late_or_stale')
        if self.pending is not None:return dict(accepted=False,reason='already_queued')
        expected=self.reference_at(switch_time);actual=candidate.evaluate(0.)
        errors={key:float(np.max(abs(actual[key]-expected[key]))) for key in self.tolerances}
        if any(errors[k]>tol for k,tol in self.tolerances.items()):
            return dict(accepted=False,reason='boundary_mismatch',errors=errors)
        self.pending=(candidate,float(switch_time))
        return dict(accepted=True,reason='queued',errors=errors)

    def advance(self,time):
        if not np.isfinite(time) or time<self.last_time:raise ValueError('Monotonic finite execution clock required')
        if self.pending is not None and time>=self.pending[1]:
            self.curve,self.start_time=self.pending;self.pending=None
        self.last_time=float(time)
        return self._sample(self.curve,self.start_time,time)
