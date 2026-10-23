"""CPU evaluation of the existing native cubic curve, not a new smoother.

Arbitrary timestamps support execution clocks without resampling/retiming
the optimized path. Only moving-start, stationary-goal boundaries supported.
The returned jerk is piecewise constant (C2 cubic), not jerk-continuous.
"""
import numpy as np
from scipy.interpolate import BSpline


class CubicExecutionCurve:
    def __init__(self,knots,start,goal,duration,start_velocity=None,start_acceleration=None):
        start=np.asarray(start,dtype=float);goal=np.asarray(goal,dtype=float)
        knots=np.asarray(knots,dtype=float).reshape(-1,len(start))
        if len(knots)==0:raise ValueError('At least one raw knot required')
        if duration<=0 or not np.isfinite(duration):raise ValueError('Positive finite duration required')
        if start.shape!=goal.shape or not all(np.isfinite(x).all() for x in (start,goal,knots)):
            raise ValueError('Invalid finite joint arrays')
        velocity=np.zeros_like(start) if start_velocity is None else np.asarray(start_velocity,dtype=float)
        acceleration=np.zeros_like(start) if start_acceleration is None else np.asarray(start_acceleration,dtype=float)
        if any(x.shape!=start.shape or not np.isfinite(x).all() for x in (velocity,acceleration)):
            raise ValueError('Invalid initial derivative')
        self.duration=float(duration);interval=self.duration/(len(knots)+4)
        leading=(start[None]+np.array([-1.,0.,1.,2.])[:,None]*interval*velocity
                 +np.array([1/3,-1/6,1/3,11/6])[:,None]*interval**2*acceleration)
        coefficients=np.concatenate((leading,knots[:-1],np.broadcast_to(goal,(4,len(goal)))))
        vector=np.arange(-3,len(coefficients)+1)*interval
        self.spline=BSpline(vector,coefficients,3,extrapolate=False)
        # Differentiate coefficients first. Summing large signed derivative
        # basis values is less stable at exactly constant rest intervals.
        self.derivatives=[self.spline]+[self.spline.derivative(k) for k in (1,2,3)]
        self.domain_end=float(vector[len(coefficients)])

    def evaluate(self,time):
        time=np.asarray(time,dtype=float)
        if not np.isfinite(time).all() or np.any(time<0) or np.any(time>self.duration):
            raise ValueError('Execution timestamp outside curve')
        # Multiplying (T / segments) * segments can round one ULP below T.
        # Clamp only that representation-level discrepancy, not valid motion
        # timestamps; otherwise SciPy correctly refuses to extrapolate and
        # would return NaN exactly at the requested stopping time.
        evaluation_time=np.minimum(time,self.domain_end)
        return {name:self.derivatives[order](evaluation_time) for order,name in
                enumerate(('position','velocity','acceleration','jerk'))}
