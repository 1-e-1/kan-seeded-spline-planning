"""One state-only horizon heuristic; not a feasibility or timing guarantee."""
import numpy as np


def moving_horizon_guess(start,goal,velocity,acceleration,safety_factor=1.5,minimum_duration=.9):
    arrays=[np.asarray(x,dtype=float) for x in (start,goal,velocity,acceleration)]
    if any(x.shape!=(8,) or not np.isfinite(x).all() for x in arrays):
        raise ValueError('Four finite eight-joint vectors required')
    if not np.isfinite(safety_factor) or safety_factor<1:raise ValueError('Finite safety factor >=1 required')
    if not np.isfinite(minimum_duration) or minimum_duration<=0:raise ValueError('Positive minimum duration required')
    q,g,v,a=arrays;distance=float(np.max(abs((g-q+np.pi)%(2*np.pi)-np.pi)))
    # Rest-to-rest quintic peak factors give a scale, not a solution. The
    # moving-boundary terms estimate braking time; optimization/verification
    # still enforce all limits and may reject the request.
    times=[1.875*distance/1.5,np.sqrt((10/np.sqrt(3))*distance/4),
           np.cbrt(60*distance/30),2*np.max(abs(v))/4,2*np.max(abs(a))/30,.6]
    return float(max(minimum_duration,safety_factor*max(times)))
