"""Geometry-preserving C3 retiming; no learned weights or scene dependencies."""
from dataclasses import dataclass, asdict
import numpy as np


def delta(b, a):
    return (np.asarray(b, dtype=np.float64)-np.asarray(a, dtype=np.float64)+np.pi)%(2*np.pi)-np.pi


@dataclass(frozen=True)
class MotionLimits:
    dt: float = 1/30
    velocity: float = 1.5       # per joint rad/s, NOT hardware-certified limits
    acceleration: float = 4.0 # rad/s^2
    jerk: float = 30.0        # rad/s^3

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError('All motion limits must be finite and positive')


def unwrap(q):
    q = np.asarray(q, np.float64)
    if q.ndim != 2 or len(q) < 1 or not np.isfinite(q).all():
        raise ValueError('Expected finite [T>=1, DoF] positions')
    return np.concatenate((q[:1], q[:1]+np.cumsum(delta(q[1:], q[:-1]), axis=0)))


def smoothness(q, dt=1/30):
    """Periodic differences; include the start, rest-to-rest boundary derivatives.

    Three repeated boundary positions make startup and stopping part of metrics.
    Aggregation is per-joint RMS and max absolute joint component, not L2 norm.
    """
    q = unwrap(q)
    padded = np.concatenate((np.repeat(q[:1], 3, 0), q, np.repeat(q[-1:], 3, 0)))
    out = {'duration_s': (len(q)-1)*dt, 'samples': len(q),
           'joint_path_length_rad': float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum())}
    for order, name in enumerate(('velocity', 'acceleration', 'jerk'), 1):
        d = np.diff(padded, n=order, axis=0)/dt**order
        out[name+'_rms'] = float(np.sqrt(np.mean(d*d)))
        out[name+'_max'] = float(np.max(np.abs(d)))
        out[name+'_energy'] = float(np.sum(d*d)*dt)
    return out


def retime_c3(positions, limits=MotionLimits()):
    """Septic blend per certified edge: s=35u^4-84u^5+70u^6-20u^7.

    Zero velocity/acceleration/jerk at every knot; exactly the original polyline,
    no corner cutting. Conservative analytic derivative maxima 2.1875, 7.514,
    52.5 determine durations. A finite-difference check also includes stops.
    Costs: stop at non-collinear knots, longer motion. Not a dynamics controller.
    """
    path = unwrap(positions)
    path = path[np.r_[True, np.max(np.abs(np.diff(path, axis=0)), axis=1)>1e-10]]
    result = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        size = float(np.max(np.abs(b-a)))
        duration = max(2.1875*size/limits.velocity,
                       np.sqrt(7.514*size/limits.acceleration),
                       np.cbrt(52.5*size/limits.jerk))
        steps = max(4, int(np.ceil(duration/limits.dt)))
        u = np.arange(1, steps+1, dtype=np.float64)/steps
        s = u**4*(35+u*(-84+u*(70-20*u)))
        result.extend(a+s[:,None]*(b-a))
    result = np.asarray(result)
    stats = smoothness(result, limits.dt)
    for name in ('velocity', 'acceleration', 'jerk'):
        if stats[name+'_max'] > getattr(limits, name)*(1+1e-6):
            raise RuntimeError(f'Retiming violated {name} limit')
    return result


def compact_collinear(positions, tolerance=2e-5):
    """Remove numerical densification only; subsequent geometric audit mandatory."""
    p = unwrap(positions)
    if len(p) < 3:
        return p
    keep = [0]
    for i in range(1, len(p)-1):
        a, b = p[i]-p[keep[-1]], p[i+1]-p[i]
        scale = np.linalg.norm(a)*np.linalg.norm(b)
        if scale > 1e-12 and (np.dot(a,b) <= 0 or
                np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b)) > tolerance):
            keep.append(i)
    return p[keep+[len(p)-1]]
