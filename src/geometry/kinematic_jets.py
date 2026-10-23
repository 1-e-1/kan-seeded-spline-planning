"""Analytic TCP time derivatives from joint q/v/a/j for the frozen POE model.

No finite differences of sampled positions, force model, or trajectory editing.
"""
from math import comb
import numpy as np
from src.geometry.portable_scene import skew


def tcp_jets(model, q, velocity, acceleration, jerk, path=None):
    q, velocity, acceleration, jerk = map(lambda x: np.atleast_2d(np.asarray(x, dtype=float)),
                                         (q, velocity, acceleration, jerk))
    assert q.shape == velocity.shape == acceleration.shape == jerk.shape
    index = model.index[path or model.meta['tool_path']]
    batch = len(q)
    product = np.zeros((4, batch, 4, 4))
    product[0] = np.eye(4)
    for joint in range(model.depth[index]):
        s = model.screws[joint]
        W = skew(s[3:]); W2 = W@W
        theta = q[:, joint]-model.q_ref[joint]
        sine, cosine = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
        E = np.tile(np.eye(4), (batch, 1, 1))
        E[:, :3, :3] += sine*W+(1-cosine)*W2
        V = theta[:, None, None]*np.eye(3)+(1-cosine)*W+(theta[:, None, None]-sine)*W2
        E[:, :3, 3] = V@s[:3]
        S = np.zeros((4, 4)); S[:3, :3] = W; S[:3, 3] = s[:3]
        S2, S3 = S@S, S@S@S
        v, a, j = (x[:, joint, None, None] for x in (velocity, acceleration, jerk))
        derivatives = [E, E@(S*v), E@(S2*v*v+S*a), E@(S3*v*v*v+3*S2*v*a+S*j)]
        new = np.zeros_like(product)
        for order in range(4):
            for k in range(order+1):
                new[order] += comb(order, k)*(product[k]@derivatives[order-k])
        product = new
    return (product@model.home[index])[:, :, :3, 3]


def derivative_metrics(values, dt, vector_norm=False):
    """Integral-normalized RMS; vector RMS for TCP, per-coordinate RMS for joints."""
    values = np.asarray(values, dtype=float)
    duration = (len(values)-1)*dt
    energy = float(np.trapz(np.sum(values*values, axis=1), dx=dt))
    return dict(rms=float(np.sqrt(energy/duration/(1 if vector_norm else values.shape[1]))),
                max=float(np.max(np.linalg.norm(values, axis=1) if vector_norm else abs(values))),
                energy=energy)
