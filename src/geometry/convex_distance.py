"""Stateless convex-hull separation with witness points (GJK distance).

Unsigned: overlap returns zero, NOT penetration depth. Queries that fail the
support-gap certificate raise instead of returning a guessed safe distance.
"""
from itertools import combinations
import numpy as np


def _closest_simplex(points):
    best=None
    for size in range(1,min(4,len(points))+1):
        for ids in combinations(range(len(points)),size):
            w=points[list(ids)]
            if size==1:weights=np.ones(1)
            else:
                # Affine least-squares projection with last vertex as origin.
                basis=(w[:-1]-w[-1]).T
                coeff=np.linalg.lstsq(basis,-w[-1],rcond=1e-12)[0]
                weights=np.r_[coeff,1-coeff.sum()]
            if weights.min() < -1e-10:continue
            weights=np.maximum(weights,0);weights/=weights.sum();v=weights@w;norm=float(v@v)
            if best is None or norm<best[0]:best=(norm,ids,weights,v)
    if best is None:raise RuntimeError('invalid_gjk_simplex')
    _,ids,weights,v=best;keep=weights>1e-10
    return np.array(ids)[keep],weights[keep]/weights[keep].sum(),v


def convex_distance(vertices_a,vertices_b,tolerance=1e-9,max_iterations=128):
    a=np.asarray(vertices_a,np.float64);b=np.asarray(vertices_b,np.float64)
    if any(x.ndim!=2 or x.shape[1]!=3 or len(x)==0 or not np.isfinite(x).all() for x in (a,b)):
        raise ValueError('Expected finite nonempty [N,3] convex vertices')
    if tolerance<=0 or max_iterations<1:raise ValueError('Invalid convergence settings')
    # Translation improves conditioning for meshes far from the world origin.
    origin=(a.mean(0)+b.mean(0))/2;a=a-origin;b=b-origin
    direction=a.mean(0)-b.mean(0)
    if np.linalg.norm(direction)<1e-10:direction=np.array([1.,0.,0.])
    def support(v):
        ia=int(np.argmax(a@v));ib=int(np.argmin(b@v));return a[ia],b[ib]
    pa,pb=support(-direction);sa=np.array([pa]);sb=np.array([pb]);v=pa-pb;weights=np.ones(1)
    gap=float('inf')
    for iteration in range(max_iterations):
        norm=float(np.linalg.norm(v))
        if norm<=tolerance:gap=0.;break
        pa,pb=support(-v);w=pa-pb;gap=float(v@v-v@w)
        # Gap / ||v|| bounds the distance optimality error for separated hulls.
        if gap<=tolerance*norm:break
        if np.any(np.linalg.norm((sa-sb)-w,axis=1)<1e-13):
            if gap>1e-8*norm:raise RuntimeError('gjk_duplicate_without_convergence')
            break
        sa=np.vstack((sa,pa));sb=np.vstack((sb,pb))
        ids,weights,v=_closest_simplex(sa-sb);sa=sa[ids];sb=sb[ids]
        v=weights@(sa-sb)
    else:raise RuntimeError('gjk_iteration_limit')
    point_a=weights@sa+origin;point_b=weights@sb+origin;distance=float(np.linalg.norm(point_a-point_b))
    return {'distance':distance,'point_a':point_a,'point_b':point_b,'normal':(point_a-point_b)/distance if distance>tolerance else np.zeros(3),
            'iterations':iteration+1,'distance_error_bound':max(0,gap)/max(distance,tolerance)}
