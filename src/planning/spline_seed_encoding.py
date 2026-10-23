"""Fixed linear encoding of geometric seed samples as native spline knots.

Only initialization changes. A quintic progress law gives zero start/stop
velocity and acceleration before projection into the native basis. This is
not a repair or smoother applied to the optimizer's returned trajectory.
"""
import numpy as np


def build_encoding(position_basis,start_response,goal_response,phase):
    basis=np.asarray(position_basis,dtype=float)
    n=basis.shape[1];phase=np.asarray(phase,dtype=float)
    sample_grid=np.linspace(0,1,n)
    interpolation=np.zeros((len(phase),n))
    upper=np.clip(np.searchsorted(sample_grid,phase,side='right'),1,n-1)
    fraction=(phase-sample_grid[upper-1])/(sample_grid[upper]-sample_grid[upper-1])
    interpolation[np.arange(len(phase)),upper-1]=1-fraction
    interpolation[np.arange(len(phase)),upper]=fraction
    inverse=np.linalg.pinv(basis,rcond=1e-9)
    return dict(path_matrix=inverse@interpolation,start_matrix=inverse@start_response,
                goal_matrix=inverse@goal_response,rank=int(np.linalg.matrix_rank(basis)))


def encode_seed(paths,start,goal,encoding):
    paths=np.asarray(paths,dtype=float)
    knots=(np.einsum('ij,...jk->...ik',encoding['path_matrix'],paths)
        -encoding['start_matrix'][:,None]*np.asarray(start)
        -encoding['goal_matrix'][:,None]*np.asarray(goal))
    # The interpolation basis ignores the last raw knot, but solve_cspace
    # reads it as the per-seed implicit goal BEFORE interpolation. Preserve
    # that API contract; the null-space minimum-norm solution would be zero.
    knots[...,-1,:]=goal
    return knots
