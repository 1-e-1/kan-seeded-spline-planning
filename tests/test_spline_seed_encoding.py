import numpy as np
from src.planning.spline_seed_encoding import encode_seed


def test_seed_goal_survives_null_space_encoding():
    n=5
    matrix=np.eye(n);matrix[-1]=0
    encoding=dict(path_matrix=matrix,start_matrix=np.zeros(n),goal_matrix=np.zeros(n))
    paths=np.random.default_rng(1).normal(size=(3,n,8))
    goal=np.arange(8)*.1
    encoded=encode_seed(paths,np.zeros(8),goal,encoding)
    np.testing.assert_allclose(encoded[:,:-1],paths[:,:-1])
    np.testing.assert_allclose(encoded[:,-1],np.broadcast_to(goal,(3,8)))
    assert not np.array_equal(paths[:,-1],encoded[:,-1])
