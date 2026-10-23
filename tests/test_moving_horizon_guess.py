import numpy as np
import pytest
from src.planning.moving_horizon_guess import moving_horizon_guess


def test_horizon_is_state_only_and_finite():
    z=np.zeros(8);g=z.copy();g[0]=1
    t=moving_horizon_guess(z,g,z,z)
    assert np.isfinite(t) and t>=1.5*np.cbrt(2)
    assert moving_horizon_guess(z,2*g,z,z)>t
    assert moving_horizon_guess(z,z,z,z)>0


def test_horizon_rejects_invalid_inputs():
    z=np.zeros(8)
    with pytest.raises(ValueError):moving_horizon_guess(z,z,z,z,safety_factor=.5)
    with pytest.raises(ValueError):moving_horizon_guess(z,z,z,np.full(8,np.nan))
