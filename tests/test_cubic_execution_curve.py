import numpy as np
import pytest
from src.planning.cubic_execution_curve import CubicExecutionCurve


def test_moving_start_stationary_goal():
    rng=np.random.default_rng(82);a,b,v,acc=rng.normal(size=(4,8))
    curve=CubicExecutionCurve(rng.normal(size=(32,8)),a,b,7.2,v,acc)
    states=curve.evaluate([0,7.2])
    np.testing.assert_allclose(states['position'],[a,b],atol=1e-12)
    np.testing.assert_allclose(states['velocity'],[v,np.zeros(8)],atol=1e-12)
    np.testing.assert_allclose(states['acceleration'],[acc,np.zeros(8)],atol=1e-12)


def test_clock_bounds_and_no_extrapolation():
    curve=CubicExecutionCurve(np.zeros((16,8)),np.zeros(8),np.ones(8),3.)
    for value in (-.01,3.001,np.nan):
        with pytest.raises(ValueError):curve.evaluate(value)
    assert curve.evaluate(np.arange(90)/30)['position'].shape==(90,8)


def test_arbitrary_duration_endpoint_roundoff_is_not_extrapolation():
    knots=np.linspace(0,1,32)[:,None]*np.ones((1,8))
    for duration in np.random.default_rng(23).uniform(.01,100,1000):
        curve=CubicExecutionCurve(knots,np.zeros(8),np.ones(8),duration)
        endpoint=curve.evaluate(duration)
        np.testing.assert_allclose(endpoint['position'],1.,atol=1e-12)
        for key in ('velocity','acceleration','jerk'):
            np.testing.assert_allclose(endpoint[key],0.,atol=1e-10)
